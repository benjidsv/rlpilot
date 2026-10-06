"""MLflow integration. Metric names are grouped by prefix, which is how the MLflow UI groups charts:

Tier 1 (dashboard)
  eval/lap_time_best, eval/lap_time_median   deterministic policy over a few laps; plotted with
  eval/bot_lap_time_best, ..._median         the built-in bot's lap times on the same car and track
  eval/completion_rate, eval/progress        fraction of eval episodes finishing a lap; distance / track length
  episode/progress                           training episodes: distance / track length
  end/{off_track,stuck_crashed,timeout,lap_done}   why training episodes end (fractions)
  reward/{progress,offtrack,damage,terminal,total} per-episode reward components
  ppo/{approx_kl,clip_fraction,entropy,explained_variance}
Tier 2 (behavior / perf)
  behavior/{steer_delta,offtrack_frac,mean_speed,full_throttle_frac}, eval/behavior/*
  perf/{env_steps_per_s,total_steps_per_s,reset_share}
Tier 3 (artifacts, at each eval)
  eval/<timesteps>/track_map.png, speed_vs_distance.png, traces.npz
  best/ (model + VecNormalize stats of the best eval so far; tag best_step), checkpoints/<timesteps>/
Everything else SB3 logs is kept under sb3/. Nothing here runs per env step.
"""

from __future__ import annotations

import dataclasses
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.vec_env import VecNormalize

from . import plots, workdirs
from . import reference as reference_mod
from ._native import PROTOCOL_VERSION
from .config import EnvConfig
from .evaluate import evaluate
from .reward import END_GROUPS, REWARD_COMPONENTS
from .vec_env import SpeedDreamsVecEnv


def _git(*args: str, cwd=workdirs.ROOT) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_provenance(cfg: EnvConfig, **extra_params) -> None:
    """Log the env config as params and code versions as tags on the active run."""
    mlflow.log_params({f"env.{k}": v for k, v in cfg.flat().items()} | extra_params)
    engine = workdirs.ROOT / "speed-dreams-code"
    mlflow.set_tags({
        "rlpilot.git_sha": _git("rev-parse", "HEAD"),
        "rlpilot.git_dirty": str(bool(_git("status", "--porcelain"))),
        "engine.git_sha": _git("rev-parse", "HEAD", cwd=engine),
        "protocol_version": str(PROTOCOL_VERSION),
    })


# SB3 key -> (dashboard name, sign)
_PPO_HEALTH = {"train/approx_kl": ("ppo/approx_kl", 1), "train/clip_fraction": ("ppo/clip_fraction", 1),
               "train/entropy_loss": ("ppo/entropy", -1), "train/explained_variance": ("ppo/explained_variance", 1)}


class MLflowOutputFormat(KVWriter):
    """Forwards SB3 logger records to MLflow: the four PPO health numbers under ppo/, the rest under sb3/."""

    def write(self, key_values, key_excluded, step=0):
        metrics = {}
        for k, v in key_values.items():
            if not isinstance(v, (int, float, np.integer, np.floating)) or isinstance(v, bool):
                continue
            name, sign = _PPO_HEALTH.get(k, ("sb3/" + k, 1))
            metrics[name] = sign * float(v)
        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def close(self):
        pass


def _base_env(training_env) -> tuple[SpeedDreamsVecEnv, object]:
    """(SpeedDreamsVecEnv, SB3VecEnv) under any SB3 VecEnv wrappers."""
    venv = training_env
    while not hasattr(venv, "env_seconds"):
        venv = venv.venv
    return venv.env, venv


class StdClampCallback(BaseCallback):
    """PPO only: keeps the Gaussian policy's action std inside [min_std, max_std] and logs the std of each action
    dimension (`policy/std_<name>`) every `every` timesteps. Without a cap the entropy bonus can grow the std
    without bound once the (clipped) actions saturate: pilot-4 (shifting, ent_coef 0.01) reached std 127.
    The clamp runs before every rollout, i.e. right after each update."""

    def __init__(self, names: tuple[str, ...], min_std: float = 0.135, max_std: float = 1.0, every: int = 20_000):
        super().__init__()
        self.names, self.every, self._last = names, every, 0
        self.lo, self.hi = float(np.log(min_std)), float(np.log(max_std))

    def _on_rollout_start(self):
        import torch
        with torch.no_grad():
            self.model.policy.log_std.clamp_(self.lo, self.hi)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            std = self.model.policy.log_std.detach().exp().cpu().numpy()
            mlflow.log_metrics({f"policy/std_{n}": float(v) for n, v in zip(self.names, std)}, step=self.num_timesteps)
        return True


class TrainingMetricsCallback(BaseCallback):
    """Tier 1/2 training metrics from ended episodes, logged at most every `every` timesteps."""

    def __init__(self, every: int = 20_000, window: int = 200):
        super().__init__()
        self.every, self.buf = every, deque(maxlen=window)
        self._last = 0

    def _on_training_start(self):
        _, self._sb3env = _base_env(self.training_env)
        self._t = time.perf_counter()
        self._env_s, self._env_n, self._steps = 0.0, 0, self.num_timesteps
        self._reset_s = _base_env(self.training_env)[0].client.stats()["worker_reset_seconds"]

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "stats" in info:
                self.buf.append(info["stats"])
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self._log()
        return True

    def _log(self):
        m = {}
        if self.buf:
            b = list(self.buf)
            m["episode/progress"] = float(np.mean([s["progress"] for s in b]))
            m["episode/laps"] = float(np.mean([s["laps"] for s in b]))
            groups = [END_GROUPS[s["end"]] for s in b]
            for g in sorted(set(END_GROUPS.values())):
                m[f"end/{g}"] = groups.count(g) / len(b)
            total = 0.0
            for c in REWARD_COMPONENTS:
                m[f"reward/{c}"] = float(np.mean([s[f"reward_{c}"] for s in b]))
                total += m[f"reward/{c}"]
            m["reward/total"] = total
            for k in ("steer_delta", "offtrack_frac", "mean_speed", "full_throttle_frac"):
                m[f"behavior/{k}"] = float(np.mean([s[k] for s in b]))
            self.buf.clear()
        env, sb3env = _base_env(self.training_env)
        now = time.perf_counter()
        wall, n = now - self._t, self.num_timesteps - self._steps
        env_s, env_n = sb3env.env_seconds - self._env_s, sb3env.env_steps - self._env_n
        reset_s = env.client.stats()["worker_reset_seconds"]
        if env_s > 0:
            m["perf/env_steps_per_s"] = env_n / env_s
        if wall > 0:
            m["perf/total_steps_per_s"] = n / wall
            m["perf/reset_share"] = (reset_s - self._reset_s) / (env.num_envs * wall)
        self._t, self._steps, self._env_s, self._env_n, self._reset_s = now, self.num_timesteps, sb3env.env_seconds, \
            sb3env.env_steps, reset_s
        mlflow.log_metrics(m, step=self.num_timesteps)


class EvalCallback(BaseCallback):
    """Every `every` timesteps (and at the end): deterministic eval over `laps` laps on `episodes` envs,
    Tier 1 eval metrics, and the track map / speed trace artifacts."""

    def __init__(self, cfg: EnvConfig, every: int = 100_000, episodes: int = 8, laps: int = 3,
                 launch_noise: float = 0.1, verbose: int = 0):
        super().__init__(verbose)
        self.best_score = -np.inf
        self.cfg = dataclasses.replace(cfg, n_workers=episodes, laps_per_episode=laps, run_dir=None,
                                       max_episode_seconds=min(400.0, 140.0 * laps))
        self.every, self.launch_noise = every, launch_noise
        self._last = 0

    def _on_training_start(self):
        self.ref = reference_mod.load(self.cfg.reference_race, laps=3)
        mlflow.log_params({"eval.episodes": self.cfg.n_workers, "eval.laps": self.cfg.laps_per_episode,
                           "eval.launch_noise": self.launch_noise, "eval.bot_race": self.cfg.reference_race,
                           "eval.bot_lap_time_best": round(self.ref.best_lap, 3)})
        self.env = SpeedDreamsVecEnv(self.cfg)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self.run()
        return True

    def _on_training_end(self):
        if self.num_timesteps != self._last:
            self.run()
        self.env.close()

    def run(self):
        norm = self.training_env if isinstance(self.training_env, VecNormalize) else None

        def policy(obs):
            if norm is not None and norm.norm_obs:
                obs = norm.normalize_obs(obs)
            return self.model.predict(obs, deterministic=True)[0]

        res = evaluate(policy, self.env, self.ref, launch_noise=self.launch_noise)
        step = self.num_timesteps
        mlflow.log_metrics(res.metrics, step=step)

        # Best checkpoint by eval: pace first, with a moderate reliability penalty. Score is
        # minus the median lap time (150 s if no lap finished) minus 10 s per unit of
        # non-completion (no lap finished) and 10 s per lap of missing progress.
        m = res.metrics
        short = max(0.0, self.cfg.laps_per_episode - m["eval/progress"])
        score = -(m.get("eval/lap_time_median", 150.0) + 10.0 * (1 - m["eval/completion_rate"]) + 10.0 * short)
        mlflow.log_metric("eval/score", score, step=step)
        if score > self.best_score:
            self.best_score = score
            save_model(self.model, "best")  # overwrites the previous best in the run's artifacts
            mlflow.log_metrics({"best/score": score, "best/step": step}, step=step)
            mlflow.set_tag("best_step", str(step))

        lap_trace, lap = res.best_lap_trace(0)
        # compare like with like: lap 1 starts from the grid, later laps are flying laps
        bot_lap = (lap or 1) if (lap or 1) <= len(self.ref.lap_times) else None
        bot_trace = self.ref.lap_trace(bot_lap)
        label = f"lap {lap}, {res.lap_times[0][lap - 1]:.2f} s" if lap else "no lap completed (lap 1, partial)"
        with tempfile.TemporaryDirectory() as d:
            fig = plots.track_map(self.ref, lap_trace if lap else res.traces[0],
                                  f"Eval at {step:,} steps: {label}")
            fig.savefig(Path(d) / "track_map.png")
            fig2 = plots.speed_vs_distance(lap_trace, bot_trace,
                                           f"Speed vs distance at {step:,} steps: agent {label}, bot lap "
                                           f"{bot_lap or 'best'} {self.ref.lap_times[(bot_lap or 0) - 1 if bot_lap else self.ref.lap_times.argmin()]:.2f} s")
            fig2.savefig(Path(d) / "speed_vs_distance.png")
            plt.close(fig)
            plt.close(fig2)
            np.savez_compressed(Path(d) / "traces.npz",
                                **{f"ep{i}_{k}": v for i, tr in enumerate(res.traces) for k, v in tr.items()})
            mlflow.log_artifacts(d, f"eval/{step:010d}")
        if self.verbose:
            print({k: round(v, 3) for k, v in res.metrics.items()})


class CheckpointCallback(BaseCallback):
    """Every `every` timesteps: save the model and VecNormalize stats to checkpoints/<timesteps>/ in the
    run's artifacts (a hard kill or power loss then loses at most `every` steps)."""

    def __init__(self, every: int = 1_000_000):
        super().__init__()
        self.every, self._last = every, 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            save_model(self.model, f"checkpoints/{self.num_timesteps:010d}")
        return True


def save_model(model, artifact_path: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        model.save(Path(d) / "model.zip")
        env = model.get_env()
        if isinstance(env, VecNormalize):
            env.save(str(Path(d) / "vecnormalize.pkl"))
        mlflow.log_artifacts(d, artifact_path)
