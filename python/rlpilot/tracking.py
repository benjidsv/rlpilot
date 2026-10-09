"""MLflow integration. Metric names are grouped by prefix, which is how the MLflow UI groups charts:

Tier 1 (dashboard)
  eval/lap_time_best, eval/lap_time_median   deterministic policy over a few laps; plotted with
  eval/bot_lap_time_best, ..._median         the built-in bot's lap times on the same car and track
  eval/flying_lap_time_best, ..._median      the same without lap 1 (standing start, ~4 s slower)
  eval/completion_rate, eval/progress        fraction of eval episodes finishing a lap; distance / track length
  eval/fail_dist_from_start, eval/fail_lap   median position along the lap (m) / mean lap where failed eval episodes ended
  (eval/* is the first training track.) With several tracks, per track and averaged over training / held-out tracks:
  eval_track/<track>/{progress,completion_rate,flying_gap_pct,fail_dist_from_start},
  eval_train/*, eval_heldout/* (flying_gap_pct: median flying lap vs the bot's best, in %)
  episode/progress                           training episodes: distance / track length
  end/{off_track,stuck_crashed,timeout,lap_done}   why training episodes end (fractions)
  reward/{progress,offtrack,damage,terminal,total} per-episode reward components
  ppo/{approx_kl,clip_fraction,entropy,explained_variance}
Tier 2 (behavior / perf)
  behavior/{steer_delta,offtrack_frac,mean_speed,full_throttle_frac}, eval/behavior/*
  behavior/line_dev   mean |offset from the bot's racing line| (m); behavior/abs_offset: mean |to_middle| / half
                      width (0 = centre line, 1 = on the edge), vs eval_track/<t>/bot_abs_offset (the bot's own)
  perf/{env_steps_per_s,total_steps_per_s,reset_share}
Tier 3 (artifacts, at each eval)
  eval/<timesteps>/track_map.png, speed_vs_distance.png, traces.npz
  best/ (model + VecNormalize stats of the best eval in the best window of 3 evals; tag best_step),
  checkpoints/<timesteps>/
Everything else SB3 logs is kept under sb3/. Nothing here runs per env step.
"""

from __future__ import annotations

import dataclasses
import shutil
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
from .evaluate import rollout, summarize
from .reward import END_GROUPS, REWARD_COMPONENTS
from .track import RacingLine, reference_for, track_of
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


class LineShapingFadeCallback(BaseCallback):
    """Fades the racing-line shaping coefficient linearly from `k` to 0 at `fade` x total timesteps (fade 0: keep k),
    so the end of training is unbiased by the bot's line. Set before every rollout; logged as line/shaping_k."""

    def __init__(self, k: float, total_timesteps: int, fade: float):
        super().__init__()
        self.k, self.end = k, fade * total_timesteps

    def _on_rollout_start(self):
        k = self.k * max(0.0, 1.0 - self.num_timesteps / self.end) if self.end > 0 else self.k
        _base_env(self.training_env)[0].set_line_shaping(k)
        mlflow.log_metric("line/shaping_k", k, step=self.num_timesteps)

    def _on_step(self) -> bool:
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
            for k in ("steer_delta", "offtrack_frac", "mean_speed", "full_throttle_frac", "line_dev", "abs_offset"):
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
    """Every `every` timesteps (and at the end): deterministic eval over `laps` laps with `episodes` envs per track,
    Tier 1 eval metrics, and the track map / speed trace artifacts.

    All evaluated tracks (the training tracks to evaluate, then the `heldout` races, never trained on) run at once
    in one env kept between evals: worker i drives track i % n_tracks, so an eval lasts about as long as its longest
    track rather than the sum of all of them. The first training track gets the eval/* metrics and artifacts; with
    several tracks, per-track and averaged metrics go to eval_track/, eval_train/, eval_heldout/.

    best/ holds the model of the best-scoring eval within the best window of `smooth` consecutive evals (by mean
    score, logged as eval/score_smoothed), so that one lucky eval does not win. Each eval's model is kept locally
    until it leaves the window."""

    def __init__(self, cfg: EnvConfig, every: int = 100_000, episodes: int = 8, laps: int = 3,
                 launch_noise: float = 0.1, verbose: int = 0, heldout: tuple[str, ...] = (),
                 eval_train: tuple[str, ...] = (), smooth: int = 3):
        super().__init__(verbose)
        # training tracks to evaluate (default: all of them); the first one gets eval/* and the artifacts
        self.train_races = tuple(eval_train) or tuple(cfg.races) or (cfg.race,)
        self.heldout = tuple(r for r in heldout if r not in self.train_races)
        self.races = self.train_races + self.heldout
        # evaluated on the stock car (factor observations, if any, read 0). Workers never change track: no
        # track switching, and no recycling (whose respawn would draw a random track).
        self.cfg = dataclasses.replace(cfg, n_workers=episodes * len(self.races), laps_per_episode=laps,
                                       run_dir=None, race=self.races[0],
                                       races=self.races if len(self.races) > 1 else (),
                                       track_switch_steps=2**62, recycle_resets=2**62,
                                       car_grip=(1.0, 1.0), car_mass=(1.0, 1.0), car_power=(1.0, 1.0))
        self.every, self.launch_noise, self.smooth = every, launch_noise, max(1, smooth)
        self._last = 0
        self.best_score = -np.inf
        self._window: list[tuple[int, float, Path]] = []  # (step, score, local model snapshot) of the last evals

    def _on_training_start(self):
        self.refs = {r: reference_mod.load(reference_for(r), laps=3) for r in self.races}
        self.ref = self.refs[self.train_races[0]]
        mlflow.log_params({"eval.episodes": self.cfg.n_workers // len(self.races),
                           "eval.laps": self.cfg.laps_per_episode, "eval.smooth": self.smooth,
                           "eval.launch_noise": self.launch_noise, "eval.bot_race": reference_for(self.train_races[0]),
                           "eval.bot_lap_time_best": round(self.ref.best_lap, 3),
                           "eval.train_tracks": ",".join(track_of(r) for r in self.train_races),
                           "eval.heldout_tracks": ",".join(track_of(r) for r in self.heldout)})
        self.env = SpeedDreamsVecEnv(self.cfg)
        n_tracks = len(self.races)
        mlflow.log_metrics({f"eval_track/{track_of(r)}/bot_abs_offset": RacingLine.for_race(r).abs_offset
                            for r in self.races}, step=0)
        assert list(self.env.client.races) == [self.races[i % n_tracks] for i in range(self.env.num_envs)]
        # time for `laps` laps at 70% of the bot's pace, plus the standing start
        self._limits = np.array([self.cfg.laps_per_episode * self.refs[self.races[i % n_tracks]].best_lap / 0.7 + 20.0
                                 for i in range(self.env.num_envs)])
        self._tmp = tempfile.TemporaryDirectory(prefix="rlpilot-eval-")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self.run()
        return True

    def _on_training_end(self):
        if self.num_timesteps != self._last:
            self.run()
        self.env.close()
        self._tmp.cleanup()

    def run(self):
        norm = self.training_env if isinstance(self.training_env, VecNormalize) else None

        def policy(obs):
            if norm is not None and norm.norm_obs:
                obs = norm.normalize_obs(obs)
            return self.model.predict(obs, deterministic=True)[0]

        n_tracks, n = len(self.races), self.env.num_envs
        traces, lap_times, stats = rollout(policy, self.env, self.launch_noise, clean=range(n_tracks),
                                           time_limits=self._limits)
        results = {}
        for t, race in enumerate(self.races):
            idx = range(t, n, n_tracks)  # the first one (env t) is the clean episode
            results[race] = summarize([traces[i] for i in idx], [lap_times[i] for i in idx], [stats[i] for i in idx],
                                      self.refs[race])
        res = results[self.train_races[0]]
        step = self.num_timesteps
        mlflow.log_metrics(res.metrics, step=step)

        # Score: pace first, with a moderate reliability penalty. Minus the median lap time (150 s if no lap
        # finished) minus 10 s per unit of non-completion (no lap finished) and 10 s per lap of missing progress.
        # With several tracks: the mean over the evaluated training and held-out tracks, with lap times in % of
        # the bot's best lap (~1 s per % on Jarama), so long and short tracks weigh the same. Held-out tracks
        # are never trained on but do select the checkpoint, so their metrics at best/ are optimistic.
        m = res.metrics
        short = max(0.0, self.cfg.laps_per_episode - m["eval/progress"])
        score = -(m.get("eval/lap_time_median", 150.0) + 10.0 * (1 - m["eval/completion_rate"]) + 10.0 * short)
        if n_tracks > 1:
            agg, scores = self._track_metrics({r: x.metrics for r, x in results.items()})
            mlflow.log_metrics(agg, step=step)
            score = float(np.mean(scores))
        self._select_best(step, score)

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

    def _select_best(self, step: int, score: float) -> None:
        snap = Path(self._tmp.name) / f"{step:012d}"
        _save_local(self.model, snap)
        self._window.append((step, score, snap))
        if len(self._window) > self.smooth:
            shutil.rmtree(self._window.pop(0)[2], ignore_errors=True)
        smoothed = float(np.mean([s for _, s, _ in self._window]))
        mlflow.log_metrics({"eval/score": score, "eval/score_smoothed": smoothed}, step=step)
        if smoothed > self.best_score:
            self.best_score = smoothed
            best_step, best, best_snap = max(self._window, key=lambda w: w[1])
            mlflow.log_artifacts(str(best_snap), "best")  # overwrites the previous best in the run's artifacts
            mlflow.log_metrics({"best/score": smoothed, "best/step": best_step, "best/eval_score": best}, step=step)
            mlflow.set_tag("best_step", str(best_step))

    def _track_metrics(self, per_track: dict[str, dict[str, float]]) -> tuple[dict[str, float], list[float]]:
        out, groups, scores = {}, {"eval_train": [], "eval_heldout": []}, []
        for race, m in per_track.items():
            bot = self.refs[race].best_lap
            t = {"progress": m["eval/progress"], "completion_rate": m["eval/completion_rate"],
                 "line_dev": m["eval/behavior/line_dev"], "abs_offset": m["eval/behavior/abs_offset"]}
            if "eval/flying_lap_time_median" in m:
                t["flying_gap_pct"] = 100.0 * (m["eval/flying_lap_time_median"] / bot - 1)
            if "eval/fail_dist_from_start" in m:
                t["fail_dist_from_start"] = m["eval/fail_dist_from_start"]
            out |= {f"eval_track/{track_of(race)}/{k}": v for k, v in t.items()}
            groups["eval_train" if race in self.train_races else "eval_heldout"].append(t)
            short = max(0.0, self.cfg.laps_per_episode - m["eval/progress"])
            lap_pct = 100.0 * m["eval/lap_time_median"] / bot if "eval/lap_time_median" in m else 140.0
            scores.append(-(lap_pct + 10.0 * (1 - m["eval/completion_rate"]) + 10.0 * short))
        for g, ts in groups.items():
            for k in ("progress", "completion_rate", "flying_gap_pct", "line_dev", "abs_offset"):
                vals = [t[k] for t in ts if k in t]
                if vals:
                    out[f"{g}/{k}"] = float(np.mean(vals))
            if ts:  # tracks where at least one flying lap was completed
                out[f"{g}/tracks_with_flying_lap"] = float(np.mean(["flying_gap_pct" in t for t in ts]))
        return out, scores


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


def _save_local(model, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    model.save(d / "model.zip")
    env = model.get_env()
    if isinstance(env, VecNormalize):
        env.save(str(d / "vecnormalize.pkl"))


def save_model(model, artifact_path: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        _save_local(model, Path(d))
        mlflow.log_artifacts(d, artifact_path)
