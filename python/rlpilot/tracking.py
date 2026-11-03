"""MLflow integration. The UI makes one flat section per first path segment, sorted by name, so the prefix is the folder:

dashboard/         what to watch over a whole run, and only here:
  {train,heldout}_{completion,progress,gap_pct}, heldout_flying_lap_tracks   evals, averaged per track group. gap_pct:
                   median flying lap vs the bot's best, in %, over the tracks that have a flying lap
  score_smoothed   what selects best/ (tag best_step)
  rollout_{progress,reward,steer_delta,off_track,crashed}   training episodes; off_track, crashed: how they end (fractions)
  ppo_{approx_kl,entropy,explained_variance}, steps_per_s
driving/           training episodes: {steer_reversal_frac,steer_limit_frac,steer_lock_frac,steer_abs,pedal_delta,
                   offtrack_frac,mean_speed,full_throttle_frac,line_dev,abs_offset}
                   (line_dev: m from the bot's line; abs_offset: mean |to_middle| / half width, 0 = centre, 1 = edge)
eval/              {flying_lap_time_best,flying_lap_time_median,driving/*}: the first training track only
                   {train,heldout}_{line_dev,abs_offset}, train_flying_lap_tracks, score
obs/               <group>: largest |value| the policy received, nonfinite, dead_slots. Every 1M steps; min, max, abs_mean
                   and std per group are in the artifact obs_stats.csv
perf/              {env_steps_per_s,reset_share,tracks_covered,min_track_samples} (worst update since the last log)
ppo/               {clip_fraction,value_loss,policy_gradient_loss,learning_rate,std_<action>}
reward/<component> per-episode, only the components with a non-zero weight (the total is on the dashboard)
track_progress/<track>, track_gap_pct/<track>, track_fail_dist/<track>   one folder per metric, one chart per track.
                   fail_dist: median position along the lap (m) where failed eval episodes ended
Artifacts:
  eval/<timesteps>/{track_map.png,speed_vs_distance.png,traces.npz}, tracks/assignment_{start,end}.json (worker -> race),
  tracks/bot_abs_offset.json, best/ (model + VecNormalize stats of the best eval in the best window of 3 evals),
  checkpoints/<timesteps>/
Nothing here runs per env step. summarize() computes more than is logged: scripts and the score read the rest.
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
from .evaluate import eval_config, eval_time_limits, rollout, summarize
from .reward import END_GROUPS, REWARD_COMPONENTS
from .track import RacingLine, reference_for, track_of
from .vec_env import SpeedDreamsVecEnv


SCORE_GAP_WEIGHT = 0.02  # score per % of pace: 5% is worth 0.1 of a track's completion
SCORE_GAP_CAP = 10.0     # %, also what a track without a flying lap counts as

_REWARD_WEIGHT = {"steer": "steer_rate", "line": "line_shaping"}  # reward component -> RewardConfig field, where they differ
DRIVING_STATS = ("steer_reversal_frac", "steer_limit_frac", "steer_lock_frac", "steer_abs", "pedal_delta", "offtrack_frac",
                 "mean_speed", "full_throttle_frac", "line_dev", "abs_offset")
EVAL_DRIVING_STATS = ("steer_delta", "steer_reversal_frac", "pedal_delta", "offtrack_frac", "mean_speed", "full_throttle_frac")


def track_score(m: dict[str, float], bot_lap: float, laps: int) -> float:
    """Checkpoint score on one track: completion first, pace second. The share of the eval's laps finished (so a
    crash on lap 2 counts) minus SCORE_GAP_WEIGHT x the capped flying-lap gap to the bot (%)."""
    finished = min(m["eval/laps_completed"] / laps, 1.0)
    fly = m.get("eval/flying_lap_time_median")
    gap = np.clip(100.0 * (fly / bot_lap - 1), -SCORE_GAP_CAP, SCORE_GAP_CAP) if fly is not None else SCORE_GAP_CAP
    return float(finished - SCORE_GAP_WEIGHT * gap)


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


# SB3 key -> (MLflow name, sign). Other keys go to sb3/ (SAC's losses), except for what says the same as something else:
_SB3_NAMES = {"train/approx_kl": ("dashboard/ppo_approx_kl", 1), "train/entropy_loss": ("dashboard/ppo_entropy", -1),
              "train/explained_variance": ("dashboard/ppo_explained_variance", 1),
              "train/clip_fraction": ("ppo/clip_fraction", 1), "train/value_loss": ("ppo/value_loss", 1),
              "train/policy_gradient_loss": ("ppo/policy_gradient_loss", 1),
              "train/learning_rate": ("ppo/learning_rate", 1)}
_SB3_DROPPED = ("rollout/", "time/", "train/loss", "train/std", "train/clip_range", "train/n_updates")

class MLflowOutputFormat(KVWriter):
    """Forwards SB3 logger records to MLflow, see _SB3_NAMES."""

    def write(self, key_values, key_excluded, step=0):
        metrics = {}
        for k, v in key_values.items():
            if not isinstance(v, (int, float, np.integer, np.floating)) or isinstance(v, bool):
                continue
            if k.startswith(_SB3_DROPPED):
                continue
            name, sign = _SB3_NAMES.get(k, ("sb3/" + k, 1))
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
    dimension (`ppo/std_<name>`) every `every` timesteps. Without a cap the entropy bonus can grow the std
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
            mlflow.log_metrics({f"ppo/std_{n}": float(v) for n, v in zip(self.names, std)}, step=self.num_timesteps)
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
    """Training metrics from ended episodes, logged at most every `every` timesteps."""

    def __init__(self, every: int = 20_000, window: int = 200):
        super().__init__()
        self.every, self.buf = every, deque(maxlen=window)
        self._last = 0

    def _on_training_start(self):
        _, self._sb3env = _base_env(self.training_env)
        self._t = time.perf_counter()
        self._env_s, self._env_n, self._steps = 0.0, 0, self.num_timesteps
        env = _base_env(self.training_env)[0]
        self._reset_s = env.client.stats()["worker_reset_seconds"]
        self._log_assignment(env, "start")
        self._tracks: list[np.ndarray] = []  # samples per track of each update since the last log

    @staticmethod
    def _log_assignment(env: SpeedDreamsVecEnv, when: str) -> None:
        mlflow.log_dict({str(i): r for i, r in enumerate(env.client.races)}, f"tracks/assignment_{when}.json")

    def _on_rollout_start(self):
        self._track_steps0 = _base_env(self.training_env)[0].track_steps.copy()

    def _on_rollout_end(self):
        self._tracks.append(_base_env(self.training_env)[0].track_steps - self._track_steps0)

    def _on_training_end(self):
        self._log_assignment(_base_env(self.training_env)[0], "end")

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
            m["dashboard/rollout_progress"] = float(np.mean([s["progress"] for s in b]))
            groups = [END_GROUPS[s["end"]] for s in b]
            m["dashboard/rollout_off_track"] = groups.count("off_track") / len(b)
            m["dashboard/rollout_crashed"] = groups.count("stuck_crashed") / len(b)
            rewards = {c: float(np.mean([s[f"reward_{c}"] for s in b])) for c in REWARD_COMPONENTS}
            m["dashboard/rollout_reward"] = sum(rewards.values())
            weights = _base_env(self.training_env)[0].cfg.reward
            m |= {f"reward/{c}": v for c, v in rewards.items() if getattr(weights, _REWARD_WEIGHT.get(c, c)) != 0}
            m["dashboard/rollout_steer_delta"] = float(np.mean([s["steer_delta"] for s in b]))
            for k in DRIVING_STATS:
                m[f"driving/{k}"] = float(np.mean([s[k] for s in b]))
            self.buf.clear()
        env, sb3env = _base_env(self.training_env)
        if self._tracks:
            samples = np.stack(self._tracks)
            self._tracks.clear()
            m["perf/tracks_covered"] = int((samples > 0).sum(axis=1).min())
            m["perf/min_track_samples"] = int(samples.min())
        now = time.perf_counter()
        wall, n = now - self._t, self.num_timesteps - self._steps
        env_s, env_n = sb3env.env_seconds - self._env_s, sb3env.env_steps - self._env_n
        reset_s = env.client.stats()["worker_reset_seconds"]
        if env_s > 0:
            m["perf/env_steps_per_s"] = env_n / env_s
        if wall > 0:
            m["dashboard/steps_per_s"] = n / wall
            m["perf/reset_share"] = (reset_s - self._reset_s) / (env.num_envs * wall)
        self._t, self._steps, self._env_s, self._env_n, self._reset_s = now, self.num_timesteps, sb3env.env_seconds, \
            sb3env.env_steps, reset_s
        mlflow.log_metrics(m, step=self.num_timesteps)


class ObsStatsCallback(BaseCallback):
    """Per named observation group, what the policy actually received over the last rollout (not what a test assumed):
    the largest magnitude as a metric, and min, max, mean magnitude and spread in the artifact obs_stats.csv (one chart
    each would be 4 per group), plus the count of non-finite values and of slots with one value throughout. Every `every`
    timesteps; reads the rollout buffer, so it costs nothing during rollouts."""

    def __init__(self, every: int = 1_000_000):
        super().__init__()
        self.every, self._last = every, -every
        self._csv = ["step,group,min,max,abs_mean,std"]

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if self.num_timesteps - self._last < self.every:
            return
        self._last = self.num_timesteps
        layout = _base_env(self.training_env)[0].obs_layout
        obs = self.model.rollout_buffer.observations.reshape(-1, layout.dim)
        m = {"obs/nonfinite": int((~np.isfinite(obs)).sum()), "obs/dead_slots": int((obs.max(axis=0) == obs.min(axis=0)).sum())}
        for g in layout.groups:
            v = layout.group(obs, g.name)
            m[f"obs/{g.name}"] = float(np.abs(v).max())
            self._csv.append(f"{self.num_timesteps},{g.name},{v.min():.4g},{v.max():.4g},{np.abs(v).mean():.4g},{v.std():.4g}")
        mlflow.log_metrics(m, step=self.num_timesteps)
        mlflow.log_text("\n".join(self._csv) + "\n", "obs_stats.csv")


class EvalCallback(BaseCallback):
    """Every `every` timesteps (and at the end): deterministic eval over `laps` laps with `episodes` envs per track,
    eval metrics, and the track map / speed trace artifacts.

    All evaluated tracks (the training tracks to evaluate, then the `heldout` races, never trained on) run at once
    in one env kept between evals: worker i drives track i % n_tracks, so an eval lasts about as long as its longest
    track rather than the sum of all of them. The first training track gets the artifacts and its own lap times and
    driving under eval/; per-track and averaged metrics go to track_*/, dashboard/ and eval/.

    best/ holds the model of the best-scoring eval within the best window of `smooth` consecutive evals (by mean
    score, logged as dashboard/score_smoothed), so that one lucky eval does not win. Only full windows compete: until the
    first one, best/ is provisional (the best eval so far, so that a short or killed run still has one) and the first
    full window replaces it whatever its score. Each eval's model is kept locally until it leaves the window."""

    def __init__(self, cfg: EnvConfig, every: int = 100_000, episodes: int = 8, laps: int = 3,
                 launch_noise: float = 0.1, verbose: int = 0, heldout: tuple[str, ...] = (),
                 eval_train: tuple[str, ...] = (), smooth: int = 3, at_start: bool = False):
        super().__init__(verbose)
        # training tracks to evaluate (default: all of them); the first one gets eval/* and the artifacts
        self.train_races = tuple(eval_train) or tuple(cfg.races) or (cfg.race,)
        self.heldout = tuple(r for r in heldout if r not in self.train_races)
        self.races = self.train_races + self.heldout
        self.cfg = eval_config(cfg, self.races, episodes, laps)
        self.every, self.launch_noise, self.smooth = every, launch_noise, max(1, smooth)
        self.at_start = at_start  # also evaluate at step 0: the baseline of a resumed run
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
        mlflow.log_dict({track_of(r): RacingLine.for_race(r).abs_offset for r in self.races}, "tracks/bot_abs_offset.json")
        assert list(self.env.client.races) == [self.races[i % n_tracks] for i in range(self.env.num_envs)]
        self._limits = eval_time_limits([self.refs[r].best_lap for r in self.races], self.env.num_envs,
                                        self.cfg.laps_per_episode)
        self._tmp = tempfile.TemporaryDirectory(prefix="rlpilot-eval-")
        if self.at_start:
            self.run()

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
        first = {k: res.metrics[k] for k in ("eval/flying_lap_time_best", "eval/flying_lap_time_median") if k in res.metrics}
        first |= {f"eval/driving/{k}": res.metrics[f"eval/behavior/{k}"] for k in EVAL_DRIVING_STATS}
        mlflow.log_metrics(first, step=step)

        # Score: track_score, averaged over the evaluated training and held-out tracks. Held-out tracks are never
        # trained on but do select the checkpoint, so their metrics at best/ are optimistic.
        agg, scores = self._track_metrics({r: x.metrics for r, x in results.items()})
        mlflow.log_metrics(agg, step=step)
        self._select_best(step, float(np.mean(scores)))

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
        mlflow.log_metrics({"eval/score": score, "dashboard/score_smoothed": smoothed}, step=step)
        full = len(self._window) == self.smooth
        if not full or smoothed > self.best_score:
            if full:
                self.best_score = smoothed
            best_step, _, best_snap = max(self._window, key=lambda w: w[1])
            mlflow.log_artifacts(str(best_snap), "best")  # overwrites the previous best in the run's artifacts
            mlflow.set_tag("best_step", str(best_step))

    def _track_metrics(self, per_track: dict[str, dict[str, float]]) -> tuple[dict[str, float], list[float]]:
        out, groups, scores = {}, {"train": [], "heldout": []}, []
        for race, m in per_track.items():
            bot, name = self.refs[race].best_lap, track_of(race)
            t = {"progress": m["eval/progress"], "completion": m["eval/completion_rate"],
                 "line_dev": m["eval/behavior/line_dev"], "abs_offset": m["eval/behavior/abs_offset"]}
            out[f"track_progress/{name}"] = t["progress"]
            if "eval/flying_lap_time_median" in m:
                t["gap_pct"] = 100.0 * (m["eval/flying_lap_time_median"] / bot - 1)
                out[f"track_gap_pct/{name}"] = t["gap_pct"]
            if "eval/fail_dist_from_start" in m:
                out[f"track_fail_dist/{name}"] = m["eval/fail_dist_from_start"]
            groups["train" if race in self.train_races else "heldout"].append(t)
            scores.append(track_score(m, bot, self.cfg.laps_per_episode))
        for g, ts in groups.items():
            if not ts:
                continue
            for k, folder in (("progress", "dashboard"), ("completion", "dashboard"), ("gap_pct", "dashboard"),
                              ("line_dev", "eval"), ("abs_offset", "eval")):
                vals = [t[k] for t in ts if k in t]
                if vals:
                    out[f"{folder}/{g}_{k}"] = float(np.mean(vals))
            # gap_pct only averages tracks with a flying lap: read it with this
            folder = "dashboard" if g == "heldout" else "eval"
            out[f"{folder}/{g}_flying_lap_tracks"] = float(np.mean(["gap_pct" in t for t in ts]))
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
