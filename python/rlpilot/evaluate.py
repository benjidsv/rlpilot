"""Deterministic-policy evaluation over a few laps, compared to the built-in bot.

The simulator is deterministic, so with a deterministic policy every eval episode would be identical.
Episode 0 is run exactly as the policy drives; the others get small action noise during the launch
only (launch_seconds), which makes the trajectories diverge so completion rate and median lap time
measure robustness rather than repeating one sample.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ._native import F
from .config import EnvConfig
from .reference import Reference, lap_slice, lap_times_from_trace
from .reward import END_GROUPS, TERM_REASONS
from .vec_env import BEHAVIOR_STATS, SpeedDreamsVecEnv

TRACE_FIELDS = {"time": "time", "laps": "laps", "dist_from_start": "dist_from_start", "dist_raced": "dist_raced",
                "speed_x": "speed_x", "x": "pos_x", "y": "pos_y", "last_lap_time": "last_lap_time",
                "wheels_off": "wheels_off", "gear": "gear", "rpm": "rpm"}


def eval_config(cfg: EnvConfig, races: tuple[str, ...], episodes: int, laps: int) -> EnvConfig:
    """The env of an evaluation: `episodes` workers per race, all at once (worker i on race i % n), on the stock car (factor
    observations, if any, read 0). Workers never change track, and with one reset per eval, recycling isn't needed."""
    return dataclasses.replace(cfg, n_workers=episodes * len(races), laps_per_episode=laps, run_dir=None, race=races[0],
                               races=races if len(races) > 1 else (), track_switch_steps=0, recycle_resets=2**62, launch_share=0.0,
                               car_grip=(1.0, 1.0), car_mass=(1.0, 1.0), car_power=(1.0, 1.0))


def eval_time_limits(best_laps: list[float], n_workers: int, laps: int) -> np.ndarray:
    """Per-worker episode limits: `laps` laps at 70% of the bot's pace, plus the standing start. best_laps: per race."""
    return np.array([laps * best_laps[i % len(best_laps)] / 0.7 + 20.0 for i in range(n_workers)])


@dataclass
class EvalResult:
    metrics: dict[str, float]
    traces: list[dict[str, np.ndarray]]   # one per episode: TRACE_FIELDS + steer, pedal
    lap_times: list[np.ndarray]           # completed laps per episode
    stats: list[dict[str, float]] = field(default_factory=list)

    def best_lap_trace(self, episode: int = 0) -> tuple[dict[str, np.ndarray], int | None]:
        """Fastest completed lap of an episode, or its first (partial) lap if none was completed."""
        times = self.lap_times[episode]
        lap = int(np.argmin(times)) + 1 if len(times) else 1
        return lap_slice(self.traces[episode], lap), (lap if len(times) else None)


def evaluate(policy: Callable[[np.ndarray], np.ndarray], env: SpeedDreamsVecEnv, reference: Reference | None = None,
             launch_noise: float = 0.1, launch_seconds: float = 1.0, seed: int = 0) -> EvalResult:
    """One episode per env, all on the same track (env 0 runs clean)."""
    traces, lap_times, stats = rollout(policy, env, launch_noise, launch_seconds, seed)
    return summarize(traces, lap_times, stats, reference)


def rollout(policy: Callable[[np.ndarray], np.ndarray], env: SpeedDreamsVecEnv, launch_noise: float = 0.1,
            launch_seconds: float = 1.0, seed: int = 0, clean=(0,), time_limits: np.ndarray | None = None
            ) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray], list[dict[str, float]]]:
    """Runs one episode per env; the envs in `clean` get no launch noise. `time_limits` [n] (seconds) overrides
    the env's max_episode_seconds per env (envs on different tracks). Returns per-env traces, lap times, stats.
    Envs whose episode ended are paused (not simulated) until all have ended."""
    n = env.num_envs
    rng = np.random.default_rng(seed)
    noise_steps = int(round(launch_seconds / env.cfg.step_seconds))
    noisy = np.ones(n, bool)
    noisy[list(clean)] = False
    cols = [F[f] for f in TRACE_FIELDS.values()]
    rows: list[list[np.ndarray]] = [[] for _ in range(n)]
    acts: list[list[np.ndarray]] = [[] for _ in range(n)]
    stats: list[dict | None] = [None] * n
    active = np.ones(n, bool)

    env.set_time_limits(time_limits)
    try:
        obs, _ = env.reset()
        t = 0
        while active.any():
            action = np.asarray(policy(obs), np.float32).reshape(n, env.act_dim).copy()
            if t < noise_steps:
                action[noisy] += rng.normal(0, launch_noise, action[noisy].shape).astype(np.float32)
            obs, _, term, trunc, info = env.step(action, active)
            done = term | trunc
            for i in np.flatnonzero(active):
                raw = info["final_raw"][i] if done[i] else env._raw[i]
                rows[i].append(raw[cols].copy())
                acts[i].append(env.applied[i, :2].copy())
                if done[i]:
                    s = {k: float(v[i]) for k, v in info["stats"].items()}
                    r = int(s.pop("term_reason"))
                    s["end"] = TERM_REASONS[r] if r >= 0 else "time_limit"
                    s["return"] = float(info["episode"]["r"][i])
                    stats[i] = s
                    active[i] = False
            t += 1
    finally:
        env.set_time_limits(None)

    traces, lap_times = [], []
    for i in range(n):
        data = np.stack(rows[i])
        a = np.stack(acts[i])
        tr = {k: data[:, j] for j, k in enumerate(TRACE_FIELDS)} | {"steer": a[:, 0], "pedal": a[:, 1]}
        traces.append(tr)
        lap_times.append(lap_times_from_trace(tr))
    return traces, lap_times, stats


def summarize(traces: list[dict[str, np.ndarray]], lap_times: list[np.ndarray], stats: list[dict[str, float]],
              reference: Reference | None = None) -> EvalResult:
    """eval/* metrics over a set of episodes on one track (the first one is the clean episode)."""
    n = len(traces)
    all_laps = np.concatenate(lap_times)
    m = {
        "eval/completion_rate": float(np.mean([len(x) >= 1 for x in lap_times])),
        "eval/progress": float(np.mean([s["progress"] for s in stats])),
        "eval/laps_completed": float(np.mean([len(x) for x in lap_times])),
    }
    if len(all_laps):
        m["eval/lap_time_best"] = float(all_laps.min())
        m["eval/lap_time_median"] = float(np.median(all_laps))
    flying = np.concatenate([x[1:] for x in lap_times])  # lap 1 is from a standing start
    if len(flying):
        m["eval/flying_lap_time_best"] = float(flying.min())
        m["eval/flying_lap_time_median"] = float(np.median(flying))
    # Where failed episodes ended (not lap_done / time_limit): one spot means one corner the policy can't take.
    failed = [i for i, s in enumerate(stats) if END_GROUPS[s["end"]] not in ("lap_done", "timeout")]
    if failed:
        m["eval/fail_dist_from_start"] = float(np.median([traces[i]["dist_from_start"][-1] for i in failed]))
        m["eval/fail_lap"] = float(np.mean([traces[i]["laps"][-1] for i in failed]))
    if reference is not None and len(reference.lap_times):
        m["eval/bot_lap_time_best"] = reference.best_lap
        m["eval/bot_lap_time_median"] = reference.median_lap
        if len(all_laps):
            m["eval/gap_to_bot_best"] = m["eval/lap_time_best"] - reference.best_lap
    for k in BEHAVIOR_STATS:
        m[f"eval/behavior/{k}"] = float(np.mean([s[k] for s in stats]))
    groups = [END_GROUPS[s["end"]] for s in stats]
    for g in sorted(set(END_GROUPS.values())):
        m[f"eval/end/{g}"] = groups.count(g) / n
    return EvalResult(m, traces, lap_times, stats)
