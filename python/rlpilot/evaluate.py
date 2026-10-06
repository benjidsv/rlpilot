"""Deterministic-policy evaluation over a few laps, compared to the built-in bot.

The simulator is deterministic, so with a deterministic policy every eval episode would be identical.
Episode 0 is run exactly as the policy drives; the others get small action noise during the launch
only (launch_seconds), which makes the trajectories diverge so completion rate and median lap time
measure robustness rather than repeating one sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ._native import F
from .reference import Reference, lap_slice, lap_times_from_trace
from .reward import END_GROUPS, TERM_REASONS
from .vec_env import SpeedDreamsVecEnv

TRACE_FIELDS = {"time": "time", "laps": "laps", "dist_from_start": "dist_from_start", "dist_raced": "dist_raced",
                "speed_x": "speed_x", "x": "pos_x", "y": "pos_y", "last_lap_time": "last_lap_time",
                "wheels_off": "wheels_off", "gear": "gear", "rpm": "rpm"}


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
    n = env.num_envs
    rng = np.random.default_rng(seed)
    noise_steps = int(round(launch_seconds / env.cfg.step_seconds))
    cols = [F[f] for f in TRACE_FIELDS.values()]
    rows: list[list[np.ndarray]] = [[] for _ in range(n)]
    acts: list[list[np.ndarray]] = [[] for _ in range(n)]
    stats: list[dict | None] = [None] * n
    active = np.ones(n, bool)

    obs, _ = env.reset()
    t = 0
    while active.any():
        action = np.asarray(policy(obs), np.float32).reshape(n, env.act_dim).copy()
        if t < noise_steps and n > 1:
            action[1:] += rng.normal(0, launch_noise, action[1:].shape).astype(np.float32)
        obs, _, term, trunc, info = env.step(action)
        done = term | trunc
        for i in np.flatnonzero(active):
            raw = info["final_raw"][i] if done[i] else env._raw[i]
            rows[i].append(raw[cols].copy())
            acts[i].append(np.clip(action[i, :2], -1, 1))
            if done[i]:
                s = {k: float(v[i]) for k, v in info["stats"].items()}
                r = int(s.pop("term_reason"))
                s["end"] = TERM_REASONS[r] if r >= 0 else "time_limit"
                s["return"] = float(info["episode"]["r"][i])
                stats[i] = s
                active[i] = False
        t += 1

    traces, lap_times = [], []
    for i in range(n):
        data = np.stack(rows[i])
        a = np.stack(acts[i])
        tr = {k: data[:, j] for j, k in enumerate(TRACE_FIELDS)} | {"steer": a[:, 0], "pedal": a[:, 1]}
        traces.append(tr)
        lap_times.append(lap_times_from_trace(tr))

    all_laps = np.concatenate(lap_times)
    m = {
        "eval/completion_rate": float(np.mean([len(x) >= 1 for x in lap_times])),
        "eval/progress": float(np.mean([s["progress"] for s in stats])),
        "eval/laps_completed": float(np.mean([len(x) for x in lap_times])),
    }
    if len(all_laps):
        m["eval/lap_time_best"] = float(all_laps.min())
        m["eval/lap_time_median"] = float(np.median(all_laps))
    if reference is not None and len(reference.lap_times):
        m["eval/bot_lap_time_best"] = reference.best_lap
        m["eval/bot_lap_time_median"] = reference.median_lap
        if len(all_laps):
            m["eval/gap_to_bot_best"] = m["eval/lap_time_best"] - reference.best_lap
    for k in ("steer_delta", "offtrack_frac", "mean_speed", "full_throttle_frac"):
        m[f"eval/behavior/{k}"] = float(np.mean([s[k] for s in stats]))
    groups = [END_GROUPS[s["end"]] for s in stats]
    for g in sorted(set(END_GROUPS.values())):
        m[f"eval/end/{g}"] = groups.count(g) / n
    return EvalResult(m, traces, lap_times, stats)
