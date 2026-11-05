"""Throughput of SpeedDreamsVecEnv (and optionally the SB3 adapter) with random or scripted actions.

    python scripts/bench_env.py --workers 1 2 4 5 --steps 20000 [--policy scripted] [--sb3]
    python scripts/bench_env.py --preset proto7 --sb3 --steps 3000 [--eval-workers 80]

Presets reproduce a run's env: proto6 = 20 workers on 35 tracks with the bot's line, proto7 = one worker per training
track with steering increments and road edges. Memory: the sum of the workers' resident sizes, and of their footprints
(which don't double-count shared pages) with --eval-workers, which also holds a run's eval env open.
"""

import argparse
import re
import subprocess
import sys
import time

import numpy as np

from rlpilot import EnvConfig, SpeedDreamsVecEnv
from rlpilot._native import F, N_FIELDS, N_RAYS
from rlpilot.evaluate import eval_config
from rlpilot.splits import PROTO7_EVAL_TRAIN, PROTO7_TEST, PROTO7_TRAIN, PROTO7_VALIDATION
from rlpilot.track import race_for


def preset(name: str | None, n: int, frame_skip: int) -> EnvConfig:
    common = dict(n_workers=n, frame_skip=frame_skip, gearbox="points", lookahead=True, car_obs=True, spin_us=50,
                  car_grip=(0.9, 1.1), car_mass=(0.9, 1.1), car_power=(0.9, 1.1))
    if name == "proto6":
        return EnvConfig(races=tuple(race_for(t) for t in PROTO7_TRAIN + PROTO7_TEST), line_obs=True,
                         track_switch_steps=200_000, **common)
    if name == "proto7":
        return EnvConfig(races=tuple(race_for(t) for t in PROTO7_TRAIN), edge_points=30, steer_limit=0.0625, steer_history=3,
                         **common)
    return EnvConfig(n_workers=n, frame_skip=frame_skip)


def scripted(env: SpeedDreamsVecEnv) -> np.ndarray:
    raw = env._raw
    steer = 1.5 * (raw[:, F["angle"]] - 0.5 * raw[:, F["to_middle"]] / np.maximum(raw[:, F["track_width"]], 1))
    if env.cfg.steer_limit:
        steer = (steer - env._prev_action[:, 0]) / env.cfg.steer_limit
    target = 10 + 0.25 * raw[:, N_FIELDS + N_RAYS // 2]  # faster when the road ahead is clear
    pedal = np.where(raw[:, F["speed_x"]] < target, 0.8, np.where(raw[:, F["speed_x"]] > target + 5, -0.5, 0))
    return np.concatenate([np.stack([steer, pedal], 1), np.zeros((len(raw), env.act_dim - 2))], 1).astype(np.float32)


def rss_gb(env: SpeedDreamsVecEnv) -> float:
    pids = ",".join(str(p.pid) for p in env.client.procs)
    return sum(int(x) for x in subprocess.run(["ps", "-o", "rss=", "-p", pids], capture_output=True, text=True).stdout.split()) / 1e6


def footprint_gb(env: SpeedDreamsVecEnv) -> float:
    """Sum of the workers' physical footprints, from the macOS `footprint` tool, one process at a time."""
    if sys.platform != "darwin":
        return float("nan")
    units = {"B": 1e-9, "KB": 1e-6, "MB": 1e-3, "GB": 1.0}
    total = 0.0
    for p in env.client.procs:
        out = subprocess.run(["footprint", str(p.pid)], capture_output=True, text=True).stdout
        found = re.search(r"Footprint:\s+([\d.]+)\s+(\w+)", out)
        if found:
            total += float(found.group(1)) * units[found.group(2)]
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, nargs="+", default=None, help="default: 1 2 4, or the preset's own count")
    p.add_argument("--steps", type=int, default=10000, help="vector steps")
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--policy", choices=["random", "scripted"], default="scripted")
    p.add_argument("--sb3", action="store_true", help="go through the SB3 VecEnv adapter")
    p.add_argument("--preset", choices=["proto6", "proto7"], default=None)
    p.add_argument("--eval-workers", type=int, default=0, help="also hold an idle eval env of this size, for memory")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    for n in a.workers or {"proto6": [20], "proto7": [29]}.get(a.preset, [1, 2, 4]):
        cfg = preset(a.preset, n, a.frame_skip)
        env = SpeedDreamsVecEnv(cfg)
        venv = env
        if a.sb3:
            from rlpilot.sb3 import SB3VecEnv
            venv = SB3VecEnv(env)
        venv.reset()
        episodes, dist = 0, []
        t = time.perf_counter()
        for _ in range(a.steps):
            act = scripted(env) if a.policy == "scripted" else rng.uniform(-1, 1, (n, env.act_dim)).astype(np.float32)
            if a.sb3:
                _, _, dones, infos = venv.step(act)
                episodes += int(dones.sum())
            else:
                _, _, term, trunc, info = venv.step(act)
                if "stats" in info:
                    episodes += int((term | trunc).sum())
                    dist += list(info["stats"]["dist_raced"][info["_stats"]])
        wall = time.perf_counter() - t
        s = env.client.stats()
        print(f"{a.preset or 'workers'} n={n} fs={a.frame_skip} {a.policy}{' sb3' if a.sb3 else ''}: "
              f"{n * a.steps / wall:,.0f} env steps/s ({n * a.steps * env.cfg.step_seconds / wall:,.0f}x real time), "
              f"{1e6 * wall / a.steps:.0f} us per vector step, obs dim {env.obs_layout.dim}, "
              f"{episodes} episodes, reset {1e3 * s['worker_reset_seconds'] / max(s['worker_resets'], 1):.1f} ms each, "
              f"reset share {100 * s['worker_reset_seconds'] / (n * wall):.0f}%"
              + (f", mean dist {np.mean(dist):.0f} m" if dist else ""))
        print(f"  memory, {n} training workers: rss {rss_gb(env):.2f} GB, footprint {footprint_gb(env):.2f} GB")
        if a.eval_workers:
            races = tuple(race_for(t) for t in PROTO7_EVAL_TRAIN + PROTO7_VALIDATION)
            ev = SpeedDreamsVecEnv(eval_config(cfg, races, a.eval_workers // len(races), 2))
            try:
                ev.reset()
                print(f"  memory, {a.eval_workers} eval workers: rss {rss_gb(ev):.2f} GB, footprint {footprint_gb(ev):.2f} GB")
            finally:
                ev.close()
        venv.close()


if __name__ == "__main__":
    main()
