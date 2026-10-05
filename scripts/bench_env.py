"""Throughput of SpeedDreamsVecEnv (and optionally the SB3 adapter) with random or scripted actions.

    python scripts/bench_env.py --workers 1 2 4 5 --steps 20000 [--policy scripted] [--sb3]
"""

import argparse
import time

import numpy as np

from rlpilot import EnvConfig, SpeedDreamsVecEnv
from rlpilot._native import F, N_FIELDS, N_RAYS


def scripted(env: SpeedDreamsVecEnv) -> np.ndarray:
    raw = env._raw
    steer = 1.5 * (raw[:, F["angle"]] - 0.5 * raw[:, F["to_middle"]] / np.maximum(raw[:, F["track_width"]], 1))
    target = 10 + 0.25 * raw[:, N_FIELDS + N_RAYS // 2]  # faster when the road ahead is clear
    pedal = np.where(raw[:, F["speed_x"]] < target, 0.8, np.where(raw[:, F["speed_x"]] > target + 5, -0.5, 0))
    return np.stack([steer, pedal], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--steps", type=int, default=10000, help="vector steps")
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--policy", choices=["random", "scripted"], default="scripted")
    p.add_argument("--sb3", action="store_true", help="go through the SB3 VecEnv adapter")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    for n in a.workers:
        env = SpeedDreamsVecEnv(EnvConfig(n_workers=n, frame_skip=a.frame_skip))
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
        print(f"workers={n} fs={a.frame_skip} {a.policy}{' sb3' if a.sb3 else ''}: "
              f"{n * a.steps / wall:,.0f} env steps/s ({n * a.steps * env.cfg.step_seconds / wall:,.0f}x real time), "
              f"{episodes} episodes, reset {1e3 * s['worker_reset_seconds'] / max(s['worker_resets'], 1):.1f} ms each, "
              f"reset share {100 * s['worker_reset_seconds'] / (n * wall):.0f}%"
              + (f", mean dist {np.mean(dist):.0f} m" if dist else ""))
        venv.close()


if __name__ == "__main__":
    main()
