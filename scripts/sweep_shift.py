"""Eval one checkpoint under different automatic shift-up points (fraction of redline) to test the gearbox.

    python scripts/sweep_shift.py mlruns/1/<run>/artifacts/checkpoints/0055000000 --up 0.7 0.8 0.9 0.95
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from rlpilot import EnvConfig, SpeedDreamsVecEnv
from rlpilot import reference
from rlpilot.evaluate import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="directory with model.zip")
    p.add_argument("--up", type=float, nargs="+", default=[0.7, 0.8, 0.9, 0.95])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--laps", type=int, default=3)
    a = p.parse_args()

    model = PPO.load(os.path.join(a.checkpoint, "model.zip"), device="cpu")
    ref = reference.load(laps=a.laps)
    cfg = EnvConfig(n_workers=a.episodes, laps_per_episode=a.laps, run_dir=None, max_episode_seconds=min(400.0, 140.0 * a.laps))
    print(f"bot: best lap {ref.best_lap:.2f}  top speed {ref.trace['speed_x'].max() * 3.6:.0f} km/h")
    print("up    laps  best    median  top km/h  completion")
    for up in a.up:
        os.environ["RLP_SHIFT_UP"] = str(up)  # inherited by the spawned workers
        env = SpeedDreamsVecEnv(cfg)
        try:
            res = evaluate(lambda o: model.predict(o, deterministic=True)[0], env, ref)
        finally:
            env.close()
        m = res.metrics
        top = max(float(t["speed_x"].max()) for t in res.traces) * 3.6
        print(f"{up:.2f}  {m['eval/laps_completed']:.2f}  {m.get('eval/lap_time_best', float('nan')):6.2f}  "
              f"{m.get('eval/lap_time_median', float('nan')):6.2f}  {top:7.0f}   {m['eval/completion_rate']:.2f}")


if __name__ == "__main__":
    main()
