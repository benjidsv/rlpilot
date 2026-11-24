"""Evaluate a checkpoint on the scenario suite (and optionally grid laps).

    python scripts/eval.py runs/<run>/checkpoints/best.pt [--splits validation train] [--grid-laps]
    python scripts/eval.py runs/<run>/checkpoints/best.pt --splits test --test   # the final exam, once

Writes the per-scenario results next to the checkpoint's run (runs/<run>/eval/<checkpoint>-<splits>.csv), and prints the
metrics and a per-track table. Compare two result files with scripts/compare.py.
"""

import argparse
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rlpilot import evaluate, obs
from rlpilot.tracks import load_split

sys.path.insert(0, str(Path(__file__).parent))
from train import Agent  # noqa: E402  (the policy is defined in the single-file trainer)


def load_policy(path: Path):
    ckpt = torch.load(path, weights_only=False)
    obs.configure(ckpt["args"].get("obs_height", "none"))  # the inputs it was trained with
    space = types.SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (obs.DIM,), np.float32))
    agent = Agent(space, ckpt["args"]["hidden"], activation=ckpt["args"].get("activation", "tanh"),
                  learned_std=ckpt["args"].get("learned_std", False))
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    mean, std = ckpt["obs_mean"], np.sqrt(ckpt["obs_var"] + ckpt["epsilon"])
    fixed = ckpt.get("obs_fixed", np.zeros(len(mean), bool))  # inputs that skip the normalization, as in training
    mean, std = np.where(fixed, 0.0, mean), np.where(fixed, 1.0, std)
    torch.set_num_threads(1)  # as in training's evals: the same thread count gives the same actions

    def policy(o):
        x = np.clip((o - mean) / std, -10, 10).astype(np.float32)
        with torch.no_grad():
            return agent.act_deterministic(torch.as_tensor(x)).numpy()
    return policy, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--splits", nargs="+", default=["validation"], choices=["train", "validation", "test"])
    p.add_argument("--test", action="store_true", help="required to touch the test split (once, at the very end)")
    p.add_argument("--workers", type=int, default=48, help="results don't depend on it")
    p.add_argument("--grid-laps", action="store_true", help="also drive 2 laps from the grid on each track of the splits")
    a = p.parse_args()
    if "test" in a.splits and not a.test:
        raise SystemExit("the test split is for the final evaluation only: pass --test if this is it")
    policy, ckpt = load_policy(a.checkpoint)
    rows = evaluate.run_suite(policy, evaluate.load_suite(tuple(a.splits)), a.workers)
    out = a.checkpoint.parent.parent / "eval" / f"{a.checkpoint.stem}-{'-'.join(a.splits)}.csv"
    evaluate.write_results(rows, out)
    print(f"checkpoint {a.checkpoint} (step {ckpt['step']}), results in {out}\n")
    for k, v in evaluate.summarize(rows).items():
        print(f"  {k:40s} {v:.3f}")
    table = evaluate.per_track(rows)
    cols = [c for c in table[0] if c not in ("track", "split", "scenarios")]
    print(f"\n  {'track':15s} {'split':10s} " + " ".join(f"{c:>9s}" for c in cols))
    for r in table:
        print(f"  {r['track']:15s} {r['split']:10s} " + " ".join(f"{r[c]:9.2f}" for c in cols))
    lost = evaluate.lost_breakdown(rows)
    cols = [c for c in lost[0] if c not in ("split", "tier")]
    print("\n  recovered scenarios by the first way they lost control; damage over the scenarios that took any")
    print(f"  {'split':10s} {'tier':10s} " + " ".join(f"{c:>13s}" for c in cols))
    for r in lost:
        print(f"  {r['split']:10s} {r['tier']:10s} " + " ".join(
            f"{r[c]:13.0f}" if c == "median damage" else f"{r[c]:13d}" for c in cols))
    if a.grid_laps:
        tracks = [t for s in a.splits for t in load_split()[s]]
        print("\n  grid laps (2 laps from a standing start):")
        for r in evaluate.grid_laps(policy, tracks):
            times = ", ".join(f"{x:.2f}" for x in r["lap_times"])
            print(f"  {r['track']:15s} {r['end']:12s} laps {r['laps']}  [{times}]  best / bot {r['best_vs_bot']:.3f}")


if __name__ == "__main__":
    main()
