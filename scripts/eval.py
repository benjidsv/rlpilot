"""Evaluate a checkpoint on the scenario suite (and optionally grid laps).

    python scripts/eval.py runs/<run>/checkpoints/best.pt [--splits validation train] [--grid-laps]
    python scripts/eval.py runs/<run>/checkpoints/best.pt --splits test --test   # the final exam, once

Writes the per-scenario results next to the checkpoint's run (runs/<run>/eval/<checkpoint>-<splits>.csv), and prints the
metrics and a per-track table. Compare two result files with scripts/analysis/compare.py.
"""

import argparse
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rlpilot import controls, evaluate, obs
from rlpilot.tracks import load_split

sys.path.insert(0, str(Path(__file__).parent))
from train import Agent  # noqa: E402  (the policy is defined in the single-file trainer)


def load_policy(path: Path):
    """The checkpoint's deterministic policy (raw observations -> actions), and the checkpoint."""
    agent, normalize, ckpt = load_agent(path)

    def policy(o):
        with torch.no_grad():
            return agent.act_deterministic(torch.as_tensor(normalize(o))).numpy()
    return policy, ckpt


SHIFT_MODES = ("argmax", "sample", "hazard")


class Driver:
    """The checkpoint's policy for watching and telemetry: steer and pedal at the mean, the shift picked by `mode`:
    - argmax: the most likely shift at each step (the in-run eval, Agent.act_deterministic). A policy trained by
      sampling can hold P(up) at ~0.3 per step on the rev limiter: sampled it upshifts within ~3 steps, argmax never.
    - sample: drawn from the policy's shift probabilities, as in training (seeded; another seed, another drive).
    - hazard: deterministic, the median of the sampled timing: shift once the chance that sampling would have shifted
      since the last gear change passes 1/2 (down and up each keep their own; the likelier one goes first).
    Call it with the normalized observations [n, obs] and the cars' current gears [n] (hazard restarts its clocks when
    a car's gear changes)."""

    def __init__(self, agent, mode: str = "argmax", seed: int = 0):
        if mode not in SHIFT_MODES:
            raise ValueError(f"shift mode {mode!r} not in {SHIFT_MODES}")
        self.agent, self.mode = agent, mode
        self.gen = torch.Generator().manual_seed(seed)
        self.stay = None  # hazard: per car, P(sampling would not have shifted down / up since the last gear change)
        self.prev_gear = None

    def __call__(self, x: torch.Tensor, gear: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if self.mode == "argmax":
                return self.agent.act_deterministic(x).numpy()
            h = self.agent.actor_body(x)
            mean = self.agent.actor_mean(h).numpy()
            p = torch.softmax(self.agent.shift_logits(h), 1)
        if self.mode == "sample":
            shift = torch.multinomial(p, 1, generator=self.gen)[:, 0].numpy()
        else:
            p = p.numpy()
            if self.stay is None or len(self.stay) != len(p):
                self.stay, self.prev_gear = np.ones((len(p), 2)), None
            if self.prev_gear is not None:
                self.stay[gear != self.prev_gear] = 1.0
            self.prev_gear = np.array(gear, copy=True)
            self.stay[:, 0] *= 1 - p[:, 0]
            self.stay[:, 1] *= 1 - p[:, 2]
            shift = np.ones(len(p), np.int64)  # hold
            shift[self.stay[:, 1] < 0.5] = 2
            shift[(self.stay[:, 0] < 0.5) & (self.stay[:, 0] <= self.stay[:, 1])] = 0
        return np.concatenate([mean, shift[:, None].astype(np.float32)], 1)


def load_agent(path: Path):
    """The checkpoint's agent, its input normalization (raw observations -> what the agent takes), and the checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    obs.configure(ckpt["args"].get("obs_height", "none"))  # the inputs it was trained with
    controls.configure(ckpt["args"].get("gearbox", "manual"), ckpt["args"].get("auto_upshift", 0.92))  # who shifted
    space = types.SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (obs.DIM,), np.float32))
    agent = Agent(space, ckpt["args"]["hidden"], activation=ckpt["args"].get("activation", "tanh"),
                  learned_std=ckpt["args"].get("learned_std", False),
                  pedal_std_floor=ckpt["args"].get("pedal_std_floor", 0.0))
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    mean, std = ckpt["obs_mean"], np.sqrt(ckpt["obs_var"] + ckpt["epsilon"])
    fixed = ckpt.get("obs_fixed", np.zeros(len(mean), bool))  # inputs that skip the normalization, as in training
    mean, std = np.where(fixed, 0.0, mean), np.where(fixed, 1.0, std)
    torch.set_num_threads(1)  # as in training's evals: the same thread count gives the same actions

    def normalize(o):
        return np.clip((o - mean) / std, -10, 10).astype(np.float32)
    return agent, normalize, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--splits", nargs="+", default=["validation"], choices=["train", "validation", "test"])
    p.add_argument("--test", action="store_true", help="required to touch the test split (once, at the very end)")
    p.add_argument("--workers", type=int, default=48, help="results don't depend on it")
    p.add_argument("--grid-laps", action="store_true", help="also drive laps from the grid on each track of the splits")
    p.add_argument("--laps", type=int, default=2, help="laps per track for --grid-laps")
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
        split_of = {t: s for s in a.splits for t in load_split()[s]}
        rows = evaluate.grid_laps(policy, tracks, a.laps)
        print(f"\n  grid laps ({a.laps} from a standing start; clean = before the first loss of control, pace = bot "
              "time / car time over that stretch):")
        for r in rows:
            times = ", ".join(f"{x:.2f}" for x in r["lap_times"])
            print(f"  {r['track']:15s} {r['end']:12s} completion {r['completion']:.2f}  clean {r['clean']:.2f}  "
                  f"pace {r['pace']:.3f}  [{times}]")
        for k, v in evaluate.summarize_laps(rows, split_of).items():
            print(f"  {k:28s} {v:.3f}")


if __name__ == "__main__":
    main()
