"""The reward sweep: tests of reward variants, 3 seeds each (the default for any comparison: one run's result moves by
~0.055 in validation nominal survival from seed to seed), one after the other on this machine. 3M-step runs follow a
10M learning-rate schedule; --steps 10000000 runs a full 10M schedule with an eval every 2M. Seed-major: every variant's seed 1, then every seed 2, ..., so a sweep stopped early still
covers every variant. Resumable: a run whose final eval CSV exists is skipped, so rerunning the command continues.

    caffeinate -i uv run python scripts/sweep.py                  # the whole sweep (~8.5 min per run)
    uv run python scripts/sweep.py --dry-run                      # list the runs and their commands
    caffeinate -i uv run python scripts/sweep.py --only base --seeds 1 --steps 10000000   # one 10M run
    uv run python scripts/sweep_report.py [--steps 10000000]      # the results, per variant over seeds

Runs are runs/sw-<variant>__<seed>__<time>/ (3M) or runs/sw<N>m-<variant>__<seed>__<time>/; each run's output goes to
runs/sweep/<prefix>-<variant>-s<seed>.log (3M: <variant>-s<seed>.log) and one line per finished run to
runs/sweep/status.tsv. Every variant spells out every reward argument, so the sweep doesn't
depend on train.py's defaults.
"""

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

STEPS, ANNEAL = 3_000_000, 10_000_000  # the default run: a 10M run's first 3M
LONG_EVAL_EVERY = 2_000_000            # runs longer than STEPS are evaluated this often (3M runs: once, at the end)
SEEDS = (1, 2, 3)

# The phase-1 reward: 0.1 per metre (signed), damage 0.001 per point, -10 on failure, no control cost, gamma 0.995.
BASE = {"lost-control-cost": 0.0, "best-only-progress": False, "terminal": 10.0, "gamma": 0.995,
        "progress-units": "metres", "lost-control-causes": ("contact", "backwards", "far_off")}

VARIANTS = {
    # the reference, and the noise between seeds
    "base": {},
    # control-3m split in two: the cost alone, best-only progress alone, both (control-3m again)
    "cost2": {"lost-control-cost": 2.0},
    "best": {"best-only-progress": True},
    "best-cost2": {"best-only-progress": True, "lost-control-cost": 2.0},
    # the cost's size
    "cost0.5": {"lost-control-cost": 0.5},
    "cost1": {"lost-control-cost": 1.0},
    # giving up made dearer: the failure is 10 s away, so -10 is worth ~-2.9 at gamma 0.995. A bigger terminal, or a
    # longer horizon (gamma 0.998: -10 worth ~-6.1), alone and with the cost
    "t20": {"terminal": 20.0},
    "cost1-t20": {"lost-control-cost": 1.0, "terminal": 20.0},
    "cost2-t20": {"lost-control-cost": 2.0, "terminal": 20.0},
    "g998": {"gamma": 0.998},
    "cost2-g998": {"lost-control-cost": 2.0, "gamma": 0.998},
    # facing backwards uncharged: crashed starts froze facing backwards, and a turn round can cross 90 degrees often
    "cost2-nobackwards": {"lost-control-cost": 2.0, "lost-control-causes": ("contact", "far_off")},
    # progress in the bot's seconds (the next planned test), alone and with a cost
    "botsec": {"progress-units": "bot_seconds"},
    "botsec-cost1": {"progress-units": "bot_seconds", "lost-control-cost": 1.0},
    # the sweep's lead (botsec-cost1) with best-only progress, which raised crashed completion on its own
    "botsec-best-cost1": {"progress-units": "bot_seconds", "best-only-progress": True, "lost-control-cost": 1.0},
}


def prefix(steps: int) -> str:
    return "sw" if steps == STEPS else f"sw{steps / 1e6:g}m"


def command(variant: str, seed: int, steps: int) -> list[str]:
    cmd = [sys.executable, "scripts/train.py", "--exp-name", f"{prefix(steps)}-{variant}", "--seed", str(seed),
           "--total-timesteps", str(steps), "--anneal-timesteps", str(max(ANNEAL, steps))]
    if steps > STEPS:
        cmd += ["--eval-every", str(LONG_EVAL_EVERY)]
    for k, v in (BASE | VARIANTS[variant]).items():
        if isinstance(v, bool):
            cmd.append(f"--{k}" if v else f"--no-{k}")
        elif isinstance(v, tuple):
            cmd += [f"--{k}", *v]
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def finished(variant: str, seed: int, steps: int = STEPS) -> list[str]:
    """Final eval CSVs of this variant and seed (the eval at the last iteration is the one named after it)."""
    return [p for d in glob.glob(f"runs/{prefix(steps)}-{variant}__{seed}__*") for p in glob.glob(f"{d}/eval/*.csv")
            if Path(p).stem.isdigit() and int(Path(p).stem) >= steps - 20_000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=list(VARIANTS), help="variants to run (default: all)")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--steps", type=int, default=STEPS, help="run length (3M: a 10M schedule's start; more: full schedule)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    variants = a.only or list(VARIANTS)
    todo = [(v, s) for s in a.seeds for v in variants if not finished(v, s, a.steps)]
    out = Path("runs/sweep")
    out.mkdir(parents=True, exist_ok=True)
    per_run = 8.5 * a.steps / STEPS  # minutes
    print(f"{len(todo)} runs to go ({len(variants) * len(a.seeds) - len(todo)} done), ~{len(todo) * per_run / 60:.1f} h")
    for i, (v, s) in enumerate(todo):
        cmd = command(v, s, a.steps)
        if a.dry_run:
            print(" ".join(cmd[1:]))
            continue
        t0 = time.time()
        name = v if a.steps == STEPS else f"{prefix(a.steps)}-{v}"
        print(f"[{time.strftime('%H:%M')}] {i + 1}/{len(todo)} {name} seed {s}", flush=True)
        with open(out / f"{name}-s{s}.log", "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        minutes = (time.time() - t0) / 60
        with open(out / "status.tsv", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{name}\t{s}\t{code}\t{minutes:.1f}\n")
        print(f"         exit {code}, {minutes:.1f} min" + ("  <- FAILED, see the log" if code else ""), flush=True)


if __name__ == "__main__":
    main()
