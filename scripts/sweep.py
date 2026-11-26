"""The sweep: tests of training variants (the reward, then the steering noise), 3 seeds each (the default for any comparison: one run's result moves by
~0.055 in validation nominal survival from seed to seed), one after the other on this machine. 3M-step runs follow a
10M learning-rate schedule; --steps 10000000 runs a full 10M schedule with an eval every 2M. Seed-major: every variant's seed 1, then every seed 2, ..., so a sweep stopped early still
covers every variant. Resumable: a run whose final eval CSV exists is skipped, so rerunning the command continues.

    caffeinate -i uv run python scripts/sweep.py                  # the whole sweep (~8.5 min per run)
    uv run python scripts/sweep.py --dry-run                      # list the runs and their commands
    caffeinate -i uv run python scripts/sweep.py --only base --seeds 1 --steps 10000000   # one 10M run
    uv run python scripts/analysis/sweep_report.py [--steps 10000000]      # the results, per variant over seeds

Runs are runs/sw-<variant>__<seed>__<time>/ (3M) or runs/sw<N>m-<variant>__<seed>__<time>/; each run's output goes to
runs/sweep/<prefix>-<variant>-s<seed>.log (3M: <variant>-s<seed>.log) and one line per finished run to
runs/sweep/status.tsv. Every variant spells out every reward argument, so the sweep doesn't
depend on train.py's defaults.

On a terminal, each run's header is followed by train.py's progress bar (live for the current run, kept with its final
numbers for the finished ones); the log gets the whole output. The sweep's end time comes from the runs it timed.
"""

import argparse
import codecs
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STEPS, ANNEAL = 3_000_000, 10_000_000  # the default run: a 10M run's first 3M
LONG_EVAL_EVERY = 2_000_000  # runs longer than STEPS are evaluated this often (3M runs: once, at the end)
SEEDS = (1, 2, 3)

# The phase-1 reward: 0.1 per metre (signed), damage 0.001 per point, -10 on failure, no control cost, gamma 0.995.
# Training's steering noise doesn't depend on the speed, and the std is one number for every state (--no-learned-std,
# while train.py now learns it per state): the setup these sweeps' "base" runs were made with.
BASE = {
    "lost-control-cost": 0.0,
    "best-only-progress": False,
    "terminal": 10.0,
    "gamma": 0.995,
    "progress-units": "metres",
    "lost-control-causes": ("contact", "backwards", "far_off"),
    "steer-noise-power": 0.0,
    "steer-noise-speed": 20.0,
    "steer-noise-floor": 0.1,
    "learned-std": False,
    "obs-height": "none",
    "gearbox": "manual",
    "gae-lambda": 0.95,
    "fixed-gear-inputs": False,
    "pedal-std-floor": 0.0,
    "steer-std-floor": 0.0,
    "steer-noise-straight": 0.0,
    "weave-cost": 0.0,
    "stuck-cost": 0.0,
    "idle-cost": 0.0,
    "crashed-share": 0.0,
    "std-max": 0.0,
    "noise-beta": 0.0,
    "noise-dims": ("pedal",),
    "wall-cost": 0.0,
    "offroad-cost": 0.0,
    "width-bonus": 0.0,
    "width-fade": 0,
    "auto-upshift": 0.92,
}

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
    "cost2-nobackwards": {
        "lost-control-cost": 2.0,
        "lost-control-causes": ("contact", "far_off"),
    },
    # progress in the bot's seconds (the next planned test), alone and with a cost
    "botsec": {"progress-units": "bot_seconds"},
    "botsec-cost1": {"progress-units": "bot_seconds", "lost-control-cost": 1.0},
    # the sweep's lead (botsec-cost1) with best-only progress, which raised crashed completion on its own
    "botsec-best-cost1": {
        "progress-units": "bot_seconds",
        "best-only-progress": True,
        "lost-control-cost": 1.0,
    },
    # steering noise that shrinks with speed above 20 m/s (std x 20 / v, or x (20 / v)^2, never below x 0.1): is the noise at speed what
    # holds the steering back? The steering's rate is unchanged
    "steernoise-v1": {"steer-noise-power": 1.0},
    "steernoise-v2": {"steer-noise-power": 2.0},
    # the std of steer and pedal learned per state (an output of the actor), no speed scaling: can it keep v1's crashed
    # recovery without v2's loss in nominal driving?
    "learnedstd": {"learned-std": True},
    # the learned std with progress in the bot's seconds: more reward pressure where the car is slow, where the learned
    # std stays high (no pressure on it there)
    "learnedstd-botsec": {"learned-std": True, "progress-units": "bot_seconds"},
    # 2026-11-24: new names after the launch fix (speed along the road's slope) and wall-only contacts, so these runs
    # never mix with the ones before: the new baseline (learned std), and the road's height in the observation
    "fix-learnedstd": {"learned-std": True},
    "fix-height-centre": {"learned-std": True, "obs-height": "centre"},
    # 2026-11-24 diagnostic: the car sits in 3rd (its limiter caps it at 166 km/h) and barely brakes. With rpm-based
    # shifting while moving, does it go flat out (the gearbox was the trap) or still hold ~166 (the reward's trade-off)?
    "fix-autoshift": {"learned-std": True, "gearbox": "auto"},
    # 2026-11-24 restart (old runs deleted): train.py's defaults, measured with the lean eval (nominal + crashed + a
    # capped grid lap per track) and the sim-clock pace. Every later variant is compared with this
    "baseline": {"learned-std": True},
    # 2026-11-24 speed round, each lever alone on top of baseline (the car won't accelerate into high speed)
    # GAE lambda 0.95 credits ~0.76 s ahead, 0.98 ~1.6 s: braking and the gain from accelerating play out over seconds
    "lambda98": {"learned-std": True, "gae-lambda": 0.98},
    # the gear one-hots and the shift cooldown skip the input normalization (a rare gear read +7 sd when it engaged)
    "gearfixed": {"learned-std": True, "fixed-gear-inputs": True},
    # the pedal's learned std never below 0.1 (it collapsed to ~0.05 at speed; pace followed how fast it did)
    "pedalfloor": {"learned-std": True, "pedal-std-floor": 0.1},
    # colored pedal noise, beta 0.5 (Hollenstein et al. 2024): a 1 s average of the noise is 2x white noise's
    "colored": {"learned-std": True, "noise-beta": 0.5, "noise-dims": ("pedal",)},
    # a wall costs speed^2 (Fuchs et al. 2021): 1 per step of wall damage at 40 m/s = 10 m of progress
    "wallcost": {"learned-std": True, "wall-cost": 1.0},
    # follow-ups, picked from the seed-1 results: stronger / broader colored noise, a higher pedal floor, the wall
    # cost halved or doubled
    "colored-pink": {"learned-std": True, "noise-beta": 1.0, "noise-dims": ("pedal",)},
    "colored-both": {
        "learned-std": True,
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
    },
    "pedalfloor02": {"learned-std": True, "pedal-std-floor": 0.2},
    "wallcost05": {"learned-std": True, "wall-cost": 0.5},
    "wallcost2": {"learned-std": True, "wall-cost": 2.0},
    # 2026-11-24 combination (user's pick): the rpm gearbox (the policies' shift habits govern speed; correct upshifts
    # alone are worth ~0-0.025 pace), colored pedal noise (flat out, bot-like braking), the wall cost (walls ~free)
    # colored-both weaves on straights (steer sd 2x, lateral sd 1.0 vs 0.45 m) but is the best in corners: its
    # correlated steering noise faded with speed, std x min(1, 30 / v): full below 108 km/h, half at 216
    "colored-both-fade": {
        "learned-std": True,
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
        "steer-noise-power": 1.0,
        "steer-noise-speed": 30.0,
    },
    # the rpm gearbox + colored pedal noise: autoshift alone moved the speed governor to the pedal (lifting), colored
    # noise is what stopped the lifting
    "auto-colored": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("pedal",),
    },
    # colored pedal noise that can't collapse either: colored kept its pedal std ~0.19 by itself, floor 0.2 alone was
    # the safest arm (crashed completion +0.15) without colored's driving
    "colored-floor02": {
        "learned-std": True,
        "noise-beta": 0.5,
        "noise-dims": ("pedal",),
        "pedal-std-floor": 0.2,
    },
    # 2026-11-25: sw30m-colored-floor02 s1 was ahead of every 10M arm already at its 10M checkpoint, where its
    # learning rate was still 2/3 of the start. Two 10M tests: the floor on top of colored (colored-floor02 at 10M),
    # and the 30M run's schedule cut at 10M (seed 1 = that run's first 10M: the learning rate stays above 2e-4)
    "colored-floor02-a30": {
        "learned-std": True,
        "noise-beta": 0.5,
        "noise-dims": ("pedal",),
        "pedal-std-floor": 0.2,
        "anneal-timesteps": 30_000_000,
    },
    # more pressure (user's pick): the auto gearbox + colored + floor, walls cost speed^2, and each metre driven with
    # four wheels off costs 0.01 per metre beyond the edge (Spa's runoffs; reward.py "offroad")
    "auto-colored-pressure": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("pedal",),
        "pedal-std-floor": 0.2,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
    },
    # 2026-11-25: every policy enters corners from the middle; a bonus for using the width, fading out at mid-run
    # (meant for 30M: --width-fade 15M). And the auto box's upshift at 0.995 of the redline instead of 0.92
    "auto-colored-pressure-width": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("pedal",),
        "pedal-std-floor": 0.2,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "width-bonus": 0.004,
        "width-fade": 15_000_000,
    },
    # 2026-11-25: pressure + v1's steering exploration (std >= 0.1; learned std sits on 0.03 at every speed)
    "auto-colored-pressure-steer01": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("pedal",),
        "pedal-std-floor": 0.2,
        "steer-std-floor": 0.1,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "anneal-timesteps": 30_000_000,
    },  # 10M = pressure 30M's first 10M
    # 2026-11-25 backup if steer01 flops: colored noise on both (best corners, but weaving straights) + both floors,
    # and either a yaw-rate cost on long straights or the steering noise cut to 1/4 there. Same 30M schedule
    "auto-both-pressure-weave25": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
        "pedal-std-floor": 0.2,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "weave-cost": 0.25,
        "anneal-timesteps": 30_000_000,
    },
    "auto-both-pressure-gate": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
        "pedal-std-floor": 0.2,
        "steer-std-floor": 0.1,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "steer-noise-straight": 0.25,
        "anneal-timesteps": 30_000_000,
    },
    "auto-both-pressure-gate05": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
        "pedal-std-floor": 0.2,
        "steer-std-floor": 0.1,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "steer-noise-straight": 0.05,
        "anneal-timesteps": 30_000_000,
    },
    # 2026-11-25: gate05 + v1's std cap (0.5 on both): the learned std at standstill climbs to 1.65 after a crash and
    # the mean the eval drives stays stopped (most of gate05's no-progress ends)
    "auto-both-pressure-gate05-cap05": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
        "pedal-std-floor": 0.2,
        "steer-std-floor": 0.1,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "steer-noise-straight": 0.05,
        "std-max": 0.5,
        "anneal-timesteps": 30_000_000,
    },
    # 2026-11-25 crashed tier: cap05 30M fails 202/439 crashed scenarios, 176 never move, nearly all facing backwards
    # (ok 0.03 at 150-180 deg; 0.45 at its 14M checkpoint: learnt, then lost). Same seed = identical to cap05 until the
    # change bites. More practice (8% -> 20% of starts, from nominal) vs sitting still costing more
    "auto-both-pressure-gate05-cap05-crash20": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
        "pedal-std-floor": 0.2,
        "steer-std-floor": 0.1,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "steer-noise-straight": 0.05,
        "std-max": 0.5,
        "crashed-share": 0.2,
        "anneal-timesteps": 30_000_000,
    },
    "auto-both-pressure-gate05-cap05-stuck05": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
        "pedal-std-floor": 0.2,
        "steer-std-floor": 0.1,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "steer-noise-straight": 0.05,
        "std-max": 0.5,
        "stuck-cost": 0.5,
        "anneal-timesteps": 30_000_000,
    },
    # stuck05 was ~7% slower (a crash cost in disguise): charge standing still instead, not the way out
    "auto-both-pressure-gate05-cap05-idle05": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
        "pedal-std-floor": 0.2,
        "steer-std-floor": 0.1,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "steer-noise-straight": 0.05,
        "std-max": 0.5,
        "idle-cost": 0.5,
        "anneal-timesteps": 30_000_000,
    },
    # 2026-11-26 overnight: idle05 (10M: every lap finished, 32/32 train + 6/6 validation, pace kept) on a long schedule
    "auto-both-pressure-gate05-cap05-idle05-long": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("steer", "pedal"),
        "pedal-std-floor": 0.2,
        "steer-std-floor": 0.1,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "steer-noise-straight": 0.05,
        "std-max": 0.5,
        "idle-cost": 0.5,
        "anneal-timesteps": 100_000_000,
    },
    # 2026-11-25: the cap alone on pressure (colored pedal only, no steer floor, no gate): is the cap the whole win?
    "auto-colored-pressure-cap05": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("pedal",),
        "pedal-std-floor": 0.2,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "std-max": 0.5,
        "anneal-timesteps": 30_000_000,
    },
    "auto-colored-pressure-up995": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("pedal",),
        "pedal-std-floor": 0.2,
        "wall-cost": 1.0,
        "offroad-cost": 0.01,
        "auto-upshift": 0.995,
    },
    "auto-colored-wall": {
        "learned-std": True,
        "gearbox": "auto",
        "noise-beta": 0.5,
        "noise-dims": ("pedal",),
        "wall-cost": 1.0,
    },
}


def prefix(steps: int) -> str:
    return "sw" if steps == STEPS else f"sw{steps / 1e6:g}m"


def command(variant: str, seed: int, steps: int) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--exp-name",
        f"{prefix(steps)}-{variant}",
        "--seed",
        str(seed),
        "--total-timesteps",
        str(steps),
        "--anneal-timesteps",
        str(VARIANTS[variant].get("anneal-timesteps", max(ANNEAL, steps))),
    ]
    if steps > STEPS:
        cmd += ["--eval-every", str(LONG_EVAL_EVERY)]
    for k, v in (BASE | VARIANTS[variant]).items():
        if k == "anneal-timesteps":  # a variant's own learning-rate schedule, set above
            continue
        if isinstance(v, bool):
            cmd.append(f"--{k}" if v else f"--no-{k}")
        elif isinstance(v, tuple):
            cmd += [f"--{k}", *v]
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def finished(variant: str, seed: int, steps: int = STEPS) -> list[str]:
    """Final eval CSVs of this variant and seed (the eval at the last iteration is the one named after it)."""
    return [
        p
        for d in glob.glob(f"runs/{prefix(steps)}-{variant}__{seed}__*")
        for p in glob.glob(f"{d}/eval/*.csv")
        if Path(p).stem.isdigit() and int(Path(p).stem) >= steps - 20_000
    ]


def run(cmd: list[str], log_path: Path) -> int:
    """Runs one training: its whole output to the log and, on a terminal, its progress bar (train.py's tqdm, desc
    "train") redrawn on the current line. Returns the exit code."""
    show = sys.stdout.isatty()
    decoder = codecs.getincrementaldecoder("utf-8")(
        errors="replace"
    )  # a bar's blocks are 3 bytes, chunks can split them
    pending, shown = "", False
    env = os.environ | {
        "PYTHONUNBUFFERED": "1"
    }  # the log in the order it was written, the bar as soon as it changes
    with (
        open(log_path, "wb") as log,
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        ) as p,
    ):
        try:
            while chunk := os.read(p.stdout.fileno(), 65536):
                log.write(chunk)
                log.flush()
                if not show:
                    continue
                *done, pending = re.split(r"[\r\n]", pending + decoder.decode(chunk))
                bars = [
                    s for s in (*done, pending) if s.startswith("train:")
                ]  # tqdm writes each redraw in one go
                if bars:
                    width = shutil.get_terminal_size().columns
                    sys.stdout.write("\r\x1b[K" + bars[-1][: width - 1])
                    sys.stdout.flush()
                    shown = True
            return p.wait()
        except (
            BaseException
        ):  # as subprocess.call does: don't leave the training running
            p.kill()
            raise
        finally:
            if shown:
                sys.stdout.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(VARIANTS),
        help="variants to run (default: all)",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument(
        "--steps",
        type=int,
        default=STEPS,
        help="run length (3M: a 10M schedule's start; more: full schedule)",
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    variants = a.only or list(VARIANTS)
    todo = [(v, s) for s in a.seeds for v in variants if not finished(v, s, a.steps)]
    out = Path("runs/sweep")
    out.mkdir(parents=True, exist_ok=True)
    per_run = 8.5 * a.steps / STEPS  # minutes, a guess until this sweep has timed a run
    print(
        f"{len(todo)} runs to go ({len(variants) * len(a.seeds) - len(todo)} done), ~{len(todo) * per_run / 60:.1f} h"
    )
    timed = []  # minutes of this sweep's finished runs
    for i, (v, s) in enumerate(todo):
        cmd = command(v, s, a.steps)
        if a.dry_run:
            print(" ".join(cmd[1:]))
            continue
        t0 = time.time()
        name = v if a.steps == STEPS else f"{prefix(a.steps)}-{v}"
        estimate = sum(timed) / len(timed) if timed else per_run
        end = time.strftime(
            "%H:%M", time.localtime(t0 + (len(todo) - i) * estimate * 60)
        )
        print(
            f"[{time.strftime('%H:%M')}] {i + 1}/{len(todo)} {name} seed {s}   (sweep done ~{end})",
            flush=True,
        )
        code = run(cmd, out / f"{name}-s{s}.log")
        minutes = (time.time() - t0) / 60
        if code == 0:  # a crash's minutes say nothing about a run's length
            timed.append(minutes)
        with open(out / "status.tsv", "a") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M')}\t{name}\t{s}\t{code}\t{minutes:.1f}\n"
            )
        print(
            f"         exit {code}, {minutes:.1f} min"
            + ("  <- FAILED, see the log" if code else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
