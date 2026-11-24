"""Results of the reward sweep (scripts/sweep.py): each variant's metrics over its seeds, and its difference from `base`
measured against the noise between seeds.

    uv run python scripts/sweep_report.py [--steps 10000000] [--split train] [--base learnedstd] [--tsv report.tsv]

How to read it: one run's result moves a lot from seed to seed (the overnight 3M sweep, 14 variants x 3 seeds: sd 0.055
in validation nominal survival, 0.11 in crashed completion, 0.027 in pace). A scenario-by-scenario sign test between two
runs (compare.py) says whether two *policies* differ, which two seeds of one setup already do (p < 0.05 in 11 of 42
same-setup pairs): it can't say whether two *setups* differ. So the comparison here is over seeds: z = (variant mean -
base mean) / its standard error, using the seed sd pooled over every variant in the report. |z| >= 2 is a real
difference; 1-2 is a hint. With 3 seeds each, that takes ~0.09 in nominal survival.

"surv vs pace" is nominal survival at matched pace: every reward change so far moved the car along one line between
pace and survival (a slower car survives more; at 10M, validation survival ~ 1.09 - 0.47 x pace), never above it. The
line is fitted on this report's reward-only variants (one point per seed; it needs two of them, base included), and the
column is each run's survival minus the line at its pace: > 0 = safer than its pace alone explains, i.e. more skill
rather than more caution. Without two reward-only variants at these steps, the column is left out.

--curve prints the mean over seeds at every evaluation of the run (runs longer than 3M are evaluated every 2M).
"""

import argparse
from pathlib import Path

import numpy as np

from sweep import SEEDS, STEPS, VARIANTS, finished
from rlpilot.evaluate import read_results, summarize

METRICS = [  # (column, summarize key with {s} = split)
    ("nominal surv", "{s}/nominal/survival"),
    ("nominal compl", "{s}/nominal/completion"),
    ("nominal clean", "{s}/nominal/clean"),
    ("mild surv", "{s}/mild/survival"),
    ("disturbed surv", "{s}/disturbed/survival"),
    ("crashed compl", "{s}/crashed/completion"),
    ("pace", "{s}/pace"),
    ("lost contact", "{s}/nominal/lost_contact"),
    ("lost far off", "{s}/nominal/lost_far_off"),
]
Z_METRICS = ("nominal surv", "nominal compl", "disturbed surv", "crashed compl", "pace")
MATCHED = ("surv vs pace", "{s}/nominal/survival_vs_pace")
NOISE_KEYS = {"steer-noise-power", "steer-noise-speed", "steer-noise-floor", "learned-std"}
REWARD_ONLY = [v for v, cfg in VARIANTS.items() if not NOISE_KEYS & cfg.keys()]  # the training noise as in base


def cell(x: np.ndarray) -> str:
    return f"{np.nanmean(x):6.3f} ({np.nanstd(x, ddof=1):.3f})" if len(x) > 1 else f"{x[0]:6.3f}        "


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=STEPS, help="which runs: the 3M sweep (default) or longer ones")
    p.add_argument("--split", default="validation", choices=["validation", "train"])
    p.add_argument("--curve", action="store_true", help="also show every evaluation along the runs")
    p.add_argument("--base", default="base", choices=list(VARIANTS), help="the reference of the differences")
    p.add_argument("--tsv", type=Path, help="also write the per-seed metrics here")
    a = p.parse_args()
    runs = {}  # (variant, seed) -> final eval CSV
    for v in VARIANTS:
        for s in SEEDS:
            csvs = sorted(finished(v, s, a.steps))
            if csvs:
                runs[v, s] = Path(csvs[-1])  # the latest run if a seed was run twice
    if not runs:
        raise SystemExit(f"no finished runs of {a.steps} steps yet")
    metrics = {k: summarize(read_results(path)) for k, path in runs.items()}
    variants = [v for v in VARIANTS if any((v, s) in metrics for s in SEEDS)]

    lines = {}  # split -> (slope, intercept) of nominal survival over pace, fitted on the reward-only runs
    fitted = [v for v in variants if v in REWARD_ONLY]
    if len(fitted) >= 2:
        for split in ("validation", "train"):
            pts = np.array([(m[f"{split}/pace"], m[f"{split}/nominal/survival"]) for (v, _), m in metrics.items()
                            if v in fitted])
            lines[split] = np.polyfit(pts[:, 0], pts[:, 1], 1)
            for m in metrics.values():
                expected = np.polyval(lines[split], m[f"{split}/pace"])
                m[MATCHED[1].format(s=split)] = m[f"{split}/nominal/survival"] - expected
    columns = METRICS + [MATCHED] if lines else METRICS
    z_columns = Z_METRICS + (MATCHED[0],) if lines else Z_METRICS

    def values(v: str, key: str) -> np.ndarray:
        return np.array([metrics[v, s].get(key.format(s=a.split), np.nan) for s in SEEDS if (v, s) in metrics])

    print(f"{a.split}, {a.steps / 1e6:g}M steps: mean over seeds (sd), n = seeds done\n")
    print(f"  {'variant':18s} {'n':>2s} " + " ".join(f"{c:>15s}" for c, _ in columns))
    for v in variants:
        cells = " ".join(f"{cell(values(v, k)):>15s}" for _, k in columns)
        print(f"  {v:18s} {len(values(v, METRICS[0][1])):2d} {cells}")
    if lines:
        b, c0 = lines[a.split]
        print(f"\n  surv vs pace: nominal survival minus {c0:.2f} {b:+.2f} x pace, the line of the reward-only runs"
              f" ({', '.join(fitted)})")
    else:
        print("\n  surv vs pace left out: fewer than two reward-only variants (base included) at these steps")

    keys = dict(columns)
    pooled = {}
    for c in z_columns:
        var = [np.nanvar(values(v, keys[c]), ddof=1) for v in variants if len(values(v, keys[c])) > 1]
        pooled[c] = float(np.sqrt(np.mean(var))) if var else np.nan
    if a.base in variants and len(variants) > 1:
        print(f"\n  difference from {a.base}, z = difference / its standard error (seed sd pooled over the variants"
              f" above: {', '.join(f'{c} {pooled[c]:.3f}' for c in z_columns)})\n")
        print(f"  {'variant':18s} " + " ".join(f"{c:>20s}" for c in z_columns))
        b = {c: values(a.base, keys[c]) for c in z_columns}
        for v in variants:
            if v == a.base:
                continue
            cells = []
            for c in z_columns:
                x = values(v, keys[c])
                d = np.nanmean(x) - np.nanmean(b[c])
                se = pooled[c] * np.sqrt(1 / len(x) + 1 / len(b[c]))
                cells.append(f"{d:+.3f} (z {d / se:+.1f})")
            print(f"  {v:18s} " + " ".join(f"{c:>20s}" for c in cells))

    if a.curve:
        print(f"\n  {a.split} along the runs, mean over seeds: nominal survival / crashed completion / pace\n")
        curves = {}
        for (v, s), path in runs.items():
            for csv in sorted(path.parent.glob("*.csv")):
                if csv.stem.isdigit():
                    m = summarize(read_results(csv))
                    curves.setdefault(v, {}).setdefault(int(csv.stem), []).append(
                        [m[k.format(s=a.split)] for k in (keys["nominal surv"], keys["crashed compl"], keys["pace"])])
        for v in variants:
            row = [f"{step / 1e6:.1f}M " + "/".join(f"{x:.3f}" for x in np.nanmean(np.array(vals), axis=0))
                   for step, vals in sorted(curves[v].items())]
            print(f"  {v:18s} " + "   ".join(row))

    if a.tsv:
        keys_all = sorted({k for m in metrics.values() for k in m})
        with open(a.tsv, "w") as f:
            f.write("variant\tseed\t" + "\t".join(keys_all) + "\n")
            for (v, s), m in metrics.items():
                f.write(f"{v}\t{s}\t" + "\t".join(f"{m.get(k, float('nan')):.4f}" for k in keys_all) + "\n")
        print(f"\nper-seed metrics in {a.tsv}")


if __name__ == "__main__":
    main()
