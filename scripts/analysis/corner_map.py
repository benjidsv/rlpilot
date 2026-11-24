"""The failure map: where policies lose control, by corner (data/corners-v2.csv, made by scripts/setup/make_corners.py).

    uv run python scripts/analysis/corner_map.py run      # play the policies below on the suite + the corner scenarios
    uv run python scripts/analysis/corner_map.py report   # the map, from runs/corner-map/*.csv

`run` plays each policy's final checkpoint on the validation + train suite and on data/corner-suite-v2.csv (one
Suite, workers kept up between policies) and writes runs/corner-map/<run>.csv. Test tracks are never played.
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

from rlpilot import evaluate
from rlpilot.client import DATA
from rlpilot.tracks import Track

sys.path.insert(0, str(Path(__file__).parents[1]))  # scripts/: eval.py
from eval import load_policy  # noqa: E402

OUT = Path("runs/corner-map")
POLICIES = [f"{p}-{v}" for p in ("sw10m", "sw30m") for v in ("base", "learnedstd")]
CORNER_SUITE = DATA / "corner-suite-v2.csv"


def final_checkpoint(run_dir: str) -> Path:
    return sorted(p for p in Path(run_dir, "checkpoints").glob("*.pt") if p.stem.isdigit())[-1]


def run(a) -> None:
    runs = sorted(d for p in POLICIES for d in glob.glob(f"runs/{p}__*"))
    scenarios = evaluate.load_suite(("validation", "train")) + evaluate.load_suite(("validation", "train"),
                                                                                    CORNER_SUITE)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [r for r in runs if a.force or not (OUT / f"{Path(r).name}.csv").exists()]
    print(f"{len(scenarios)} scenarios, {len(todo)} of {len(runs)} policies to play")
    suite = evaluate.Suite(scenarios, a.workers)
    try:
        for r in todo:
            ckpt = final_checkpoint(r)
            policy, _ = load_policy(ckpt)
            rows = suite.run(policy)
            for row, sc in zip(rows, scenarios):
                row["corner"], row["lead"], row["goal"] = sc.get("corner", ""), sc.get("lead", ""), sc["goal"]
            evaluate.write_results(rows, OUT / f"{Path(r).name}.csv")
            print(f"  {Path(r).name} ({ckpt.stem})", flush=True)
    finally:
        suite.close()


def main():
    global OUT
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=48)
    r.add_argument("--force", action="store_true", help="replay policies already played")
    sub.add_parser("report")
    p.add_argument("--out", type=Path, default=Path("runs/corner-map"), help="where the results go / are read from")
    a = p.parse_args()
    OUT = a.out
    {"run": run, "report": report}[a.cmd](a)


# ---- the report ----

SETTLE = {"nominal": 0.0, "mild": 0.0, "perturbed": 100.0, "disturbed": 100.0}  # m before a corner counts
GROUP = {"nominal": "nominal", "mild": "nominal", "perturbed": "disturbed", "disturbed": "disturbed"}
KEY_LENGTH = 350.0  # m of a corner's window a scenario must cover to count as a passage (a few complexes are longer)
NUMERIC = ["bot_entry_speed", "bot_min_speed", "bot_drop", "brake_distance", "bot_decel", "turn_deg", "min_radius",
           "length", "half_width", "bot_steer_max", "approach_slope", "height_change", "vertical_curv", "bank_deg",
           "runoff_out", "grass_out", "loose_out", "gap_before", "prev_bot_min_speed", "rarity"]
CATEGORIES = ["braking", "direction", "chained", "prev_same_dir", "split"]
MODEL = ["bot_entry_speed", "bot_drop", "turn_deg", "log_radius", "half_width", "approach_slope", "height_change",
         "vertical_curv", "bank_deg", "runoff_out", "grass_out", "loose_out", "chained", "prev_same_dir", "left",
         "braking", "rarity"]
RARITY = ["bot_entry_speed", "bot_min_speed", "turn_deg", "log_radius", "length"]  # the corner's shape and speeds


def _num(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return {"True": 1.0, "False": 0.0}.get(x, float("nan"))


def _load_corners() -> list[dict]:
    rows = list(csv.DictReader(open(DATA / "corners-v2.csv")))
    out = []
    for r in rows:
        c = {k: (_num(v) if k not in ("track", "direction", "split", "id") else v) for k, v in r.items()}
        c["chained"] = float(c["gap_before"] < 50)
        c["left"] = float(c["direction"] == "left")
        c["log_radius"] = float(np.log(c["min_radius"]))
        out.append(c)
    # rarity: distance (in sd units of the shape features) to the 5th nearest training corner, the corner itself
    # excluded. High = few training corners like it
    X = np.array([[c[f] for f in RARITY] for c in out])
    X = (X - X.mean(0)) / X.std(0)
    train = np.array([c["split"] == "train" for c in out])
    for i, c in enumerate(out):
        d = np.sort(np.linalg.norm(X[train] - X[i], axis=1))
        c["rarity"] = float(d[5] if c["split"] == "train" else d[4])  # d[0] = itself for a training corner
    return out


def _policy(name: str) -> tuple[str, str, int]:
    """runs/corner-map/<run>.csv -> (setup, length, seed)."""
    exp, seed, _ = name.split("__")
    length, setup = exp.split("-", 1)
    return setup, length.removeprefix("sw"), int(seed)


def _passages(corners: list[dict]) -> tuple[list[dict], dict]:
    """One record per (policy, scenario, corner the scenario passes cleanly into): did it lose control there, did it
    then fail. Also, per policy, losses of control outside every corner's window."""
    meta = {r["id"]: r for r in csv.DictReader(open(evaluate.SUITE))}
    meta.update({r["id"]: r for r in csv.DictReader(open(CORNER_SUITE))})
    by_track: dict[str, list[int]] = {}
    for i, c in enumerate(corners):
        by_track.setdefault(c["track"], []).append(i)
    length = {t: Track.load(t).length for t in by_track}

    def owner(track: str, s: float) -> int:
        """The corner whose window holds track position s (the latest window start before it), or -1."""
        best, best_off = -1, np.inf
        for i in by_track[track]:
            off = (s - corners[i]["window_start"]) % length[track]
            if off <= corners[i]["window_length"] and off < best_off:
                best, best_off = i, off
        return best

    passages, outside = [], {}
    for path in sorted(OUT.glob("*__*.csv")):
        setup, steps, seed = _policy(path.stem)
        for r in evaluate.read_results(path):
            if r["tier"] not in SETTLE or r["track"] not in by_track:
                continue
            m = meta[r["id"]]
            s0, goal, L = float(m["s"]), float(m.get("goal") or evaluate.SCENARIO_DISTANCE), length[r["track"]]
            lost_at = float(r["lost_at"]) if r["lost_at"] != "" else np.inf
            lost_owner = owner(r["track"], float(r["lost_s"])) if r["lost_at"] != "" else -1
            if r["lost_at"] != "" and lost_owner < 0 and lost_at >= SETTLE[r["tier"]]:
                outside[path.stem] = outside.get(path.stem, 0) + 1
            peak = float(r["peak"])
            for i in by_track[r["track"]]:
                c = corners[i]
                o0 = (c["window_start"] - s0) % L
                if o0 < SETTLE[r["tier"]] or o0 + min(c["window_length"], KEY_LENGTH) > goal:
                    continue
                if lost_at < o0 or peak < o0:  # lost control before it, or never got there
                    continue
                event = lost_owner == i and lost_at <= o0 + c["window_length"]
                passages.append({"corner": i, "policy": path.stem, "setup": setup, "steps": steps, "seed": seed,
                                 "group": GROUP[r["tier"]], "source": "corner" if m.get("corner") else "suite",
                                 "event": event, "stuck": event and r["outcome"] == "failed",
                                 "lost_speed": float(r["lost_speed"]) if event else np.nan,
                                 "lost_s": float(r["lost_s"]) if event else np.nan, "row": r,
                                 "aimed": m.get("corner") == c["id"]})
    return passages, outside


def _rate(ps: list[dict]) -> str:
    n, k = len(ps), sum(p["event"] for p in ps)
    return f"{k / n:6.3f} ({k:4d}/{n:5d})" if n else f"{'':6s} ({0:4d}/{0:5d})"


def _logistic(X: np.ndarray, k: np.ndarray, n: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Binomial logistic regression (k events of n per row) by IRLS with a small ridge; returns [intercept, coefs]."""
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    R = ridge * np.eye(A.shape[1])
    R[0, 0] = 0
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ w))
        W = n * p * (1 - p) + 1e-9
        g = A.T @ (k - n * p) - R @ w
        H = (A * W[:, None]).T @ A + R
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def report(a) -> None:
    corners = _load_corners()
    passages, outside = _passages(corners)
    if not passages:
        raise SystemExit(f"no results in {OUT}: run `corner_map.py run` first")
    setups = sorted({p["setup"] for p in passages})
    policies = sorted({p["policy"] for p in passages})
    nominal = [p for p in passages if p["group"] == "nominal"]
    print(f"{len(corners)} corners ({sum(c['braking'] for c in corners):.0f} braking zones), {len(policies)} policies, "
          f"{len(passages)} passages (a scenario arriving at a corner in control)\n")

    print("Loss of control per passage, by setup (columns) | rate (losses / passages)")
    head = "  " + f"{'':34s}" + "".join(f"{s:>22s}" for s in setups + ["all"])
    print(head)
    def line(label, ps):
        print(f"  {label:34s}" + "".join(f"{_rate([p for p in ps if p['setup'] == s]):>22s}" for s in setups)
              + f"{_rate(ps):>22s}")
    for g in ("nominal", "disturbed"):
        for src in ("suite", "corner"):
            ps = [p for p in passages if p["group"] == g and p["source"] == src]
            line(f"{g} starts, {src} scenarios", ps)
    for steps in ("10m", "30m"):
        line(f"nominal, {steps} runs", [p for p in nominal if p["steps"] == steps])
    for flag, name in ((1.0, "braking zones"), (0.0, "flat out")):
        line(f"nominal, {name}", [p for p in nominal if corners[p["corner"]]["braking"] == flag])
    for sp in ("validation", "train"):
        line(f"nominal, {sp} corners", [p for p in nominal if corners[p["corner"]]["split"] == sp])
    ev = [p for p in passages if p["event"]]
    print(f"\n  losses outside every corner window (straights), per policy: "
          f"{np.mean(list(outside.values())) if outside else 0:.1f} vs {len(ev) / len(policies):.1f} in corners")

    print("\nBy corner feature (nominal + mild starts, suite and corner scenarios; corners binned by quartile).")
    print("  rate per setup | train corners in the bin | median pace through the corner (corner scenarios, clean)")
    for f in NUMERIC + CATEGORIES:
        vals = np.array([_num(str(c[f])) if f not in ("direction", "split") else 0 for c in corners])
        if f in ("direction", "split"):
            labels = sorted({c[f] for c in corners})
            bins = [(lab, lambda c, lab=lab: c[f] == lab) for lab in labels]
        elif f in CATEGORIES:
            bins = [(lab, lambda c, v=v: c[f] == v) for lab, v in (("no", 0.0), ("yes", 1.0))]
        else:
            q = np.nanpercentile(vals, [25, 50, 75])
            edges = [-np.inf, *q, np.inf]
            bins = [(f"{lo:.4g}..{hi:.4g}", lambda c, lo=lo, hi=hi: lo < c[f] <= hi or (lo == -np.inf and c[f] <= hi))
                    for lo, hi in zip(edges[:-1], edges[1:])]
        print(f"\n  {f}")
        for lab, test in bins:
            ps = [p for p in nominal if test(corners[p["corner"]])]
            ntrain = sum(test(c) and c["split"] == "train" for c in corners)
            pace = [float(p["row"]["pace"]) for p in ps if p["aimed"] and p["row"]["outcome"] in ("clean", "survived")]
            print(f"    {lab:22s}" + "".join(f"{_rate([p for p in ps if p['setup'] == s]):>22s}" for s in setups)
                  + f"   train corners {ntrain:4d}   pace {np.median(pace) if pace else np.nan:.3f}")

    # the model: every feature at once, corner-level counts pooled over policies and nominal starts
    X = np.array([[c[f] for f in MODEL] for c in corners], np.float64)
    X = np.where(np.isfinite(X), X, np.nanmean(X, 0))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    k = np.zeros(len(corners))
    n = np.zeros(len(corners))
    for p in nominal:
        n[p["corner"]] += 1
        k[p["corner"]] += p["event"]
    keep = n > 0
    w = _logistic(Z[keep], k[keep], n[keep])
    tracks = sorted({c["track"] for c in corners})
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(200):  # resample whole tracks: corners of one track aren't independent
        pick = rng.choice(len(tracks), len(tracks))
        idx = np.concatenate([[i for i, c in enumerate(corners) if c["track"] == tracks[j] and keep[i]] for j in pick])
        boot.append(_logistic(Z[idx], k[idx], n[idx]))
    lo, hi = np.percentile(np.array(boot), [5, 95], axis=0)
    print("\nAll features at once: logistic regression of a corner's loss rate (nominal starts, all policies), per sd of"
          "\neach feature; 90% interval from resampling tracks. exp(coef) = odds ratio per sd.")
    for j in np.argsort(-np.abs(w[1:])):
        flag = "  *" if lo[j + 1] > 0 or hi[j + 1] < 0 else ""
        print(f"  {MODEL[j]:18s} {w[j + 1]:+6.2f}  [{lo[j + 1]:+6.2f}, {hi[j + 1]:+6.2f}]{flag}")

    # consistency: do the same corners fail for every policy?
    per = {pol: np.full(len(corners), np.nan) for pol in policies}
    for pol in policies:
        kk, nn = np.zeros(len(corners)), np.zeros(len(corners))
        for p in nominal:
            if p["policy"] == pol:
                nn[p["corner"]] += 1
                kk[p["corner"]] += p["event"]
        per[pol] = np.where(nn >= 3, kk / np.maximum(nn, 1), np.nan)
    def corr(A, B):
        m = np.isfinite(A) & np.isfinite(B)
        return np.corrcoef(A[m], B[m])[0, 1] if m.sum() > 10 else np.nan
    same = [corr(per[x], per[y]) for i, x in enumerate(policies) for y in policies[i + 1:]
            if _policy(x)[0] == _policy(y)[0]]
    cross = [corr(per[x], per[y]) for i, x in enumerate(policies) for y in policies[i + 1:]
             if _policy(x)[0] != _policy(y)[0]]
    print(f"\nSame corners, different policies: correlation of per-corner loss rates (corners with >= 3 passages)"
          f"\n  same setup {np.nanmean(same):.2f}, base vs learned std {np.nanmean(cross):.2f}")

    fails = np.array([np.nanmean([per[pol][i] > 0.5 for pol in policies if np.isfinite(per[pol][i])])
                      if any(np.isfinite(per[pol][i]) for pol in policies) else np.nan for i in range(len(corners))])
    print("\nCorners that most policies fail (share of policies losing control in > half their passages there):")
    print(f"  {'corner':22s} {'split':10s} {'pols':>5s} {'rate':>6s} brk  turn  R    entry->min  drop  slope   dz    vc    bank runoff gap")
    for i in np.argsort(-np.nan_to_num(fails, nan=-1))[:30]:
        if not fails[i] > 0:
            break
        c = corners[i]
        print(f"  {c['id']:22s} {c['split']:10s} {fails[i]:5.2f} {k[i] / max(n[i], 1):6.2f} {'y' if c['braking'] else 'n':3s}"
              f" {c['turn_deg']:5.0f} {c['min_radius']:4.0f} {c['bot_entry_speed']:5.0f}->{c['bot_min_speed']:<4.0f}"
              f" {c['bot_drop']:5.0f} {c['approach_slope']:+.3f} {c['height_change']:+5.1f} {c['vertical_curv'] * 1e4:+5.1f}"
              f" {c['bank_deg']:+5.1f} {c['runoff_out']:5.1f} {c['gap_before']:4.0f}")

    # how losses end: recovered or stuck, and where it stopped
    print("\nAfter a loss of control at a corner (nominal starts): share that then failed (stuck), by setup")
    for s in setups:
        e = [p for p in nominal if p["event"] and p["setup"] == s]
        st = [p for p in e if p["stuck"]]
        rows = [p["row"] for p in st]
        back = np.mean([abs(float(r["end_angle"])) > np.pi / 2 for r in rows]) if rows else np.nan
        off = np.mean([int(r["end_wheels_off"]) >= 4 for r in rows]) if rows else np.nan
        still = np.mean([abs(float(r["end_speed"])) < 1 for r in rows]) if rows else np.nan
        print(f"  {s:12s} stuck {len(st)}/{len(e)} = {len(st) / max(len(e), 1):.2f};  of the stuck: facing backwards"
              f" {back:.2f}, all wheels off the road {off:.2f}, stopped (< 1 m/s) {still:.2f}")
    for f, lab in (("runoff_out", "narrowest runoff outside (m)"), ("loose_out", "gravel/sand outside (share)")):
        vals = np.array([c[f] for c in corners])
        med = np.median(vals)
        for name, test in ((f"{lab} <= {med:.2g}", lambda c: c[f] <= med), (f"{lab} > {med:.2g}", lambda c: c[f] > med)):
            e = [p for p in nominal if p["event"] and test(corners[p["corner"]])]
            print(f"  {name:40s} stuck {sum(p['stuck'] for p in e) / max(len(e), 1):.2f} of {len(e)} losses")

    # arriving too fast? the car's speed when it lost control against the bot's there
    print("\nSpeed when control was lost, against the bot's speed at that spot (nominal starts):")
    tabs = {t: Track.load(t) for t in {c["track"] for c in corners}}
    for flag, name in ((1.0, "braking zones"), (0.0, "flat out")):
        for s in setups:
            e = [p for p in nominal if p["event"] and p["setup"] == s and corners[p["corner"]]["braking"] == flag]
            ratio = []
            for p in e:
                t = tabs[corners[p["corner"]]["track"]]
                cell = int(round(p["lost_s"] / t.spacing)) % t.cells
                ratio.append(p["lost_speed"] / max(t.table[cell, -3], 1.0))
            if ratio:
                q = np.percentile(ratio, [25, 50, 75])
                print(f"  {name:14s} {s:12s} car / bot speed: median {q[1]:.2f} (quartiles {q[0]:.2f}-{q[2]:.2f}),"
                      f" faster than the bot in {np.mean(np.array(ratio) > 1):.2f} of {len(ratio)}")

    # the per-corner table
    out = OUT / "corners-report.csv"
    with open(out, "w", newline="") as f:
        cols = list(corners[0]) + [f"{x}_{s}" for s in setups for x in ("passages", "losses", "stuck")] + ["fails"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, c in enumerate(corners):
            row = dict(c, fails=fails[i])
            for s in setups:
                ps = [p for p in nominal if p["corner"] == i and p["setup"] == s]
                row |= {f"passages_{s}": len(ps), f"losses_{s}": sum(p["event"] for p in ps),
                        f"stuck_{s}": sum(p["stuck"] for p in ps)}
            w.writerow(row)
    print(f"\nper-corner table: {out}")


if __name__ == "__main__":
    main()
