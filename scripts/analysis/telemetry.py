"""How a checkpoint drives, in numbers: 1 lap from the grid on every non-test track (as the in-run eval, capped at
evaluate.LAP_CAP), recorded at each policy step, summarized against the bot. One row per run, then the mean over seeds
per variant. Deterministic: the same checkpoint gives the same numbers.

    uv run python scripts/analysis/telemetry.py sw10m-baseline sw10m-colored          # every seed of each, last checkpoint
    uv run python scripts/analysis/telemetry.py runs/<run> --tracks dijon ardennen-spa  # one run, some tracks
    uv run python scripts/analysis/telemetry.py sw10m-baseline --csv runs/sweep/telemetry.csv
    uv run python scripts/analysis/telemetry.py sw10m-colored --shift hazard          # the gear choice read another way

Columns (pooled over the tracks; "moving" = above 3 m/s; the drive counts until it ended, crash or not):
- top km/h:     median over tracks of the car's top speed; top/bot: the same as a share of the bot's top speed there
- v/bot >50:    median of the car's speed / the bot's (its fastest lap, same place) where the bot is above 50 m/s
                (180 km/h: the straights); 30-50 and <30: the same in faster and slow corners
- s/km >50, s/km <50: seconds lost to the bot per km of road where the bot is above / below 50 m/s, after the first
                400 m (the start is its own column). From the speed profiles: 1/v_car - 1/v_bot per metre
- full thr:     share of the straights (bot above 33 m/s and accelerating, wheel nearly straight) at throttle > 0.95
- thr straight: mean throttle there
- limiter:      share of the straights on the throttle at the rev limiter (rpm > 97% of the redline): the car wants
                more speed but doesn't upshift (full thr counts these too when the throttle is > 0.95)
- brake>0.9:    share of moving time on the brakes at more than 0.9; decel p99: 99th percentile of the deceleration
                (m/s^2, from the speed on the sim clock); dec>8: share of moving time decelerating above 8 m/s^2.
                The bot's own laps (its trace, same tracks, noisier): p99 15 m/s^2, 9.8% of its time above 8
- start s:      seconds behind the bot's own standing start at 200 m (median over tracks)
- gears:        share of moving time in each gear (1..6)
- walls/10km:   wall contacts (a step with wall damage after one without) per 10 km driven
"""

import argparse
import csv
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))  # scripts/: eval.py
from eval import SHIFT_MODES, Driver, load_agent  # noqa: E402
from rlpilot import evaluate  # noqa: E402
from rlpilot.client import F  # noqa: E402
from rlpilot.tracks import C, Track, load_split  # noqa: E402

FAST, MID, STRAIGHT = 50.0, 30.0, 33.0  # m/s
LOG = ("s", "dist", "v", "t", "pedal", "steer", "gear", "wall", "len", "rpm")
LIMITER = 0.97  # rpm / redline: on the limiter (at full throttle above 150 km/h the cars sit at 0.999 median)


def last_checkpoint(run: Path) -> Path:
    numbered = sorted(c for c in (run / "checkpoints").glob("*.pt") if c.stem.isdigit())
    if not numbered:
        raise SystemExit(f"no checkpoint in {run / 'checkpoints'}")
    return numbered[-1]


def runs_of(arg: str) -> list[Path]:
    p = Path(arg)
    if p.is_dir():
        return [p]
    runs = sorted(Path(r) for r in glob.glob(f"runs/{arg}__*__*"))
    if not runs:
        raise SystemExit(f"{arg!r}: not a run dir, and no runs/{arg}__*")
    return runs


def drive(checkpoint: Path, tracks: list[str], shift: str = "argmax") -> dict[str, dict[str, np.ndarray]]:
    """1 capped lap per track; per track, the per-step log (LOG) until its drive ended. `shift`: eval.Driver's mode."""
    agent, normalize, _ = load_agent(checkpoint)
    driver = Driver(agent, shift)
    g = evaluate.GridLaps(tracks, laps=1)
    env = g.env
    log = {k: [[] for _ in tracks] for k in LOG}
    wall_prev = np.zeros(len(tracks), bool)

    def policy(o):
        a = driver(torch.as_tensor(normalize(o)), env.controls.gear)
        raw = env._raw
        wall = env.reward.wall_hit.copy()
        for i in range(len(tracks)):
            for k, v in (("s", raw[i, F["dist_from_start"]]), ("dist", env.reward.raced[i]),
                         ("v", raw[i, F["speed_x"]]), ("t", raw[i, F["time"]]), ("pedal", np.clip(a[i, 1], -1, 1)),
                         ("steer", env.controls.steer[i]), ("gear", env.controls.gear[i]),
                         ("wall", wall[i] and not wall_prev[i]), ("len", env._len[i]),
                         ("rpm", raw[i, F["rpm"]] / max(raw[i, F["rpm_redline"]], 1.0))):
                log[k][i].append(float(v))
        wall_prev[:] = wall
        return a

    try:
        g.run(policy)
    finally:
        g.close()
    out = {}
    for i, t in enumerate(tracks):
        ln = np.array(log["len"][i])
        drop = np.flatnonzero(np.diff(ln) < 0)  # the drive ended: the env restarted on the grid, then was paused
        keep = slice(0, drop[0] + 1) if len(drop) else slice(None)
        out[t] = {k: np.array(log[k][i])[keep] for k in LOG}
    return out


def summarize(drives: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    tops, top_ratio, starts = [], [], []
    ratio = {"fast": [], "mid": [], "slow": []}
    lost = {"fast": [0.0, 0.0], "slow": [0.0, 0.0]}  # seconds, km
    straight_pedal, straight_lim, moving_brake, decels, gears = [], [], [], [], []
    walls, km = 0, 0.0
    for t, d in drives.items():
        T = Track.load(t)
        grid_s = np.arange(T.table.shape[0]) * T.spacing
        bot = lambda s: np.interp(np.mod(s, T.length), grid_s, T.table[:, C["bot_speed"]])
        if len(d["v"]) < 10:
            continue
        v, s, dist = d["v"], d["s"], d["dist"]
        vb = bot(s)
        moving = v > 3
        tops.append(v.max() * 3.6)
        top_ratio.append(v.max() / T.table[:, C["bot_speed"]].max())
        for name, m in (("fast", vb > FAST), ("mid", (vb > MID) & (vb <= FAST)), ("slow", vb <= MID)):
            m = m & moving
            if m.any():
                ratio[name].append(v[m] / vb[m])
        # time lost per km, from the speed profiles along the drive, after the start
        far = (dist > 400) & moving
        if far.sum() > 10:
            ds = np.diff(dist, prepend=dist[0])
            loss = ds * (1 / np.maximum(v, 1) - 1 / np.maximum(vb, 1))
            for name, m in (("fast", far & (vb > FAST)), ("slow", far & (vb <= FAST))):
                lost[name][0] += loss[m].sum()
                lost[name][1] += ds[m].sum() / 1000
        accel_ahead = bot(s + 30) > vb + 0.5
        straight = moving & (vb > STRAIGHT) & accel_ahead & (np.abs(d["steer"]) < 0.05)
        straight_pedal.append(d["pedal"][straight])
        straight_lim.append((d["pedal"][straight] > 0) & (d["rpm"][straight] > LIMITER))
        moving_brake.append(d["pedal"][moving])
        dt = np.diff(d["t"])
        ok = dt > 1e-3
        decels.append(-(np.diff(v)[ok] / dt[ok]))
        gears.append(d["gear"][moving])
        walls += int(d["wall"].sum())
        km += max(dist[-1], 0) / 1000
        bd, bt = evaluate.bot_standing(t)
        if dist[-1] > 200:
            i = np.searchsorted(dist, 200)
            starts.append((d["t"][i] - d["t"][0]) - np.interp(200, bd, bt))
    cat = lambda x: np.concatenate(x) if x else np.zeros(0)
    sp, mp, g = cat(straight_pedal), cat(moving_brake), cat(gears)
    m = {"top km/h": np.median(tops), "top/bot": np.median(top_ratio),
         "v/bot >50": np.median(cat(ratio["fast"])), "v/bot 30-50": np.median(cat(ratio["mid"])),
         "v/bot <30": np.median(cat(ratio["slow"])),
         "s/km >50": lost["fast"][0] / max(lost["fast"][1], 1e-9), "s/km <50": lost["slow"][0] / max(lost["slow"][1], 1e-9),
         "full thr": np.mean(sp > 0.95) if len(sp) else np.nan, "thr straight": np.mean(sp) if len(sp) else np.nan,
         "limiter": np.mean(cat(straight_lim)) if len(sp) else np.nan,
         "brake>0.9": np.mean(mp < -0.9), "decel p99": np.percentile(cat(decels), 99),
         "dec>8": np.mean(cat(decels) > 8),
         "start s": np.median(starts), "walls/10km": 10 * walls / max(km, 1e-9)}
    m |= {f"g{k}": np.mean(g == k) for k in range(1, 7)}
    return m


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", help="experiment names (every seed) or run dirs")
    p.add_argument("--tracks", nargs="+", help="default: the 38 validation + train tracks")
    p.add_argument("--csv", type=Path, help="also write one row per run here")
    p.add_argument("--shift", choices=SHIFT_MODES, default="argmax", help="the gear choice (eval.Driver): argmax as the "
                   "in-run eval, sample as in training (seed 0), hazard deterministic")
    a = p.parse_args()
    tracks = a.tracks or load_split()["validation"] + load_split()["train"]
    rows = []
    for arg in a.runs:
        for run in runs_of(arg):
            ckpt = last_checkpoint(run)
            m = summarize(drive(ckpt, tracks, a.shift))
            rows.append({"variant": run.name.split("__")[0], "run": run.name, "step": int(ckpt.stem)} | m)
            print(f"  {run.name} ({ckpt.stem}) done", flush=True)
    keys = [k for k in rows[0] if k not in ("variant", "run", "step")]
    fmt = lambda k, x: f"{x:{max(len(k), 6)}.0f}" if k == "top km/h" else f"{x:{max(len(k), 6)}.3f}" \
        if not k.startswith("g") else f"{x:4.2f}"
    head = lambda k: f"{k:>{max(len(k), 6)}s}" if not k.startswith("g") else f"{k:>4s}"
    print(f"\n{'run':38s} " + " ".join(head(k) for k in keys))
    for r in rows:
        print(f"{r['run'][:38]:38s} " + " ".join(fmt(k, r[k]) for k in keys))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    if any(sum(r["variant"] == v for r in rows) > 1 for v in variants):
        print(f"\n{'mean over seeds':38s}")
        for v in variants:
            vr = [r for r in rows if r["variant"] == v]
            print(f"{(v + f' (n={len(vr)})')[:38]:38s} " + " ".join(fmt(k, np.mean([r[k] for r in vr])) for k in keys))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-run rows in {a.csv}")


if __name__ == "__main__":
    main()
