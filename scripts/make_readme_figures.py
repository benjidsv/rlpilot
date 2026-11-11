"""Build the README's result tables and charts.

    python scripts/make_readme_figures.py            # plot from the committed CSVs in docs/data/
    python scripts/make_readme_figures.py --export   # first re-export them (local only: mlflow.db, mlruns/, reports/)

--export opens mlflow.db read-only. Every number in the README comes from a CSV written here, and every CSV row names
the MLflow run it comes from. Sources:

- proto-3..6 (live evals, `results_by_run.csv`): mean of the last 10 evals, 4 monitored training tracks and 6 held-out
  tracks, "completion" = at least 1 lap. Not comparable with the rows below.
- proto-7 and later (`report_by_run.csv`, `best_by_track.csv`, `checkpoint_flips.csv`): scripts/report.py's sweeps in
  reports/<run>-<id>/data/, all 29 training + 6 validation tracks, "finished" = drove both laps of a 2-lap episode.
  best/ uses 8 episodes per track, the checkpoint sweeps 2. "strict" applies the 1 s off-track and backwards limits of
  the runs up to proto-8 to every run, so proto-9's looser limits don't inflate its numbers.
- The test split (`best_by_track.csv`, split "test"): scripts/eval_test_tracks.py, saved in the run's test/episodes.json.
- PPO health (`ppo_health.csv`): training metrics per block of steps, block = [start, start + size).
"""

import argparse
import csv
import json
import shutil
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, IMG = ROOT / "docs/data", ROOT / "docs/img"
sys.path.insert(0, str(ROOT / "python"))
RUNS = {"proto-3": "1c4e0e2cb93540cd9db3946b9c659579", "proto-4": "c9058ddde8de4c7f8347bb443a599f85",
        "proto-5": "88cf392dbeca4059ac4eb4a9d3aea156", "proto-6": "116a8a8675604dd28a29046d12508db4"}
# proto-7 and later, with their report folders. proto-8 crashed at 130.9M and proto-9 at 104.9M: the -cont runs resumed
# them with the same step count and schedule, and their reports cover the -cont part only.
REPORT_RUNS = {"proto-7": "0d567d903d9e44509034d0c9b87d100a", "proto-8-cont": "b60db21f2534453685ee010e315a5584",
               "proto-9-cont": "0cfb244eb5bf4dbf91ad28dc89aab0f2", "proto-9.5": "cc2e6507a9da4e28802dfb3d1974a052"}
HEADLINE = "proto-8-cont"
HEALTH_RUNS = {"pilot-1": "3615cdf115364b888fc7aec617bf3e44", "proto-7": REPORT_RUNS["proto-7"],
               "proto-8-cont": REPORT_RUNS["proto-8-cont"], "proto-9-cont": REPORT_RUNS["proto-9-cont"],
               "proto-9.5": REPORT_RUNS["proto-9.5"], "proto-9.5-sde2": "3855ddfe9f47410682611114cb402124"}
LATE = (250e6, 300e6)  # the step window compared across proto-7..9.5: all of them reached 300M
IMAGE_RUN, IMAGE_STEP = "proto-6", "0256000000"
WINDOW = 10
ROAD_FACTOR = 4.0  # lateral exaggeration of the replay map (less in the tightest corners, see WideRoad)
LINE_SMOOTH_M = 15.0  # smoothing of the lateral offsets in the replay map


def export() -> None:
    db = sqlite3.connect(f"file:{ROOT / 'mlflow.db'}?mode=ro&immutable=1", uri=True)

    def series(run_id: str, key: str) -> list[tuple[int, float]]:
        return sorted(db.execute("select step, value from metrics where run_uuid=? and key=?", (run_id, key)))

    def last_mean(run_id: str, key: str) -> tuple[float | None, int]:
        """Mean over the last WINDOW evals of the run, and how many of them logged the metric:
        a gap only exists when a flying lap was completed."""
        steps = {s for s, _ in series(run_id, "eval_train/completion_rate")[-WINDOW:]}
        values = [v for s, v in series(run_id, key) if s in steps]
        return (statistics.mean(values), len(values)) if values else (None, 0)

    def tags(run_id: str) -> dict[str, str]:
        return dict(db.execute("select key, value from tags where run_uuid=?", (run_id,)))

    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "results_by_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "run_id", "git_sha", "git_dirty", "seed", "train_tracks", "eval_train_tracks", "heldout_tracks",
                    "eval_every_steps", "train_completion", "train_flying_gap_pct",
                    "heldout_completion", "heldout_flying_gap_pct", "heldout_share_tracks_with_flying_lap",
                    "jarama_flying_median_s", "jarama_steer_delta_final"])
        for run, rid in RUNS.items():
            params = dict(db.execute("select key, value from params where run_uuid=?", (rid,)))
            t = tags(rid)
            evals = series(rid, "eval_train/completion_rate")
            round4 = lambda key: None if (m := last_mean(rid, key)[0]) is None else round(m, 4)
            w.writerow([run, rid, t["rlpilot.git_sha"][:7], t["rlpilot.git_dirty"], params["seed"],
                        params["env.races"].count("rlpilot"), len(params["eval.train_tracks"].split(",")),
                        len(params["eval.heldout_tracks"].split(",")), evals[1][0] - evals[0][0],
                        round4("eval_train/completion_rate"), round4("eval_train/flying_gap_pct"),
                        round4("eval_heldout/completion_rate"), round4("eval_heldout/flying_gap_pct"),
                        round4("eval_heldout/tracks_with_flying_lap"), round4("eval/flying_lap_time_median"),
                        round(series(rid, "eval/behavior/steer_delta")[-1][1], 4)])

    export_reports(db, tags)
    export_ppo_health(db)

    IMG.mkdir(parents=True, exist_ok=True)
    source = ROOT / "mlruns/1" / RUNS[IMAGE_RUN] / "artifacts/eval" / IMAGE_STEP
    export_input_jitter(ROOT / "replay/jarama-proto6/jarama.controls")  # proto-6's recording, from scripts/record_lap.py
    shutil.copyfile(source / "speed_vs_distance.png", IMG / "speed_vs_distance.png")


def _race(track: str) -> str:
    return "rlpilot" if track == "jarama" else f"rlpilot-{track}"


def export_reports(db, tags) -> None:
    import numpy as np
    from rlpilot import reference
    from rlpilot.splits import PROTO7_TEST, PROTO7_TRAIN, PROTO7_VALIDATION
    from rlpilot.track import reference_for

    splits = {"train": PROTO7_TRAIN, "validation": PROTO7_VALIDATION}
    bot = {t: reference.load(reference_for(_race(t)), laps=3).best_lap for t in PROTO7_TRAIN + PROTO7_VALIDATION + PROTO7_TEST}
    finished = lambda e: e["end"] == "lap_done"
    strict = lambda e: e.get("finished_strict", finished(e))  # reports before proto-9 only had the strict rules

    def lap2_gaps(folder: Path, track: str, episodes: list[dict]) -> list[float]:
        """Lap 2 of each episode that drove both laps, against the bot's best lap on that track, in %."""
        z = np.load(folder / f"data/best_{_race(track)}.npz")
        return [100 * (float(z[f"ep{i}_last_lap_time"][-1]) / bot[track] - 1) for i, e in enumerate(episodes) if finished(e)]

    by_run, by_track, flips = [], [], []
    for run, rid in REPORT_RUNS.items():
        folder = ROOT / "reports" / f"{run}-{rid[:8]}"
        res = json.loads((folder / "data/results.json").read_text())
        t = tags(rid)
        steps = sorted(int(s) for s in res["sweeps"])
        late = [s for s in steps if LATE[0] <= s <= LATE[1]]
        for split, tracks in splits.items():
            best = [e for tr in tracks for e in res["best"][_race(tr)]]
            gaps = [statistics.median(g) for tr in tracks if (g := lap2_gaps(folder, tr, res["best"][_race(tr)]))]
            share = lambda s, rule: statistics.mean([rule(e) for tr in tracks for e in res["sweeps"][str(s)][_race(tr)]])
            by_run.append({"run": run, "run_id": rid, "git_sha": t["rlpilot.git_sha"][:7], "git_dirty": t["rlpilot.git_dirty"],
                           "best_step": t["best_step"], "split": split, "tracks": len(tracks),
                           "best_episodes": len(best), "best_finished": sum(map(finished, best)),
                           "best_finished_strict": sum(map(strict, best)),
                           "best_finished_share": round(statistics.mean(map(finished, best)), 4),
                           "best_finished_strict_share": round(statistics.mean(map(strict, best)), 4),
                           "best_tracks_with_lap2": len(gaps),
                           "best_lap2_gap_pct_median_of_tracks": round(statistics.median(gaps), 2) if gaps else None,
                           "late_window": f"{late[0] / 1e6:.0f}-{late[-1] / 1e6:.0f}M", "late_checkpoints": len(late),
                           "late_episodes_per_track": len(res["sweeps"][str(late[0])][_race(tracks[0])]),
                           "late_finished_share": round(statistics.mean(share(s, finished) for s in late), 4),
                           "late_finished_strict_share": round(statistics.mean(share(s, strict) for s in late), 4)})
            if run == HEADLINE:
                for tr in tracks:
                    eps = res["best"][_race(tr)]
                    g = lap2_gaps(folder, tr, eps)
                    by_track.append({"run": run, "run_id": rid, "track": tr, "split": split, "episodes": len(eps),
                                     "finished": sum(map(finished, eps)), "bot_best_lap_s": round(bot[tr], 3),
                                     "lap2_gap_pct_median": round(statistics.median(g), 2) if g else None})
        names = PROTO7_TRAIN + PROTO7_VALIDATION
        for a, b in zip(steps, steps[1:]):
            changed = [tr for tr in names if sum(map(strict, res["sweeps"][str(a)][_race(tr)]))
                       != sum(map(strict, res["sweeps"][str(b)][_race(tr)]))]
            flips.append({"run": run, "run_id": rid, "step_a": a, "step_b": b, "tracks": len(names),
                          "tracks_changed": len(changed), "changed": " ".join(changed)})

    rid = REPORT_RUNS[HEADLINE]
    test = json.loads((ROOT / "mlruns/1" / rid / "artifacts/test/episodes.json").read_text())
    assert str(test["best_step"]) == tags(rid)["best_step"], "the test split must come from the same best/"
    for tr in PROTO7_TEST:
        eps = [e for e in test["episodes"] if e["track"] == tr]
        g = [100 * (e["lap_times"][1] / bot[tr] - 1) for e in eps if e["end"] == "lap_done"]
        by_track.append({"run": HEADLINE, "run_id": rid, "track": tr, "split": "test", "episodes": len(eps),
                         "finished": sum(e["end"] == "lap_done" for e in eps), "bot_best_lap_s": round(bot[tr], 3),
                         "lap2_gap_pct_median": round(statistics.median(g), 2) if g else None})

    for name, rows in (("report_by_run.csv", by_run), ("best_by_track.csv", by_track), ("checkpoint_flips.csv", flips)):
        with open(DATA / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def export_ppo_health(db) -> None:
    """Means per 10M-step block. pilot-1 logged one std over all actions and its own eval (Jarama only, 4 episodes)."""
    keys = {"pilot-1": {"std": "sb3/train/std", "approx_kl": "ppo/approx_kl", "clip_fraction": "ppo/clip_fraction",
                        "eval_completion": "eval/completion_rate"}}
    later = {"std_steer": "ppo/std_steer", "std_pedal": "ppo/std_pedal", "approx_kl": "dashboard/ppo_approx_kl",
             "clip_fraction": "ppo/clip_fraction", "mean_speed_mps": "driving/mean_speed"}
    with open(DATA / "ppo_health.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "run_id", "metric", "mlflow_key", "block_M", "block_start_M", "mean", "n"])
        for run, rid in HEALTH_RUNS.items():
            for metric, key in keys.get(run, later).items():
                values = db.execute("select step, value from metrics where run_uuid=? and key=?", (rid, key)).fetchall()
                for size in (10, 1) if run == "pilot-1" else (10,):  # pilot-1 also at 1M, for its chart
                    blocks: dict[int, list[float]] = {}
                    for step, value in values:
                        blocks.setdefault(step // (size * 1_000_000) * size, []).append(value)
                    for start, vs in sorted(blocks.items()):
                        w.writerow([run, rid, metric, key, size, start, round(statistics.mean(vs), 4), len(vs)])


def export_input_jitter(controls: Path) -> None:
    """Flying-lap input activity of the replayed agent vs the bot, both at the agent's 25 Hz control rate."""
    import numpy as np
    from rlpilot import reference

    bot = reference.load("rlref").lap_trace(2)
    agent = np.fromfile(controls, dtype=[(k, "f4") for k in ("steer", "accel", "brake", "clutch")]
                        + [("gear", "i4"), ("x", "f4"), ("y", "f4")])
    start = np.array([bot["x"][0], bot["y"][0]])
    halfway = len(agent) // 3  # past the standing start, which also passes near the line
    agent = agent[halfway + np.argmin((agent["x"][halfway:] - start[0]) ** 2 + (agent["y"][halfway:] - start[1]) ** 2):]

    with open(DATA / "input_jitter.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["driver", "steer_change_per_step", "steer_centre_crossings_per_s", "brake_applications_per_s"])
        for name, steer, brake in (("agent", agent["steer"][::2], agent["brake"][::2]),
                                   ("bot", bot["steer"][::2], bot["brake"][::2])):
            seconds = len(steer) / 25
            w.writerow([name, round(float(np.abs(np.diff(steer)).mean()), 3),
                        round(int(np.sum(np.sign(steer[1:]) * np.sign(steer[:-1]) < 0)) / seconds, 2),
                        round(int(np.sum((brake[1:] > 0.05) & (brake[:-1] <= 0.05))) / seconds, 2)])


def _gauss(a, sigma: float, wrap: bool):
    import numpy as np

    r = int(4 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    pad = (a[-r:], a[:r]) if wrap else (np.full(r, a[0]), np.full(r, a[-1]))
    return np.convolve(np.concatenate([pad[0], a, pad[1]]), w / w.sum(), "valid")


class WideRoad:
    """Redraws positions with their lateral offset from the track centreline scaled up, keeping the length and shape
    of the track, so racing lines become readable. Everything lives on a 1 m grid along a smoothed centreline.

    Scaling alone would fold the road over itself where the offset exceeds a corner's radius, so the factor eases
    down in those corners only. Offsets are also smoothed along the track: scaling magnifies small wiggles too."""

    def __init__(self, left, right, factor: float, offset_smooth_m: float, fit: float = 0.9):
        import numpy as np

        raw = (left + right) / 2
        closed = np.vstack([raw, raw[:1]])
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
        grid = np.arange(0, s[-1])
        self.c = np.column_stack([_gauss(np.interp(grid, s, closed[:, i]), 6, True) for i in (0, 1)])
        t = np.roll(self.c, -1, axis=0) - np.roll(self.c, 1, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        self.n = np.column_stack([-t[:, 1], t[:, 0]])
        self.smooth = offset_smooth_m

        curvature = _gauss(np.gradient(np.unwrap(np.arctan2(t[:, 1], t[:, 0]))), 3, True)
        half_width = np.linalg.norm(left - right, axis=1).max() / 2
        limit = fit / np.maximum(np.abs(curvature) * half_width, 1e-6)  # factor that keeps the inner edge inside the turn
        tight = np.min(np.stack([np.roll(limit, i) for i in range(-25, 26, 5)]), axis=0)  # ease in before the corner
        self.factor = _gauss(np.minimum(factor, tight), 10, True)

        self.left, self.right = self.road_edge(left), self.road_edge(right)

    def _project(self, points):
        """Grid index and signed offset of ordered points, tracking along the lap so that a point can't jump."""
        import numpy as np

        idx, i = np.zeros(len(points), int), int(np.argmin(np.sum((self.c - points[0]) ** 2, axis=1)))
        for j, p in enumerate(points):
            window = (i + np.arange(-5, 40)) % len(self.c)
            i = window[np.argmin(np.sum((self.c[window] - p) ** 2, axis=1))]
            idx[j] = i
        return idx, np.sum((points - self.c[idx]) * self.n[idx], axis=1)

    def road_edge(self, edge):
        import numpy as np

        idx, offset = self._project(edge)
        order = np.argsort(idx)
        on_grid = np.interp(np.arange(len(self.c)), idx[order], offset[order], period=len(self.c))
        return self.c + (self.factor * _gauss(on_grid, 8, True))[:, None] * self.n

    def line(self, points):
        """Ordered positions of a car: lateral offsets smoothed over offset_smooth_m of track, then scaled."""
        import numpy as np

        idx, offset = self._project(points)
        along = np.r_[0, np.cumsum((np.diff(idx) + len(self.c) // 2) % len(self.c) - len(self.c) // 2)].astype(float)
        along = np.maximum.accumulate(along)
        grid = np.arange(along[0], along[-1] + 1)
        smooth = _gauss(np.interp(grid, along, offset), self.smooth, False)
        return self.c[idx] + (self.factor[idx] * np.interp(along, grid, smooth))[:, None] * self.n[idx]

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SPLIT_COLORS = {"train": "#2a78d6", "validation": "#eb6834", "test": "#1baf7a"}  # reference palette slots 1-3


def _style(plt) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK})


def _clean(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_tracks(plt) -> None:
    from matplotlib.patches import Patch

    with open(DATA / "best_by_track.csv") as f:
        rows = list(csv.DictReader(f))
    with open(DATA / "report_by_run.csv") as f:
        step = int(next(r for r in csv.DictReader(f) if r["run"] == HEADLINE)["best_step"])
    order = list(SPLIT_COLORS)
    rows.sort(key=lambda r: (order.index(r["split"]), -int(r["finished"]) / int(r["episodes"]), r["track"]))
    # a blank row between splits
    y, labels, pos = [], [], 0
    for i, r in enumerate(rows):
        if i and r["split"] != rows[i - 1]["split"]:
            pos += 1
        y.append(pos)
        pos += 1
    y = [pos - 1 - v for v in y]

    _style(plt)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 10.5), sharey=True, facecolor=SURFACE,
                                      gridspec_kw={"wspace": 0.06, "top": 0.92, "width_ratios": (1, 1)})
    _clean(left)
    _clean(right)
    share = [int(r["finished"]) / int(r["episodes"]) for r in rows]
    left.barh(y, share, height=0.62, color=[SPLIT_COLORS[r["split"]] for r in rows])
    for yi, r, s in zip(y, rows, share):
        left.text(s + 0.015, yi, f"{r['finished']}/{r['episodes']}", va="center", color=MUTED, fontsize=8.5)
    left.set_yticks(y, [r["track"] for r in rows], fontsize=9)
    left.set_xlim(0, 1.12)
    left.set_xticks([0, 0.25, 0.5, 0.75, 1])
    left.set_xlabel("episodes that drove both laps", loc="left")
    left.set_title("Reliability", loc="left", fontweight="bold")

    right.axvline(0, color=MUTED, lw=1)
    for yi, r in zip(y, rows):
        if not r["lap2_gap_pct_median"]:
            right.text(-11.6, yi, "no lap 2", va="center", color=MUTED, fontsize=8.5)
            continue
        gap = float(r["lap2_gap_pct_median"])
        right.scatter(gap, yi, s=55, color=SPLIT_COLORS[r["split"]], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        right.text(4.2, yi, f"{gap:+.1f}%", va="center", ha="right", color=MUTED, fontsize=8.5)
    right.set_xlim(-12, 4.4)
    right.set_xticks([-10, -5, 0])
    right.set_xlabel("median lap 2 vs the bot's best lap, % (negative = faster)", loc="left")
    right.set_title("Pace, on episodes that finished", loc="left", fontweight="bold")

    fig.legend(handles=[Patch(color=SPLIT_COLORS["train"], label="train (29)"),
                        Patch(color=SPLIT_COLORS["validation"], label="validation (6, selected best/)"),
                        Patch(color=SPLIT_COLORS["test"], label="test (6, evaluated once)")],
               loc="upper left", bbox_to_anchor=(0.115, 0.975), ncol=3, frameon=False, handlelength=1.2)
    fig.suptitle(f"{HEADLINE} best/ ({step / 1e6:.0f}M steps), 8 episodes per track, stock car, one seed",
                 x=0.122, ha="left", y=1.0, fontweight="bold", fontsize=12)
    fig.savefig(IMG / "best_by_track.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_health(plt) -> None:
    with open(DATA / "ppo_health.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["run"] == "pilot-1"]
    # drop the partial tail past 60M (one update, two evals); single evals per 1M are kept
    rows = [r for r in rows if int(r["block_start_M"]) < 60 or (r["metric"] == "eval_completion" and r["block_M"] == "1")]
    panels = (("eval_completion", "Eval completion (Jarama)"), ("std", "Action std (log scale)"),
              ("approx_kl", "Approx. KL"), ("clip_fraction", "Clip fraction"))
    color = SPLIT_COLORS["train"]
    _style(plt)
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), facecolor=SURFACE, gridspec_kw={"wspace": 0.32, "top": 0.78})
    for ax, (metric, title) in zip(axes, panels):
        _clean(ax, "y")
        pts = lambda size: sorted((int(r["block_start_M"]), float(r["mean"])) for r in rows
                                  if r["metric"] == metric and r["block_M"] == str(size))
        if metric == "eval_completion":  # 4 episodes on one track per eval: show each eval, and the mean per 10M
            evals = [(s, v) for s, v in pts(1)]
            ax.scatter([s + 0.5 for s, _ in evals], [v for _, v in evals], s=10, color=color, alpha=0.35, linewidth=0)
            for s, v in pts(10):
                ax.plot([s, s + 10], [v, v], color=color, lw=2, solid_capstyle="butt")
            ax.set_ylim(-0.03, 1.05)
        else:
            p = pts(1)
            ax.plot([s + 0.5 for s, _ in p], [v for _, v in p], color=color, lw=2)
        if metric == "std":
            ax.set_yscale("log")
            ax.set_yticks([0.01, 0.03, 0.1, 0.3], ["0.01", "0.03", "0.1", "0.3"])
            ax.minorticks_off()
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
        ax.set_xlim(0, 60)
        ax.set_xticks([0, 20, 40, 60], ["0", "20M", "40M", "60M"])
    fig.suptitle("pilot-1, PPO without an entropy bonus on Jarama only: means per 1M steps "
                 "(completion: one dot per eval, line = mean per 10M)", x=0.125, ha="left", y=0.98, fontweight="bold",
                 fontsize=11)
    fig.savefig(IMG / "ppo_health_pilot1.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_tracks(plt)
    plot_health(plt)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true", help="re-export docs/data and docs/img from the local MLflow store")
    a = p.parse_args()
    if a.export:
        export()
    plot()
