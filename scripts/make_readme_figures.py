"""Build the README's result tables and chart.

    python scripts/make_readme_figures.py            # plot from the committed CSVs in docs/data/
    python scripts/make_readme_figures.py --export   # first re-export them from mlflow.db and mlruns/ (local only)

--export opens mlflow.db read-only. "Last 10 evals" is used everywhere on purpose: a single eval is noisy
(see docs/gym-wrapper.md, "Eval jitter") and the best/ checkpoint was selected using the held-out tracks.
"""

import argparse
import csv
import shutil
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, IMG = ROOT / "docs/data", ROOT / "docs/img"
RUNS = {"proto-3": "1c4e0e2cb93540cd9db3946b9c659579", "proto-4": "c9058ddde8de4c7f8347bb443a599f85",
        "proto-5": "88cf392dbeca4059ac4eb4a9d3aea156", "proto-6": "116a8a8675604dd28a29046d12508db4"}
CHART_RUN, IMAGE_STEP = "proto-6", "0256000000"
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

    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "results_by_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "run_id", "git_sha", "git_dirty", "seed", "train_tracks", "eval_train_tracks", "heldout_tracks",
                    "eval_every_steps", "train_completion", "train_flying_gap_pct",
                    "heldout_completion", "heldout_flying_gap_pct", "heldout_share_tracks_with_flying_lap",
                    "jarama_flying_median_s", "jarama_steer_delta_final"])
        for run, rid in RUNS.items():
            params = dict(db.execute("select key, value from params where run_uuid=?", (rid,)))
            tags = dict(db.execute("select key, value from tags where run_uuid=?", (rid,)))
            evals = series(rid, "eval_train/completion_rate")
            round4 = lambda key: None if (m := last_mean(rid, key)[0]) is None else round(m, 4)
            w.writerow([run, rid, tags["rlpilot.git_sha"][:7], tags["rlpilot.git_dirty"], params["seed"],
                        params["env.races"].count("rlpilot"), len(params["eval.train_tracks"].split(",")),
                        len(params["eval.heldout_tracks"].split(",")), evals[1][0] - evals[0][0],
                        round4("eval_train/completion_rate"), round4("eval_train/flying_gap_pct"),
                        round4("eval_heldout/completion_rate"), round4("eval_heldout/flying_gap_pct"),
                        round4("eval_heldout/tracks_with_flying_lap"), round4("eval/flying_lap_time_median"),
                        round(series(rid, "eval/behavior/steer_delta")[-1][1], 4)])

    rid = RUNS[CHART_RUN]
    params = dict(db.execute("select key, value from params where run_uuid=?", (rid,)))
    with open(DATA / "results_by_track.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "track", "split", "completion", "flying_gap_pct", "evals_with_flying_lap", "evals"])
        for split, names in (("train", params["eval.train_tracks"]), ("heldout", params["eval.heldout_tracks"])):
            for t in names.split(","):
                gap, n_gap = last_mean(rid, f"eval_track/{t}/flying_gap_pct")
                w.writerow([CHART_RUN, t, split, round(last_mean(rid, f"eval_track/{t}/completion_rate")[0], 4),
                            None if gap is None else round(gap, 4), n_gap, WINDOW])

    IMG.mkdir(parents=True, exist_ok=True)
    source = ROOT / "mlruns/1" / rid / "artifacts/eval" / IMAGE_STEP
    export_input_jitter(ROOT / "replay/jarama.controls")  # from scripts/record_lap.py
    shutil.copyfile(source / "speed_vs_distance.png", IMG / "speed_vs_distance.png")


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


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    colors = {"train": "#2a78d6", "heldout": "#eb6834"}
    with open(DATA / "results_by_track.csv") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["split"] != "train", -float(r["completion"])))
    y = list(range(len(rows)))[::-1]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": ink})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, facecolor=surface,
                                      gridspec_kw={"wspace": 0.06, "top": 0.84})
    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.grid(axis="x", color=grid, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)

    left.barh(y, [float(r["completion"]) for r in rows], height=0.55, color=[colors[r["split"]] for r in rows])
    for yi, r in zip(y, rows):
        if float(r["completion"]) < 0.005:
            left.text(0.012, yi, "0 laps completed", va="center", color=muted, fontsize=9)
    left.set_yticks(y, [r["track"] for r in rows])
    left.set_xlim(0, 1)
    left.set_xlabel("share of eval episodes finishing at least 1 lap", loc="left")
    left.set_title("Reliability", loc="left", fontweight="bold")

    right.axvline(0, color=muted, lw=1)
    for yi, r in zip(y, rows):
        n = int(r["evals_with_flying_lap"])
        if n == 0:
            right.text(1.55, yi, "no flying lap", va="center", color=muted, fontsize=9)
            continue
        gap = float(r["flying_gap_pct"])
        right.scatter(gap, yi, s=70, color=colors[r["split"]], edgecolor=surface, linewidth=1.5, zorder=3)
        right.text(1.55, yi, f"{gap:+.2f}%   {n}/{r['evals']} evals", va="center", color=muted, fontsize=9)
    right.set_xlim(-3, 4.2)
    right.set_xticks([-3, -2, -1, 0, 1])
    right.set_xlabel("median flying lap vs bot's best, % (negative = faster)", loc="left")
    right.set_title("Pace, when a flying lap exists", loc="left", fontweight="bold")

    fig.legend(handles=[Patch(color=colors["train"], label="trained on (4 of the 35 are evaluated)"),
                        Patch(color=colors["heldout"], label="held out (never trained on)")],
               loc="upper left", bbox_to_anchor=(0.115, 0.985), ncol=2, frameon=False, handlelength=1.2)
    fig.suptitle(f"{CHART_RUN}, mean of the last {WINDOW} evals (228M-300M steps), one seed", x=0.122, ha="left", y=1.04,
                 fontweight="bold", fontsize=12)
    fig.savefig(IMG / "results_by_track.png", dpi=160, bbox_inches="tight", facecolor=surface)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true", help="re-export docs/data and docs/img from the local MLflow store")
    a = p.parse_args()
    if a.export:
        export()
    plot()
