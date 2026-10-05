"""Evaluation figures (static PNGs logged as MLflow artifacts).

Colors follow the dataviz reference palette (light surface): speed is a sequential one-hue blue ramp;
agent vs bot use categorical slots 1 (blue) and 2 (orange), which validate as a pair in both modes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .reference import Reference  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BORDER = "#b5b4ae"
AGENT = "#2a78d6"  # categorical slot 1
BOT = "#eb6834"    # categorical slot 2
SPEED_CMAP = LinearSegmentedColormap.from_list(  # sequential blue, steps 200 -> 700
    "speed", ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def track_map(ref: Reference, trace: dict[str, np.ndarray], title: str):
    """Track borders with the car's line colored by speed."""
    allb = np.vstack([ref.track_left, ref.track_right])
    span = allb.max(axis=0) - allb.min(axis=0)
    fig, ax = plt.subplots(figsize=(9, min(9, max(4, 8 * span[1] / span[0]) + 0.8)), dpi=130)
    for border in (ref.track_left, ref.track_right):
        closed = np.vstack([border, border[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=BORDER, lw=1.0, zorder=1)
    pts = np.column_stack([trace["x"], trace["y"]])
    kmh = trace["speed_x"] * 3.6
    if len(pts) > 1:
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=SPEED_CMAP, linewidth=2.5, capstyle="round", zorder=2)
        lc.set_array(0.5 * (kmh[:-1] + kmh[1:]))
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("speed (km/h)", color=TEXT_2)
        cb.outline.set_visible(False)
        ax.plot(*pts[0], "o", ms=8, color=SURFACE, mec=TEXT_2, mew=1.5, zorder=3)
        ax.annotate("start", pts[0], xytext=(6, 6), textcoords="offset points", color=TEXT_2, fontsize=9)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _time_at_distance(trace: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    d = np.maximum.accumulate(trace["dist_from_start"])
    d, i = np.unique(d, return_index=True)
    return d, trace["time"][i] - trace["time"][0]


def speed_vs_distance(agent: dict[str, np.ndarray], bot: dict[str, np.ndarray], title: str):
    """Speed along the lap for agent and bot, and the cumulative time gap (agent - bot) below it."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), dpi=130, sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ends = []
    for tr, color, label in ((bot, BOT, "built-in bot"), (agent, AGENT, "agent")):
        if len(tr["time"]):
            ax1.plot(tr["dist_from_start"], tr["speed_x"] * 3.6, color=color, lw=2, label=label)
            ends.append([tr["speed_x"][-1] * 3.6, tr["dist_from_start"][-1], label])
    if len(ends) == 2 and abs(ends[0][0] - ends[1][0]) < 12:  # keep end labels apart
        lo, hi = sorted(ends, key=lambda e: e[0])
        lo[0], hi[0] = (lo[0] + hi[0]) / 2 - 6, (lo[0] + hi[0]) / 2 + 6
    for v, d, label in ends:
        ax1.annotate(label, (d, v), xytext=(4, 0), textcoords="offset points", color=TEXT_2, fontsize=9,
                     va="center")
    ax1.set_ylabel("speed (km/h)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, borderaxespad=0.2)
    ax1.set_title(title, pad=26)

    if len(agent["time"]) > 1 and len(bot["time"]) > 1:
        da, ta = _time_at_distance(agent)
        db, tb = _time_at_distance(bot)
        d = da[(da >= db[0]) & (da <= db[-1])]
        gap = np.interp(d, da, ta) - np.interp(d, db, tb)
        ax2.fill_between(d, 0, gap, color=AGENT, alpha=0.18, lw=0)
        ax2.plot(d, gap, color=AGENT, lw=2)
        ax2.axhline(0, color=TEXT_2, lw=0.8)
        if len(gap):
            ax2.annotate(f"{gap[-1]:+.1f} s", (d[-1], gap[-1]), xytext=(4, 0), textcoords="offset points",
                         color=TEXT, fontsize=9, va="center")
    ax2.set_ylabel("time lost to bot (s)")
    ax2.set_xlabel("distance from start line (m)")
    fig.tight_layout()
    return fig
