"""Evaluation figures (static PNGs logged as MLflow artifacts).

Colors follow the dataviz reference palette (light surface): speed is a sequential one-hue blue ramp;
agent vs bot use categorical slots 1 (blue) and 2 (orange), which validate as a pair in both modes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

from .reference import Reference  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BORDER = "#b5b4ae"
ROAD = "#e6e5e0"
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


def track_map(ref: Reference, trace: dict[str, np.ndarray], title: str, marks=()):
    """Track borders with the car's line colored by speed. marks: (x, y, label) spots to point out."""
    allb = np.vstack([ref.track_left, ref.track_right])
    span = allb.max(axis=0) - allb.min(axis=0)
    fig, ax = plt.subplots(figsize=(9, min(9, max(4, 8 * span[1] / span[0]) + 0.8)), dpi=130)
    for border in (ref.track_left, ref.track_right):
        closed = np.vstack([border, border[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=BORDER, lw=1.0, zorder=1)
    pts = np.column_stack([trace["x"], trace["y"]])
    kmh = np.maximum(trace["speed_x"], 0) * 3.6  # a spinning car has a negative one, which would stretch the scale
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
    for x, y, label in marks:
        ax.plot(x, y, "X", ms=11, color=TEXT, mec=SURFACE, mew=1.5, zorder=4)
        ax.annotate(label, (x, y), xytext=(9, 7), textcoords="offset points", color=TEXT, fontsize=10, fontweight="bold",
                    zorder=5, path_effects=[withStroke(linewidth=3, foreground=SURFACE)])
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


def crash_gif(ref: Reference, trace: dict[str, np.ndarray], crash_i: int, out, title: str, before: float = 5.0,
              after: float = 3.0, fps: int = 12) -> None:
    """A zoomed top-down clip of a crash at real time: the car, and the bot from the same point at the same moment,
    with speeds and the agent's inputs. It ends with the episode, at most `after` seconds past crash_i."""
    from PIL import Image

    t = trace["time"]
    t0, t1 = t[crash_i] - before, min(t[crash_i] + after, t[-1])
    first = int(np.searchsorted(t, t0))
    pts = np.column_stack([trace["x"], trace["y"]])
    bot = ref.trace
    bot_raced = np.maximum.accumulate(bot["dist_raced"])
    b0 = min(int(np.searchsorted(bot_raced, trace["dist_raced"][first])), len(bot_raced) - 1)
    bot_t = bot["time"] - bot["time"][b0] + t[first]
    keep = (bot_t >= t[first]) & (bot_t <= t1)
    bot_pts = np.column_stack([bot["x"][keep], bot["y"][keep]])
    bot_when, bot_speed = bot_t[keep], bot["speed_x"][keep]

    window = np.vstack([pts[first:np.searchsorted(t, t1) + 1], bot_pts])
    lo, hi = window.min(axis=0) - 25, window.max(axis=0) + 25
    aspect = (0.96 * 6.6) / (0.72 * 5.4)  # of the map's axes: grow the view to fill them
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    half = np.maximum(half, [half[1] * aspect, half[0] / aspect])
    lo, hi = centre - half, centre + half
    fig = Figure(figsize=(6.6, 5.4), dpi=100, facecolor=SURFACE)
    canvas = FigureCanvasAgg(fig)
    fig.text(0.03, 0.945, title, fontweight="bold", fontsize=12, color=TEXT)
    ax = fig.add_axes([0.02, 0.2, 0.96, 0.72])
    ax.set_aspect("equal")
    ax.axis("off")
    left, right = ref.track_left, ref.track_right
    in_view = np.all((left >= lo - 50) & (left <= hi + 50), axis=1)  # drawing only this keeps a frame fast
    quads = [[left[i], left[i + 1], right[i + 1], right[i]] for i in np.flatnonzero(in_view[:-1] | in_view[1:])]
    ax.add_collection(PolyCollection(quads, facecolor=ROAD, edgecolor=ROAD, linewidth=0.5, zorder=1))
    for border in (left, right):
        ax.plot(*np.where(in_view[:, None], border, np.nan).T, color=BORDER, lw=1, zorder=2)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.plot(*pts[crash_i], "X", ms=11, color=TEXT, mec=SURFACE, mew=1.5, zorder=3)
    if len(bot_pts):
        ax.plot(bot_pts[:, 0], bot_pts[:, 1], color=BOT, lw=1.5, alpha=0.5, zorder=3)
    trail, = ax.plot([], [], color=AGENT, lw=2.5, solid_capstyle="round", zorder=4)
    bot_dot, = ax.plot([], [], "o", color=BOT, ms=9, mec=SURFACE, mew=1.5, zorder=5)
    dot, = ax.plot([], [], "o", color=AGENT, ms=10, mec=SURFACE, mew=1.5, zorder=6)

    panel = fig.add_axes([0.03, 0.02, 0.94, 0.17])
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.axis("off")
    for y, color, name in ((0.72, AGENT, "agent"), (0.28, BOT, "bot")):
        panel.plot(0.012, y, "o", color=color, ms=9)
        panel.text(0.035, y, name, va="center", color=TEXT, fontsize=11)
    speed = panel.text(0.255, 0.72, "", va="center", ha="right", color=TEXT, fontsize=10, family="DejaVu Sans Mono")
    bot_text = panel.text(0.255, 0.28, "", va="center", ha="right", color=TEXT, fontsize=10, family="DejaVu Sans Mono")
    clock = fig.text(0.97, 0.945, "", ha="right", color=TEXT_2, fontsize=11, family="DejaVu Sans Mono")
    bars = {}
    for y, label, low, high in ((0.72, "steer", "L", "R"), (0.28, "pedal", "brake", "throttle")):
        panel.text(0.285, y, label, va="center", color=TEXT_2, fontsize=10)
        panel.add_patch(Rectangle((0.42, y - 0.12), 0.48, 0.24, facecolor=ROAD, edgecolor="none"))
        panel.plot([0.66, 0.66], [y - 0.17, y + 0.17], color=TEXT_2, lw=1)
        panel.text(0.415, y, low, va="center", ha="right", color=TEXT_2, fontsize=8)
        panel.text(0.905, y, high, va="center", ha="left", color=TEXT_2, fontsize=8)
        bars[label] = panel.add_patch(Rectangle((0.66, y - 0.12), 0, 0.24, facecolor=AGENT, edgecolor="none"))

    frames = []
    for now in np.arange(t[first], t1 + 1e-6, 1 / fps):
        i = min(int(np.searchsorted(t, now)), len(t) - 1)
        trail.set_data(pts[first:i + 1, 0], pts[first:i + 1, 1])
        dot.set_data([pts[i, 0]], [pts[i, 1]])
        speed.set_text(f"{trace['speed_x'][i] * 3.6:4.0f} km/h")
        if len(bot_when):
            bot_dot.set_data([np.interp(now, bot_when, bot_pts[:, 0])], [np.interp(now, bot_when, bot_pts[:, 1])])
            bot_text.set_text(f"{np.interp(now, bot_when, bot_speed) * 3.6:4.0f} km/h")
        clock.set_text(f"{now - t[crash_i]:+.1f} s")
        # The engine steers left with positive values: on screen, right is right.
        bars["steer"].set_width(-float(trace["steer"][i]) * 0.24)
        bars["pedal"].set_width(float(trace["pedal"][i]) * 0.24)
        canvas.draw()
        rgb = Image.frombuffer("RGBA", canvas.get_width_height(), canvas.buffer_rgba()).convert("RGB")
        frames.append(rgb.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    durations = [round(1000 / fps)] * (len(frames) - 1) + [1500]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
