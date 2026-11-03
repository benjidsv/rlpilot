"""Render the road-edge observation as the policy sees it, because tests can share a wrong assumption.

Each panel is the real engine's state, read back from the observation vector through its layout (obs -> decode_edges and
the rays group), drawn in the car's frame over the true road from the dense map. The car faces up, its left is on the left.
Below it, the same edge points as the normalized values the network receives, against their point index.

    python scripts/plot_obs.py --out docs/img/proto-7-obs

A pure-pursuit driver on the bot's line and speed places the car. That is the only use of the bot here: it isn't part of the
observation. Also prints how well the points resolve Spa's Bus Stop.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlpilot import EnvConfig, SpeedDreamsVecEnv, reference
from rlpilot._native import F, RAY_RANGE
from rlpilot.features import ObsLayout, decode_edges
from rlpilot.track import RacingLine, TrackEdges, race_for, reference_for

POINTS, LIMIT, LOCK_RAD, WHEELBASE = 30, 0.0625, np.deg2rad(45), 2.6
RAY_DEG = np.array([-90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 75, 90], float)
LEFT, RIGHT, INK, MUTED, ROAD, RAY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3de", "#b5b4ad"
BUS_STOP = (6640.0, 6730.0)  # m on Spa: the chicane itself
HAIRPIN = 170.0  # m on Spa: La Source just ahead


@dataclass
class Frame:
    label: str
    race: str
    raw: np.ndarray
    obs: np.ndarray
    layout: ObsLayout
    zoom: tuple[float, float] | None = None  # lap distances to frame instead of the whole span


def drive(track: str, stops: list[tuple[str, float]], factor: float = 0.88, cap: float = 1e9, grid_label: str | None = None,
          zooms: dict[str, tuple[float, float]] | None = None) -> list[Frame]:
    """Drive one lap and keep a frame each time dist_raced passes a stop (m)."""
    race = race_for(track)
    edges, line = TrackEdges.for_race(race), RacingLine.for_race(race)
    bot = reference.load(reference_for(race), laps=3).lap_trace()
    bot_s, bot_v = bot["dist_from_start"], bot["speed_x"]
    order = np.argsort(bot_s)
    env = SpeedDreamsVecEnv(EnvConfig(n_workers=1, race=race, gearbox="points", lookahead=True, car_obs=True, edge_points=POINTS,
                                      steer_limit=LIMIT, steer_history=3, max_episode_seconds=780, spin_us=50))
    frames = []
    try:
        obs, _ = env.reset()
        if grid_label:
            frames.append(Frame(grid_label, race, env._raw[0].copy(), obs[0].copy(), env.obs_layout))
        todo = sorted(stops, key=lambda x: x[1])
        while todo:
            r = env._raw[0]
            v = float(max(r[F["speed_x"]], 0.0))
            s = float(r[F["dist_from_start"]])
            # aim at the bot's line a little ahead, in the car's frame
            preview = 6.0 + 0.45 * v
            i = int((s + preview) % edges.length / edges.spacing)
            row = edges.table[i]
            mid, across = 0.5 * (row[:2] + row[2:]), row[:2] - row[2:]
            aim = mid + across / np.hypot(*across) * line.table[int((s + preview) % edges.length) % len(line.table)]
            dx, dy = aim[0] - r[F["pos_x"]], aim[1] - r[F["pos_y"]]
            cy, sy = np.cos(r[F["yaw"]]), np.sin(r[F["yaw"]])
            ax, ay = cy * dx + sy * dy, cy * dy - sy * dx
            steer = np.arctan2(2 * WHEELBASE * ay, ax * ax + ay * ay) / LOCK_RAD
            target = min(float(np.interp(s, bot_s[order], bot_v[order], period=edges.length)) * factor, cap)
            pedal = float(np.clip(0.6 * (target - v), -1, 1))
            action = np.array([[np.clip((steer - env.applied[0, 0]) / LIMIT, -1, 1), pedal, 0.0, 0.0]], np.float32)
            obs, _, term, trunc, info = env.step(action)
            if (term | trunc)[0]:
                raise RuntimeError(f"{track}: the placement driver's episode ended at {s:.0f} m, before {todo[0]}")
            while todo and env._raw[0, F["dist_raced"]] >= todo[0][1]:
                label, _ = todo.pop(0)
                frames.append(Frame(label, race, env._raw[0].copy(), obs[0].copy(), env.obs_layout, (zooms or {}).get(label)))
    finally:
        env.close()
    return frames


def to_car(r: np.ndarray, world: np.ndarray) -> np.ndarray:
    c, s = np.cos(r[F["yaw"]]), np.sin(r[F["yaw"]])
    d = world - np.array([r[F["pos_x"]], r[F["pos_y"]]])
    return np.stack([c * d[..., 0] + s * d[..., 1], c * d[..., 1] - s * d[..., 0]], -1)


def panel(ax, prof, f: Frame) -> None:
    L, r, o = f.layout, f.raw, f.obs
    edges = TrackEdges.for_race(f.race)
    left, right = decode_edges(L, o)
    rays = L.group(o, "rays") * RAY_RANGE
    speed = float(np.hypot(r[F["speed_x"]], r[F["speed_y"]]))
    span = max(6.0 * speed, 60.0)
    s0 = float(r[F["dist_from_start"]])
    cells = np.mod((np.arange(s0 - 60, s0 + span + 120) / edges.spacing).astype(int), edges.cells)
    road_l, road_r = to_car(r, edges.table[cells, :2]), to_car(r, edges.table[cells, 2:])
    # the screen's x is -y: the car's left is on the left, ahead is up
    ax.fill(np.r_[-road_l[:, 1], -road_r[::-1, 1]], np.r_[road_l[:, 0], road_r[::-1, 0]], color=ROAD, lw=0, zorder=1)
    for d, length in zip(RAY_DEG, rays):
        a = np.deg2rad(d)
        ax.plot([0, -length * np.sin(a)], [0, length * np.cos(a)], color=RAY, lw=0.6, zorder=2)
    for pts, color, name in ((left, LEFT, "left edge points"), (right, RIGHT, "right edge points")):
        ax.plot(-pts[:, 1], pts[:, 0], "-", color=color, lw=0.8, alpha=0.5, zorder=3)
        ax.plot(-pts[:, 1], pts[:, 0], "o", color=color, ms=3.6, mec="#fcfcfb", mew=0.5, zorder=4, label=name)
    ax.add_patch(plt.Rectangle((-0.95, -2.3), 1.9, 4.6, color=INK, zorder=5))
    if f.zoom:
        a, b = f.zoom
        zc = np.mod((np.arange(a, b) / edges.spacing).astype(int), edges.cells)
        box = np.vstack([to_car(r, edges.table[zc, :2]), to_car(r, edges.table[zc, 2:])])
        pad = 6.0
        ax.set_xlim(-box[:, 1].max() - pad, -box[:, 1].min() + pad)
        ax.set_ylim(box[:, 0].min() - pad, box[:, 0].max() + pad)
    else:
        all_y = np.r_[left[:, 1], right[:, 1]]
        ax.set_xlim(-max(all_y.max(), 15) - 8, -min(all_y.min(), -15) + 8)
        ax.set_ylim(-max(10.0, 0.08 * span), span * 1.06)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{f.label}\n{speed * 3.6:.0f} km/h at s = {s0:.0f} m\nspan {span:.0f} m, spacing {span / (POINTS - 1):.1f} m",
                 fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("m, left of the car  <-  ->  right", fontsize=7.5, color=MUTED)
    ax.set_ylabel("m ahead", fontsize=7.5, color=MUTED)
    k = np.arange(POINTS)
    prof.axhline(0, color=ROAD, lw=1, zorder=0)
    prof.plot(k, L.group(o, "edge_left_y"), "-o", color=LEFT, ms=2.8, lw=1.2, label="left")
    prof.plot(k, L.group(o, "edge_right_y"), "-o", color=RIGHT, ms=2.8, lw=1.2, label="right")
    prof.set_xlabel("point index", fontsize=7.5, color=MUTED)
    prof.set_ylabel("y input (m / 50, + = left)", fontsize=7.5, color=MUTED)
    for a in (ax, prof):
        a.tick_params(labelsize=7, colors=MUTED, length=2)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(RAY)


def sheet(frames: list[Frame], title: str, path: Path) -> None:
    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(max(3.3 * n, 4.8), 8.6), height_ratios=[3.4, 1], squeeze=False,
                             facecolor="#fcfcfb", layout="constrained")
    for i, f in enumerate(frames):
        panel(axes[0, i], axes[1, i], f)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=ROAD, lw=6), plt.Line2D([], [], color=RAY, lw=0.8)]
    labels += ["true road (dense map)", "engine rays"]
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """p [m, 2], segments a, b [k, 2] -> distance of each p to the nearest segment."""
    e = b - a
    t = np.clip(((p[:, None] - a[None]) * e[None]).sum(-1) / np.maximum((e * e).sum(-1), 1e-9)[None], 0, 1)
    return np.hypot(*np.moveaxis(p[:, None] - (a[None] + t[..., None] * e[None]), -1, 0)).min(axis=1)


def resolution_report(points_list=(30, 60)) -> None:
    """Spa's Bus Stop: how far the true edges stray from the polyline through the points, along the bot's own approach."""
    race = race_for("ardennen-spa")
    edges = TrackEdges.for_race(race)
    bot = reference.load(reference_for(race), laps=3).lap_trace()
    order = np.argsort(bot["dist_from_start"])
    bot_s, bot_v = bot["dist_from_start"][order], bot["speed_x"][order]
    a, b = BUS_STOP
    chicane = np.arange(a, b) / edges.spacing
    chicane = chicane.astype(int)
    mid = 0.5 * (edges.table[:, :2] + edges.table[:, 2:])
    head = np.unwrap(np.arctan2(*np.diff(mid, axis=0).T[::-1]))
    curv = np.convolve(np.diff(head), np.ones(5) / 5, "same") / edges.spacing
    chic_c = curv[chicane[0]:chicane[-1]]
    print(f"\nBus Stop {a:.0f}-{b:.0f} m: centreline radius at its tightest {1 / np.abs(chic_c).max():.0f} m, "
          f"signed curvature peaks {chic_c.min():+.3f} / {chic_c.max():+.3f} 1/m")
    print(f"{'points':>6s} {'car at':>7s} {'to chicane':>10s} {'km/h (bot)':>10s} {'span':>6s} {'spacing':>7s} {'pts in it':>9s} "
          f"{'worst edge error, m':>19s}")
    for n in points_list:
        fractions = np.arange(n) / (n - 1)
        for s0 in (6300.0, 6430.0, 6500.0, 6560.0, 6600.0, 6630.0, 6660.0, 6690.0):
            v = float(np.interp(s0, bot_s, bot_v))
            span = max(6 * v, 60.0)
            worst, inside = 0.0, 0
            for phase in np.linspace(0, span / (n - 1), 9, endpoint=False):  # wherever the points fall on the road
                s = s0 + phase + span * fractions
                cells = np.round(s / edges.spacing).astype(int)
                mask = (s >= a) & (s < b)
                inside = max(inside, int(mask.sum()))
                seen = chicane[chicane <= cells[-1]]
                seen = seen[seen >= cells[0]]  # what is left of it, ahead of the first point
                for off in (0, 2):
                    pts = edges.table[cells, off:off + 2].astype(float)
                    worst = max(worst, float(point_segment_distance(edges.table[seen, off:off + 2].astype(float),
                                                                    pts[:-1], pts[1:]).max()) if len(seen) else 0.0)
            print(f"{n:6d} {s0:7.0f} {a - s0:10.0f} {v * 3.6:10.0f} {span:6.0f} {span / (n - 1):7.1f} {inside:9d} {worst:19.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/img/proto-7-obs"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    spa = race_for("ardennen-spa")
    length = TrackEdges.for_race(spa).length
    z = BUS_STOP
    fast = drive("ardennen-spa", [
        ("Hairpin: La Source, entry", HAIRPIN), ("Straight: Kemmel", 1500.0),
        ("Bus Stop: bot's braking point", 6430.0), ("Bus Stop: approach", 6580.0), ("Bus Stop: approach, zoomed", 6581.0),
        ("Bus Stop: at the entry", 6640.0), ("Bus Stop: at the entry, zoomed", 6641.0), ("Bus Stop: in the chicane", 6690.0),
        ("Start line: flying, 60 m before", length - 60.0)], factor=0.85, grid_label="Start line: on the grid",
        zooms={"Bus Stop: approach, zoomed": z, "Bus Stop: at the entry, zoomed": z})
    by = {f.label: f for f in fast}
    slow = drive("ardennen-spa", [("Low speed: 150 m before the chicane", 6490.0), ("Low speed: 60 m before", 6580.0)], cap=15.0)
    high = drive("ardennen-spa", [("High speed: 150 m before the chicane", 6490.0), ("High speed: 60 m before", 6580.0)], factor=0.95)
    karwada = drive("karwada", [("Figure-eight: first pass of the crossing", _crossing()[0]),
                                ("Figure-eight: second pass", _crossing()[1])], factor=0.8)

    sheet([by["Straight: Kemmel"], by["Hairpin: La Source, entry"]], "Straight and hairpin", a.out / "straight_hairpin.png")
    sheet([by[k] for k in ("Bus Stop: bot's braking point", "Bus Stop: approach", "Bus Stop: approach, zoomed",
                           "Bus Stop: at the entry, zoomed", "Bus Stop: in the chicane")],
          "Spa's Bus Stop at the bot's speed x 0.85", a.out / "bus_stop.png")
    sheet([by["Start line: on the grid"], by["Start line: flying, 60 m before"], *karwada], "Start line and figure-eight",
          a.out / "start_line.png")
    sheet([slow[0], high[0], slow[1], high[1]], "The same spots at low and high speed: the corner moves to an earlier index",
          a.out / "low_high.png")
    resolution_report()


def _crossing() -> tuple[float, float]:
    """Lap distances where Karwada crosses itself."""
    e = TrackEdges.for_race(race_for("karwada"))
    mid = 0.5 * (e.table[:, :2] + e.table[:, 2:])
    c = np.arange(0, len(mid), 3)
    d = np.hypot(mid[c, None, 0] - mid[c, 0][None], mid[c, None, 1] - mid[c, 1][None])
    far = np.abs(c[:, None] - c[None]) * e.spacing
    i, j = np.unravel_index(np.argmin(np.where(np.minimum(far, e.length - far) > 500, d, np.inf)), d.shape)
    return tuple(sorted((c[i] * e.spacing - 40.0, c[j] * e.spacing - 40.0)))


if __name__ == "__main__":
    main()
