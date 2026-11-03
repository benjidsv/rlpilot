"""Track names -> race configs, and the centreline curvature profile used for the lookahead observation.

A track `t` is raced with data/raceman/rlpilot-<t>.xml (or rlpilot.xml if that is its track) and has
its built-in bot reference in rlref-<t>.xml (rlref.xml). The geometry comes from the reference's
border samples (reference/<ref>_track.csv), so no extra engine run or protocol field is needed.
"""

from __future__ import annotations

import functools
import re

import numpy as np

from . import reference, workdirs

LOOKAHEAD_EDGES = (0, 10, 25, 50, 75, 100, 150, 200, 250, 300)  # m ahead; one feature per window between edges
CURVATURE_SCALE = 25.0  # obs = curvature (1/m) x this: a 25 m radius bend reads 1, clipped to +-3
PEAK_SMOOTHING = 10     # m, moving average of the per-metre curvature before taking a window's peak


@functools.cache
def track_of(race: str) -> str:
    """Track name of a race config (its first <section name="Tracks"> entry)."""
    text = (workdirs.DATA / "raceman" / f"{race}.xml").read_text()
    return re.search(r'<section name="Tracks">.*?name="name" val="([^"]+)"', text, re.S).group(1)


def race_for(track: str) -> str:
    if (workdirs.DATA / "raceman" / f"rlpilot-{track}.xml").exists():
        return f"rlpilot-{track}"
    if track_of("rlpilot") == track:
        return "rlpilot"
    known = sorted(p.stem.removeprefix("rlpilot-") for p in (workdirs.DATA / "raceman").glob("rlpilot-*.xml"))
    raise ValueError(f"no race config for track {track!r} (have {[track_of('rlpilot')] + known})")


def reference_for(race: str) -> str:
    return race.replace("rlpilot", "rlref", 1)


class TrackProfile:
    """Unwrapped centreline heading on a 1 m grid; `lookahead(s)` gives the curvature over each window
    [s + edges[k], s + edges[k+1]] ahead of distance-from-start s (+ = turning left, like to_middle).
    stat "mean": the window's mean curvature (heading change / width), which dilutes a short tight corner in a
    wide far window. "peak": the signed curvature of largest magnitude in the window (after PEAK_SMOOTHING),
    so a chicane 250 m ahead reads as tight as it will be when the car gets there."""

    def __init__(self, left: np.ndarray, right: np.ndarray, length: float, edges=LOOKAHEAD_EDGES,
                 stat: str = "peak"):
        if stat not in ("peak", "mean"):
            raise ValueError(f"stat must be 'peak' or 'mean', not {stat!r}")
        self.stat = stat
        mid = 0.5 * (left + right)
        seg = np.diff(np.vstack([mid, mid[:1]]), axis=0)  # closed loop
        ds = np.hypot(seg[:, 0], seg[:, 1])
        keep = ds > 1e-6
        seg, ds = seg[keep], ds[keep]
        s = np.concatenate([[0.0], np.cumsum(ds)[:-1]]) * (length / ds.sum())  # match the engine's lap length
        heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        self.length = float(length)
        self.total_turn = float(heading[-1] - heading[0] + np.angle(np.exp(1j * (heading[0] - heading[-1]))))
        self.edges = np.asarray(edges, np.float64)
        n = int(np.ceil(length)) + int(self.edges[-1]) + 2
        grid = np.arange(n, dtype=np.float64)
        s_ext = np.concatenate([s, s + length])
        h_ext = np.concatenate([heading, heading + self.total_turn])
        self.heading = np.interp(grid, s_ext, h_ext)  # heading at 0, 1, 2, ... m (extended past the line)

    @classmethod
    @functools.cache
    def for_race(cls, race: str, stat: str = "peak") -> "TrackProfile":
        track = reference.load_track(reference_for(race), laps=3)
        return cls(track.left, track.right, track.length, stat=stat)

    @property
    def n_features(self) -> int:
        return len(self.edges) - 1

    def lookahead(self, dist_from_start: np.ndarray) -> np.ndarray:
        """[n] distances (m) -> [n, n_features] scaled, clipped mean curvatures."""
        s = np.mod(np.asarray(dist_from_start, np.float64), self.length)
        if self.stat == "peak":
            return self._peaks[np.minimum(s.astype(np.int64), len(self._peaks) - 1)]
        h = np.interp(s[:, None] + self.edges[None, :], np.arange(len(self.heading)), self.heading)
        k = np.diff(h, axis=1) / np.diff(self.edges)[None, :]
        return np.clip(k * CURVATURE_SCALE, -3, 3).astype(np.float32)

    @functools.cached_property
    def _peaks(self) -> np.ndarray:
        """"peak" lookahead at every metre of the lap, [ceil(length), n_features] (window starts rounded down)."""
        k = np.diff(self.heading)  # curvature (1/m) over each metre
        w = PEAK_SMOOTHING
        k = np.convolve(np.pad(k, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")
        starts = np.arange(int(np.ceil(self.length)))
        out = np.empty((len(starts), self.n_features), np.float32)
        for j, (lo, hi) in enumerate(zip(self.edges[:-1].astype(int), self.edges[1:].astype(int))):
            win = np.lib.stride_tricks.sliding_window_view(k, hi - lo)[starts + lo]
            out[:, j] = win[np.arange(len(starts)), np.abs(win).argmax(axis=1)]
        return np.clip(out * CURVATURE_SCALE, -3, 3)

    @functools.cached_property
    def table(self) -> np.ndarray:
        """lookahead() at every metre of the lap: [ceil(length), n_features] float32, for O(1) lookups."""
        return self.lookahead(np.arange(int(np.ceil(self.length)), dtype=np.float64))


class LookaheadTable:
    """All races' lookahead tables stacked, so a vector step is one gather: row = offset[race] + metre."""

    def __init__(self, races, stat: str = "peak"):
        races = list(dict.fromkeys(races))
        profiles = [TrackProfile.for_race(r, stat) for r in races]
        self.n_features = profiles[0].n_features
        sizes = [len(p.table) for p in profiles]
        self.table = np.concatenate([p.table for p in profiles])
        self.offset = dict(zip(races, np.cumsum([0] + sizes[:-1]).tolist()))
        self.size = dict(zip(races, sizes))

    def rows(self, races) -> tuple[np.ndarray, np.ndarray]:
        """Per-env (offset, size) arrays for the given per-env races."""
        return (np.array([self.offset[r] for r in races], np.int64), np.array([self.size[r] for r in races], np.int64))

    def __call__(self, dist_from_start: np.ndarray, offset: np.ndarray, size: np.ndarray) -> np.ndarray:
        m = np.mod(dist_from_start.astype(np.int64), size)
        return self.table[offset + m]


LINE_POINTS = (0, 10, 25, 50, 75, 100, 150, 200)  # m ahead where the racing-line observation is sampled


class RacingLine:
    """The built-in bot's racing line: its lateral offset from the centreline (m, + = left, like to_middle) at
    every metre of the lap, from its fastest reference lap. Simplix computes this line itself (clothoid-smoothed,
    unitclothoid.cpp), so it is a decent line on any track that has a reference."""

    def __init__(self, left: np.ndarray, right: np.ndarray, length: float, trace: dict[str, np.ndarray]):
        mid = 0.5 * (left + right)
        seg = np.diff(np.vstack([mid, mid[:1]]), axis=0)  # closed loop, as in TrackProfile
        ds = np.hypot(seg[:, 0], seg[:, 1])
        keep = ds > 1e-6
        mid, seg, ds = mid[keep], seg[keep], ds[keep]
        self.length = float(length)
        self._s = np.concatenate([[0.0], np.cumsum(ds)[:-1]]) * (length / ds.sum())
        self._mid = mid
        self._left = np.stack([-seg[:, 1], seg[:, 0]], 1) / ds[:, None]  # unit normal pointing left
        lat = self.lateral(trace["x"], trace["y"], trace["dist_from_start"])
        grid = np.arange(int(np.ceil(length)), dtype=np.float64)
        self.table = np.interp(grid, trace["dist_from_start"], lat, period=length).astype(np.float32)
        # the bot's own mean |offset| / half width, for comparison with the agent's (behavior/abs_offset)
        half_width = 0.5 * np.hypot(*(left[keep] - right[keep]).T)
        self.abs_offset = float(np.mean(np.abs(self.table) / np.interp(grid, self._s, half_width, period=length)))

    def lateral(self, x: np.ndarray, y: np.ndarray, dist_from_start: np.ndarray) -> np.ndarray:
        """Signed lateral offset (m, + = left) of positions from the centreline at the same distance from start."""
        s = np.mod(np.asarray(dist_from_start, np.float64), self.length)
        p = lambda a: np.interp(s, self._s, a, period=self.length)
        mx, my = p(self._mid[:, 0]), p(self._mid[:, 1])
        nx, ny = p(self._left[:, 0]), p(self._left[:, 1])
        return ((np.asarray(x) - mx) * nx + (np.asarray(y) - my) * ny) / np.hypot(nx, ny)

    @classmethod
    @functools.cache
    def for_race(cls, race: str) -> "RacingLine":
        ref = reference.load(reference_for(race), laps=3)
        return cls(ref.track_left, ref.track_right, ref.track_length, ref.lap_trace())


class LineTable:
    """All races' racing lines stacked, like LookaheadTable: row = offset[race] + metre."""

    def __init__(self, races):
        races = list(dict.fromkeys(races))
        tables = [RacingLine.for_race(r).table for r in races]
        self.table = np.concatenate(tables)
        sizes = [len(t) for t in tables]
        self.offset = dict(zip(races, np.cumsum([0] + sizes[:-1]).tolist()))
        self.size = dict(zip(races, sizes))
        self.points = np.asarray(LINE_POINTS, np.int64)

    def rows(self, races) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.offset[r] for r in races], np.int64), np.array([self.size[r] for r in races], np.int64))

    def __call__(self, dist_from_start: np.ndarray, offset: np.ndarray, size: np.ndarray, ahead=(0,)) -> np.ndarray:
        """Line offset (m) at each distance + each of `ahead` metres: [n, len(ahead)]."""
        m = np.mod(dist_from_start.astype(np.int64)[:, None] + np.asarray(ahead, np.int64)[None, :], size[:, None])
        return self.table[offset[:, None] + m]


class TrackEdges:
    """One track's left and right borders on a uniform grid of centreline distance from the start line, closed: row
    `cells` is the start line again, so interpolating across the line needs no special case.

    The borders are indexed by the distance along the lap, never by the nearest point: at a figure-eight's crossing,
    the nearest border can belong to the other branch. Their distance is the centreline polyline's, anchored at the
    engine's lap length rather than rescaled to it: on Imola the map doesn't quite close (a 5 m last gap where 2 m is
    the norm, in the engine's own geometry), and a rescale spreads that error over the lap, 3 m at worst, against
    0.18 m (p99) when it stays in the last cell. Geometry only: nothing here knows where any car drove."""

    def __init__(self, left: np.ndarray, right: np.ndarray, length: float):
        mid = 0.5 * (left + right)
        ds = np.hypot(*np.diff(mid, axis=0).T)
        keep = np.concatenate([[True], ds > 1e-6])  # no duplicate abscissa for np.interp
        s = np.concatenate([[0.0], np.cumsum(ds)])[keep]
        if s[-1] >= length:
            raise ValueError(f"track map ({s[-1]:.1f} m before closing) is longer than the lap ({length:.1f} m)")
        self.length = float(length)
        self.cells = int(np.ceil(length))
        self.spacing = self.length / self.cells  # just under 1 m
        grid = np.arange(self.cells + 1) * self.spacing
        s_closed = np.concatenate([s, [length]])
        columns = np.hstack([left[keep], right[keep]])
        columns = np.vstack([columns, columns[:1]])
        self.table = np.stack([np.interp(grid, s_closed, c) for c in columns.T], 1).astype(np.float32)  # lx ly rx ry

    @classmethod
    @functools.cache
    def for_race(cls, race: str) -> "TrackEdges":
        track = reference.load_track(reference_for(race), laps=3)
        return cls(track.left, track.right, track.length)


class EdgeTable:
    """All races' borders stacked, so a vector step is one gather: row = offset[race] + cell, per env and per point.

    Points k = 0..n-1 are at centreline distance s + span x k / (n - 1) along the lap, with
    span = max(horizon x speed, floor): the first one is abreast of the car, and a corner reaches a given point index
    earlier when the car is faster. Returned in the car's frame, in metres: x forward, y + = left, like to_middle."""

    def __init__(self, races, n_points: int, horizon: float, floor: float):
        if n_points < 2:
            raise ValueError("an edge needs at least 2 points")
        self.horizon, self.floor = float(horizon), float(floor)
        self.fractions = np.arange(n_points, dtype=np.float64)[None, :] / (n_points - 1)
        self.races = list(dict.fromkeys(races))
        edges = [TrackEdges.for_race(r) for r in self.races]
        self.table = np.concatenate([e.table for e in edges])
        sizes = [len(e.table) for e in edges]
        self.offset = dict(zip(self.races, np.cumsum([0] + sizes[:-1]).tolist()))
        self.cells = {r: e.cells for r, e in zip(self.races, edges)}
        self.spacing = {r: e.spacing for r, e in zip(self.races, edges)}

    def rows(self, races) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-env (offset, cells, spacing) arrays for the given per-env races."""
        return (np.array([self.offset[r] for r in races], np.int64), np.array([self.cells[r] for r in races], np.int64),
                np.array([self.spacing[r] for r in races], np.float64))

    def span(self, speed: np.ndarray) -> np.ndarray:
        """Distance (m) covered by the points at a speed (m/s, its magnitude)."""
        return np.maximum(self.horizon * np.abs(speed), self.floor)

    def world(self, dist_from_start: np.ndarray, speed: np.ndarray, offset: np.ndarray, cells: np.ndarray,
              spacing: np.ndarray) -> np.ndarray:
        """[n] state, per-env rows -> [n, n_points, 4] world coordinates (left x, left y, right x, right y)."""
        s = np.asarray(dist_from_start, np.float64)[:, None] + self.span(speed)[:, None] * self.fractions
        u = s / spacing[:, None]
        cell = np.floor(u)
        frac = (u - cell).astype(np.float32)[..., None]
        row = offset[:, None] + np.mod(cell.astype(np.int64), cells[:, None])  # wraps, whatever the sign of s
        a = self.table[row]
        return a + frac * (self.table[row + 1] - a)

    def car_frame(self, x: np.ndarray, y: np.ndarray, yaw: np.ndarray, world: np.ndarray) -> np.ndarray:
        """[n] pose, [n, n_points, 4] world -> [n, n_points, 4] car frame (left x, left y, right x, right y), metres."""
        c, s = np.cos(yaw)[:, None, None], np.sin(yaw)[:, None, None]
        dx = world[..., ::2] - x[:, None, None]
        dy = world[..., 1::2] - y[:, None, None]
        out = np.empty_like(world)
        out[..., ::2] = c * dx + s * dy
        out[..., 1::2] = c * dy - s * dx
        return out
