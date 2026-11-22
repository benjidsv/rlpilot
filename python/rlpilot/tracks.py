"""Tracks: names, the frozen split, and each track's geometry and bot reference on a ~1 m grid along the lap.

A track `t` is raced with data/raceman/rlpilot-<t>.xml (rlpilot.xml for Jarama) and has a built-in bot reference in
rlref-<t>.xml. Running that bot once in the worker's trace mode gives the track map (road edges), the road height at
each edge, the width and the surface beside the road, and the bot's laps; they are cached in reference/, keyed by the
race file's content.

Everything is indexed by the distance along the lap (`s`, m from the start line), never by the nearest point: at a
figure-eight's crossing the nearest point can belong to the other branch.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .client import DATA, ROOT, run_worker_once

CACHE = ROOT / "reference"
SPLIT_FILE = DATA / "split-v2.json"
REF_LAPS = 3
TRACE_COLUMNS = ("time", "laps", "dist_from_start", "dist_raced", "speed_x", "x", "y", "last_lap_time",
                 "steer", "accel", "brake")
# Columns of Track.table, per grid cell.
COLUMNS = ("left_x", "left_y", "right_x", "right_y",  # road edges, world frame (m)
           "left_z", "right_z",                        # road height at each edge (m)
           "side_left", "side_right",                  # width beside each edge before the barrier (m)
           "grass_left", "grass_right",                # 1 where the widest surface beside the edge is grass
           "loose_left", "loose_right",                # 1 where it is loose: sand, gravel or dirt (else it is paved)
           "bot_offset",                               # the bot's lateral offset from the centreline (m, + = left)
           "bot_speed",                                # m/s
           "bot_time",                                 # s since the start line on its fastest lap
           "bot_steer")                                # steering command in [-1, 1]
C = {name: i for i, name in enumerate(COLUMNS)}


@functools.cache
def track_of(race: str) -> str:
    """Track name of a race config (its first <section name="Tracks"> entry)."""
    text = (DATA / "raceman" / f"{race}.xml").read_text()
    return re.search(r'<section name="Tracks">.*?name="name" val="([^"]+)"', text, re.S).group(1)


def race_for(track: str) -> str:
    """The race config the agent drives `track` with."""
    if (DATA / "raceman" / f"rlpilot-{track}.xml").exists():
        return f"rlpilot-{track}"
    if track_of("rlpilot") == track:
        return "rlpilot"
    raise ValueError(f"no race config for track {track!r}")


def all_tracks() -> list[str]:
    return sorted([p.stem.removeprefix("rlpilot-") for p in (DATA / "raceman").glob("rlpilot-*.xml")] + [track_of("rlpilot")])


def load_split() -> dict[str, list[str]]:
    """{"train": [...], "validation": [...], "test": [...]}, drawn once by scripts/make_split.py."""
    return json.loads(SPLIT_FILE.read_text())["tracks"]


# ---- the bot reference files ----

def _prefix(track: str) -> Path:
    """Prefix of the track's cached reference files, running the bot (and the geometry dumps) if they're missing."""
    ref = race_for(track).replace("rlpilot", "rlref", 1)
    key = hashlib.sha1((DATA / "raceman" / f"{ref}.xml").read_bytes() + str(REF_LAPS).encode()).hexdigest()[:10]
    prefix = CACHE / f"{ref}-{key}"
    CACHE.mkdir(exist_ok=True)
    if not Path(f"{prefix}_laps.csv").exists():
        run_worker_once(ref, ["-o", str(prefix), "-n", str(REF_LAPS)])
    for flag, suffix in (("-z", "height"), ("-b", "sides"), ("-g", "surfaces")):  # geometry dumps, no bot laps needed
        if not Path(f"{prefix}_{suffix}.csv").exists():
            run_worker_once(ref, [flag, f"{prefix}_{suffix}.csv"], timeout=120)
    return prefix


# What a driver would call the surface beside the road, from the track's material names: what it looks like, not its
# exact grip. A name that matches no class, or several (blends such as "b-asphalt-grass6", tarmac fading into grass),
# falls back on its physics.
KIND_WORDS = {"loose": ("sand", "gravel", "dirt"),
              "grass": ("grass",),
              "paved": ("asphalt", "road", "concrete", "conc", "curb", "kerb", "tarmac", "pit", "tile", "rwy", "none")}


def surface_kind(material: str, friction: float, rollres: float) -> str:
    """"paved", "grass" or "loose" for a surface beside the road."""
    name = material.lower()
    kinds = [k for k, words in KIND_WORDS.items() if any(w in name for w in words)]
    if len(kinds) == 1:
        return kinds[0]
    if rollres >= 0.1:
        return "loose"
    return "paved" if friction >= 0.7 else "grass"


def surface_kinds(path: str) -> np.ndarray:
    """The surfaces dump -> [m, 4] floats: grass left, grass right, loose left, loose right (paved = all 0)."""
    rows = [line.split() for line in open(path) if not line.startswith("#")]
    out = np.zeros((len(rows), 4), np.float64)
    for i, (lm, lf, lr, rm, rf, rr) in enumerate(rows):
        for side, (m, f, r) in enumerate(((lm, lf, lr), (rm, rf, rr))):
            kind = surface_kind(m, float(f), float(r))
            out[i, side] = kind == "grass"
            out[i, 2 + side] = kind == "loose"
    return out


def lap_times(trace: dict[str, np.ndarray]) -> np.ndarray:
    """Lap times recorded at each start-line crossing after the first (the engine sets last_lap_time then)."""
    laps = trace["laps"]
    ends = np.flatnonzero((laps[1:] > laps[:-1]) & (laps[1:] >= 2)) + 1
    return trace["last_lap_time"][ends].astype(np.float64)


@dataclass
class Track:
    name: str
    length: float        # the engine's lap length (m)
    table: np.ndarray    # [cells + 1, len(COLUMNS)] float32; row `cells` is the start line again (closed)
    spacing: float       # m per cell, just under 1
    bot_lap: float       # the bot's fastest lap (s)

    @property
    def cells(self) -> int:
        return len(self.table) - 1

    @classmethod
    @functools.cache
    def load(cls, name: str) -> "Track":
        prefix = _prefix(name)
        with open(f"{prefix}_track.csv") as f:
            length = float(f.readline().split("length=")[1].split()[0])
        edges = np.loadtxt(f"{prefix}_track.csv", ndmin=2)              # [m, 4] at the engine's border samples
        height = np.loadtxt(f"{prefix}_height.csv", ndmin=2)            # [m, 2] left, right
        sides = np.loadtxt(f"{prefix}_sides.csv", ndmin=2)              # [m, 2]
        kinds = surface_kinds(f"{prefix}_surfaces.csv")                 # [m, 4] grass l/r, gravel l/r
        geometry, cells, spacing = _on_lap_grid(edges, length, np.column_stack([edges, height, sides, kinds]))

        data = np.loadtxt(f"{prefix}_laps.csv", ndmin=2)
        trace = {c: data[:, i] for i, c in enumerate(TRACE_COLUMNS)}
        times = lap_times(trace)
        best = int(np.argmin(times)) + 1  # lap k is driven while `laps` == k; times[0] is lap 1's
        lap = {k: v[trace["laps"] == best] for k, v in trace.items()}
        s = lap["dist_from_start"]
        order = np.argsort(s)
        grid = np.arange(cells + 1) * spacing
        mid = 0.5 * (geometry[:, 0:2] + geometry[:, 2:4])
        left = geometry[:, 0:2] - geometry[:, 2:4]
        left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-6)  # unit vector, right edge -> left edge
        # the bot's offset: its position against the centreline at its own distance from the start
        cell = np.clip(np.rint(s / spacing).astype(np.int64), 0, cells)
        offset = np.einsum("ij,ij->i", np.column_stack([lap["x"], lap["y"]]) - mid[cell], left[cell])
        t = lap["time"] - lap["time"][0]
        per = lambda v: np.interp(grid, s[order], v[order], period=length)
        bot = np.column_stack([per(offset), per(lap["speed_x"]), np.interp(grid, s[order], t[order]), per(lap["steer"])])
        table = np.column_stack([geometry, bot]).astype(np.float32)
        return cls(name, length, table, spacing, float(times[best - 1]))

    def bot_time_between(self, s0: np.ndarray, distance: float) -> np.ndarray:
        """Time (s) the bot takes from s0 to s0 + distance on its fastest lap, across the start line if need be."""
        t0 = TrackTable([self]).at(np.zeros(len(s0), np.int64), np.asarray(s0)[:, None], "bot_time")[:, 0]
        t1 = TrackTable([self]).at(np.zeros(len(s0), np.int64), np.asarray(s0)[:, None] + distance, "bot_time")[:, 0]
        return np.mod(t1 - t0, self.bot_lap)


def _on_lap_grid(edges: np.ndarray, length: float, columns: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Values at the border samples [m, k] -> a uniform grid of centreline distance from the start line, closed: row
    `cells` is the start line again. The distance is the centreline polyline's, anchored at the engine's lap length
    rather than rescaled to it: on Imola the map doesn't quite close (a 5 m last gap where 2 m is the norm), and a
    rescale would spread that error over the lap (3 m at worst, against 0.18 m p99 when it stays in the last cell)."""
    mid = 0.5 * (edges[:, 0:2] + edges[:, 2:4])
    ds = np.hypot(*np.diff(mid, axis=0).T)
    keep = np.concatenate([[True], ds > 1e-6])  # no duplicate abscissa for np.interp
    s = np.concatenate([[0.0], np.cumsum(ds)])[keep]
    if s[-1] >= length:
        raise ValueError(f"track map ({s[-1]:.1f} m before closing) is longer than the lap ({length:.1f} m)")
    cells = int(np.ceil(length))
    spacing = float(length) / cells
    grid = np.arange(cells + 1) * spacing
    s_closed = np.concatenate([s, [length]])
    closed = np.vstack([columns[keep], columns[:1]])
    return np.stack([np.interp(grid, s_closed, c) for c in closed.T], 1), cells, spacing


class TrackTable:
    """Several tracks' tables stacked, so that looking anything up for a batch of envs is one gather."""

    def __init__(self, tracks: list[Track]):
        self.tracks = list(tracks)
        self.table = np.concatenate([t.table for t in self.tracks])
        sizes = [len(t.table) for t in self.tracks]
        self.offset = np.cumsum([0] + sizes[:-1]).astype(np.int64)
        self.cells = np.array([t.cells for t in self.tracks], np.int64)
        self.spacing = np.array([t.spacing for t in self.tracks], np.float64)
        self.length = np.array([t.length for t in self.tracks], np.float64)

    def at(self, track: np.ndarray, s: np.ndarray, columns: str | tuple[str, ...] = COLUMNS) -> np.ndarray:
        """track [n] (index in self.tracks), s [n, k] (m from the start line, any value: it wraps) -> the named
        columns, linearly interpolated: [n, k] for one name, [n, k, len(columns)] for a tuple."""
        cols = C[columns] if isinstance(columns, str) else [C[c] for c in columns]
        u = np.asarray(s, np.float64) / self.spacing[track][:, None]
        cell = np.floor(u)
        row = self.offset[track][:, None] + np.mod(cell.astype(np.int64), self.cells[track][:, None])
        a = self.table[row][..., cols]
        b = self.table[row + 1][..., cols]
        frac = (u - cell).astype(np.float32)
        return a + (frac if isinstance(columns, str) else frac[..., None]) * (b - a)


def half_width(left_xy: np.ndarray, right_xy: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(left_xy - right_xy, axis=-1)


def allocate(weights: dict[str, float], n_workers: int) -> list[str]:
    """Spread n_workers over the tracks in proportion to their weights (e.g. road length), at least one each, by
    largest remainder. Returns one track name per worker, grouped by track."""
    names = sorted(weights)
    if n_workers < len(names):
        raise ValueError(f"{n_workers} workers can't cover {len(names)} tracks")
    w = np.array([weights[t] for t in names], np.float64)
    share = (n_workers - len(names)) * w / w.sum()  # one each, then the rest by weight
    counts = 1 + np.floor(share).astype(int)
    for i in np.argsort(-(share - np.floor(share)))[: n_workers - counts.sum()]:
        counts[i] += 1
    return [t for t, c in zip(names, counts) for _ in range(c)]
