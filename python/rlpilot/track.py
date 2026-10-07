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
CURVATURE_SCALE = 25.0  # obs = mean curvature (1/m) x this: a 25 m radius bend reads 1, clipped to +-3


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
    """Unwrapped centreline heading on a 1 m grid; `lookahead(s)` gives the mean curvature over each window
    [s + edges[k], s + edges[k+1]] ahead of distance-from-start s (+ = turning left, like to_middle)."""

    def __init__(self, left: np.ndarray, right: np.ndarray, length: float, edges=LOOKAHEAD_EDGES):
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
    def for_race(cls, race: str) -> "TrackProfile":
        ref = reference.load(reference_for(race), laps=3)
        return cls(ref.track_left, ref.track_right, ref.track_length)

    @property
    def n_features(self) -> int:
        return len(self.edges) - 1

    def lookahead(self, dist_from_start: np.ndarray) -> np.ndarray:
        """[n] distances (m) -> [n, n_features] scaled, clipped mean curvatures."""
        s = np.mod(np.asarray(dist_from_start, np.float64), self.length)
        h = np.interp(s[:, None] + self.edges[None, :], np.arange(len(self.heading)), self.heading)
        k = np.diff(h, axis=1) / np.diff(self.edges)[None, :]
        return np.clip(k * CURVATURE_SCALE, -3, 3).astype(np.float32)

    @functools.cached_property
    def table(self) -> np.ndarray:
        """lookahead() at every metre of the lap: [ceil(length), n_features] float32, for O(1) lookups."""
        return self.lookahead(np.arange(int(np.ceil(self.length)), dtype=np.float64))


class LookaheadTable:
    """All races' lookahead tables stacked, so a vector step is one gather: row = offset[race] + metre."""

    def __init__(self, races):
        races = list(dict.fromkeys(races))
        profiles = [TrackProfile.for_race(r) for r in races]
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
