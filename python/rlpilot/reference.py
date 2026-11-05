"""Built-in bot reference on the same track and car: lap times, per-step trace, and track borders.

Produced once by running rlpilot-worker in trace mode on the reference race (e.g. data/raceman/
rlref.xml: the rlpilot race with the agent replaced by simplix) and cached in reference/.
The cache is keyed by the race file's content, so editing the race regenerates it.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import workdirs

CACHE = workdirs.ROOT / "reference"
TRACE_COLUMNS = ("time", "laps", "dist_from_start", "dist_raced", "speed_x", "x", "y", "last_lap_time",
                 "steer", "accel", "brake")


@dataclass
class TrackGeometry:
    left: np.ndarray    # [m, 2] border polylines, from the start line on, with the same samples on both sides
    right: np.ndarray
    length: float       # the engine's lap length


@dataclass
class Reference:
    name: str
    lap_times: np.ndarray        # completed laps, in order (first one from a standing start)
    trace: dict[str, np.ndarray]  # TRACE_COLUMNS, sampled every 20 ms
    track_left: np.ndarray       # [m, 2] border polylines
    track_right: np.ndarray
    track_length: float

    @property
    def best_lap(self) -> float:
        return float(self.lap_times.min()) if len(self.lap_times) else float("nan")

    @property
    def median_lap(self) -> float:
        return float(np.median(self.lap_times)) if len(self.lap_times) else float("nan")

    def lap_trace(self, lap: int | None = None) -> dict[str, np.ndarray]:
        """Samples of one completed lap (1-based; default: the fastest)."""
        if lap is None:
            lap = int(np.argmin(self.lap_times)) + 1
        return lap_slice(self.trace, lap)


def lap_slice(trace: dict[str, np.ndarray], lap: int) -> dict[str, np.ndarray]:
    """Lap k (1-based) is driven while the start-line crossing counter `laps` equals k."""
    m = trace["laps"] == lap
    return {k: v[m] for k, v in trace.items()}


def lap_times_from_trace(trace: dict[str, np.ndarray]) -> np.ndarray:
    """Lap times recorded at each crossing after the first (the engine sets last_lap_time then)."""
    laps = trace["laps"]
    ends = np.flatnonzero((laps[1:] > laps[:-1]) & (laps[1:] >= 2)) + 1
    return trace["last_lap_time"][ends].astype(np.float64)


def _cached(race: str, laps: int) -> Path:
    """Prefix of a race's cached reference files, generating them if needed."""
    race_xml = workdirs.DATA / "raceman" / f"{race}.xml"
    key = hashlib.sha1(race_xml.read_bytes() + str(laps).encode()).hexdigest()[:10]
    prefix = CACHE / f"{race}-{key}"
    if not Path(f"{prefix}_laps.csv").exists():
        _generate(race, laps, prefix)
    return prefix


def load_track(race: str = "rlref", laps: int = 3) -> TrackGeometry:
    """Only the track map, which the engine samples from its segments before the race. Never reads the bot's laps:
    everything that feeds a policy input must come through here."""
    prefix = _cached(race, laps)
    track = np.loadtxt(f"{prefix}_track.csv", ndmin=2)
    with open(f"{prefix}_track.csv") as f:
        length = float(f.readline().split("length=")[1].split()[0])
    return TrackGeometry(track[:, 0:2], track[:, 2:4], length)


def load_heights(race: str = "rlref", laps: int = 3) -> np.ndarray:
    """Road height (m) at the track map's points, row for row: [m, 2] left, right. In a file of its own, written on first
    use from the engine's geometry, so that no bot laps have to run again."""
    path = Path(f"{_cached(race, laps)}_height.csv")
    if not path.exists():
        with tempfile.TemporaryDirectory(prefix="rlpilot-ref-") as tmp:
            (d,) = workdirs.prepare(Path(tmp), 1, race)
            log = d / "worker.log"
            with open(log, "wb") as out:
                rc = subprocess.run([str(workdirs.WORKER_BIN), "-lc", f"{d}/", "-s", race, "-z", str(path), "-nr", "-tl", "1"],
                                    stdout=out, stderr=subprocess.STDOUT, timeout=120).returncode
            if rc != 0 or not path.exists():
                raise RuntimeError(f"heights run failed (rc={rc}):\n{log.read_text(errors='replace')[-2000:]}")
    return np.loadtxt(path, ndmin=2)


def load(race: str = "rlref", laps: int = 3) -> Reference:
    geometry = load_track(race, laps)
    data = np.loadtxt(f"{_cached(race, laps)}_laps.csv", ndmin=2)
    trace = {c: data[:, i] for i, c in enumerate(TRACE_COLUMNS)}
    return Reference(race, lap_times_from_trace(trace), trace, geometry.left, geometry.right, geometry.length)


def _generate(race: str, laps: int, prefix: Path) -> None:
    CACHE.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rlpilot-ref-") as tmp:
        (d,) = workdirs.prepare(Path(tmp), 1, race)
        log = d / "worker.log"
        with open(log, "wb") as out:
            rc = subprocess.run([str(workdirs.WORKER_BIN), "-lc", f"{d}/", "-s", race, "-o", str(prefix),
                                 "-n", str(laps), "-nr", "-tl", "1"], stdout=out, stderr=subprocess.STDOUT,
                                timeout=600).returncode
        if rc != 0 or not Path(f"{prefix}_laps.csv").exists():
            raise RuntimeError(f"reference run failed (rc={rc}):\n{log.read_text(errors='replace')[-2000:]}")
