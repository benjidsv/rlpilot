"""Evaluation on the fixed scenario suite (data/suite-v2.csv, see scripts/make_suite.py), and grid laps.

A scenario is a start (track, place, tier, offsets, kick) and a task: drive SCENARIO_DISTANCE along the track. Every
checkpoint of every run plays exactly the same list, with the same batch layout, so results are reproducible and two
runs can be compared scenario by scenario (scripts/compare.py).

Outcome of a scenario, best first:
- clean:     reached the distance, never fully off the road, never lost control
- survived:  reached it, went fully off the road at some point but kept control
- recovered: reached it, but lost control on the way (hit something, faced backwards, or got far off the road)
- failed:    didn't reach it (no progress for 10 s, wrecked, or out of time)
"survival" = clean or survived. Crashed starts begin out of control, so for them the useful number is "completion"
(anything but failed).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from .client import DATA, F
from .env import ENDS, RacingEnv
from .starts import KINDS, Starts
from .tracks import Track, allocate

SUITE = DATA / "suite-v2.csv"
SCENARIO_DISTANCE = 500.0  # m along the track
SCENARIO_SECONDS = 60.0    # 500 m at 30 km/h on average: a car that hasn't made it by then failed
OUTCOMES = ("failed", "recovered", "survived", "clean")
STARTS_COLUMNS = ("kind", "s", "lateral", "yaw", "speed", "kick_steer", "kick_pedal", "kick_steps")
Policy = Callable[[np.ndarray], np.ndarray]  # observations [n, DIM] -> actions [n, 3], deterministic


def load_suite(splits: tuple[str, ...] = ("validation",), path: Path = SUITE) -> list[dict]:
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in splits]
    for r in rows:
        for k in ("s", "lateral", "yaw", "speed", "kick_steer", "kick_pedal", "bot_time"):
            r[k] = float(r[k])
        r["kick_steps"] = int(r["kick_steps"])
        r["kind"] = KINDS.index(r["tier"])
    return rows


class _Layout:
    """The suite's fixed layout: workers spread over the tracks in proportion to their number of scenarios, each working
    through its share in suite order. `start_fn` hands each env its next scenario (`running`, -1 = none left, then a
    grid start and the env is paused)."""

    def __init__(self, suite: list[dict], n_workers: int):
        counts = {}
        for r in suite:
            counts[r["track"]] = counts.get(r["track"], 0) + 1
        self.suite, self.tracks = suite, allocate(counts, max(n_workers, len(counts)))
        self.queues: list[list[int]] = [[] for _ in self.tracks]
        for t in counts:  # a track's scenarios, dealt round-robin to its workers
            mine = [w for w, wt in enumerate(self.tracks) if wt == t]
            for j, i in enumerate([i for i, r in enumerate(suite) if r["track"] == t]):
                self.queues[mine[j % len(mine)]].append(i)
        self.running = np.full(len(self.tracks), -1)

    def start_fn(self, idx: np.ndarray) -> Starts:
        cols = {k: np.zeros(len(idx), np.int64 if k in ("kind", "kick_steps") else np.float64) for k in STARTS_COLUMNS}
        for j, e in enumerate(idx):
            self.running[e] = self.queues[e].pop(0) if self.queues[e] else -1
            if self.running[e] >= 0:
                for k in STARTS_COLUMNS:
                    cols[k][j] = self.suite[self.running[e]][k]
        return Starts(**cols)

    def bad_launch(self, idx: np.ndarray, attempt: int) -> Starts:
        ids = [self.suite[self.running[e]]["id"] for e in idx if self.running[e] >= 0]
        raise RuntimeError(f"suite scenarios with a non-finite launch: {ids} (make_suite.py checks launches: regenerate)")


def run_suite(policy: Policy, suite: list[dict], n_workers: int = 48) -> list[dict]:
    """Play every scenario once; returns one result row per scenario, in suite order."""
    lay = _Layout(suite, n_workers)
    results: list[dict | None] = [None] * len(suite)
    lengths = {t: Track.load(t).length for t in set(lay.tracks)}
    env = RacingEnv(lay.tracks, episode_seconds=SCENARIO_SECONDS, max_distance=SCENARIO_DISTANCE,
                    start_fn=lay.start_fn, retry_fn=lay.bad_launch)
    try:
        obs, _ = env.reset()
        while (lay.running >= 0).any():
            before = lay.running.copy()
            obs, _, _, _, info = env.step(policy(obs), before >= 0)
            if "episode_end" not in info:
                continue
            e = info["episode_end"]
            for i in np.flatnonzero(info["_episode_end"]):
                sc = suite[before[i]]
                end = ENDS[e["end"][i]]
                reached = end == "distance"
                outcome = ("recovered" if e["lost_control"][i] else "survived" if e["ever_off"][i] else "clean") \
                    if reached else "failed"
                results[before[i]] = {"id": sc["id"], "split": sc["split"], "track": sc["track"], "tier": sc["tier"],
                                      "s": sc["s"], "outcome": outcome, "end": end,
                                      "distance": round(float(e["distance"][i]), 1),
                                      "seconds": round(float(e["seconds"][i]), 2), "bot_seconds": sc["bot_time"],
                                      "pace": round(sc["bot_time"] / e["seconds"][i], 4) if reached else "",
                                      "end_s": round(float(np.mod(sc["s"] + e["distance"][i], lengths[sc["track"]])), 1)}
    finally:
        env.close()
    assert all(r is not None for r in results)
    return results


def check_launches(suite: list[dict], n_workers: int = 48) -> list[int]:
    """Launch every scenario once (with its kick) without driving it; returns the indices of those whose state comes
    back non-finite. The engine is deterministic, so a suite without any always launches cleanly."""
    lay = _Layout(suite, n_workers)
    bad: list[int] = []

    def record(idx: np.ndarray, attempt: int) -> Starts:
        bad.extend(int(lay.running[e]) for e in idx if lay.running[e] >= 0)
        z = np.zeros(len(idx))
        return Starts(np.zeros(len(idx), np.int64), z, z, z, z, z, z, np.zeros(len(idx), np.int64))  # the grid

    env = RacingEnv(lay.tracks, start_fn=lay.start_fn, retry_fn=record)
    try:
        env.reset()
        while (lay.running >= 0).any():
            env._start(np.flatnonzero(lay.running >= 0))
    finally:
        env.close()
    return sorted(set(bad))


def write_results(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_results(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict], n_boot: int = 2000) -> dict[str, float]:
    """Metrics per split ("{split}/..."): pooled over scenarios, so every 200 m of road weighs the same; the 90% interval
    of the headline is a bootstrap over tracks (the unit of generalization)."""
    m: dict[str, float] = {}
    for split in sorted({r["split"] for r in rows}):
        sp = [r for r in rows if r["split"] == split]
        for tier in sorted({r["tier"] for r in sp}, key=KINDS.index):
            t = [r for r in sp if r["tier"] == tier]
            m[f"{split}/{tier}/survival"] = float(np.mean([r["outcome"] in ("clean", "survived") for r in t]))
            m[f"{split}/{tier}/completion"] = float(np.mean([r["outcome"] != "failed" for r in t]))
            m[f"{split}/{tier}/clean"] = float(np.mean([r["outcome"] == "clean" for r in t]))
        nom = [r for r in sp if r["tier"] == "nominal"]
        if not nom:
            continue
        paces = [float(r["pace"]) for r in nom if r["outcome"] in ("clean", "survived")]
        m[f"{split}/pace"] = float(np.median(paces)) if paces else float("nan")
        by_track = {}
        for r in nom:
            by_track.setdefault(r["track"], []).append(r["outcome"] in ("clean", "survived"))
        groups = list(by_track.values())
        rng = np.random.default_rng(0)
        boot = [np.mean(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n_boot)]
        m[f"{split}/nominal/survival_lo"], m[f"{split}/nominal/survival_hi"] = map(float, np.percentile(boot, [5, 95]))
    return m


def per_track(rows: list[dict]) -> list[dict]:
    """One row per track: survival per tier, and pace."""
    out = []
    for track in sorted({r["track"] for r in rows}):
        t = [r for r in rows if r["track"] == track]
        row = {"track": track, "split": t[0]["split"], "scenarios": len(t)}
        for tier in sorted({r["tier"] for r in t}, key=KINDS.index):
            tt = [r for r in t if r["tier"] == tier]
            row[tier] = float(np.mean([r["outcome"] in ("clean", "survived") for r in tt])) if tier != "crashed" \
                else float(np.mean([r["outcome"] != "failed" for r in tt]))
        paces = [float(r["pace"]) for r in t if r["tier"] == "nominal" and r["outcome"] in ("clean", "survived")]
        row["pace"] = float(np.median(paces)) if paces else float("nan")
        out.append(row)
    return out


def grid_laps(policy: Policy, tracks: list[str], laps: int = 2) -> list[dict]:
    """The real task: `laps` laps from a standing start on the grid, one episode per track (the sim and the policy are
    deterministic, so more would be copies). Ends when the laps are done, or on a failure."""
    def grid(idx):
        z = np.zeros(len(idx))
        return Starts(np.zeros(len(idx), np.int64), z, z, z, z, z, z, np.zeros(len(idx), np.int64))
    bot = {t: Track.load(t).bot_lap for t in tracks}
    env = RacingEnv(list(tracks), episode_seconds=laps * max(bot.values()) / 0.6 + 30, start_fn=grid)
    out = []
    try:
        obs, _ = env.reset()
        active = np.ones(len(tracks), bool)
        times: list[list[float]] = [[] for _ in tracks]
        prev_laps = env._raw[:, F["laps"]].copy()
        end = [""] * len(tracks)
        while active.any():
            obs, _, term, trunc, info = env.step(policy(obs), active)
            raw = env._raw
            for i in np.flatnonzero(active):
                done = term[i] or trunc[i]
                if not done and raw[i, F["laps"]] > prev_laps[i] and raw[i, F["laps"]] >= 2:
                    times[i].append(float(raw[i, F["last_lap_time"]]))
                prev_laps[i] = raw[i, F["laps"]]
                if len(times[i]) >= laps or done:
                    active[i] = False
                    end[i] = "laps_done" if len(times[i]) >= laps else ENDS[info["episode_end"]["end"][i]]
    finally:
        env.close()
    for i, t in enumerate(tracks):
        out.append({"track": t, "end": end[i], "laps": len(times[i]), "lap_times": times[i], "bot_lap": bot[t],
                    "best_vs_bot": (min(times[i]) / bot[t]) if times[i] else float("nan")})
    return out

