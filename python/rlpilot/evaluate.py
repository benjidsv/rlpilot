"""Evaluation on the fixed scenario suite (data/suite-v2.csv, see scripts/setup/make_suite.py), and grid laps.

A scenario is a start (track, place, tier, offsets, kick) and a task: drive SCENARIO_DISTANCE along the track. Every
checkpoint of every run plays exactly the same list, so results are reproducible and two runs can be compared scenario
by scenario (scripts/analysis/compare.py). Each scenario is a fresh race, so its result doesn't depend on the worker layout
(checked: 48 and 64 workers give the same CSV).

Outcome of a scenario, best first:
- clean:     reached the distance, never fully off the road, never scraped the ground, never lost control
- survived:  reached it, went fully off the road or scraped the ground at some point, but kept control
- recovered: reached it, but lost control on the way (hit a wall, faced backwards, or got far off the road)
- failed:    didn't reach it (no progress for 10 s, wrecked, or out of time)
A scrape is damage from the ground (bottoming out in a dip, a kerb, a landing), not from a wall: the car kept control,
but it isn't clean. Caveat: the bot scrapes too, at a few dips (2026-11-24: 33 times in 599 km, at fixed spots on
Charmey, Ole-road-1, Brondehach, Spring, E-track-4 and Ruudskogen), so a scrape there is the track, not the driving,
and "clean" is slightly too strict on those tracks.
Losing control uses the reward's tests (reward.LOST_CAUSES): `lost_first` is the first cause seen, `contact`,
`backwards` and `far_off` count how often each was entered (a state the launch began in isn't counted), `damage` is in
points.
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
from .reward import LOST_CAUSES
from .starts import KINDS, Starts
from .tracks import TRACE_COLUMNS, Track, _prefix

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
        r["goal"] = float(r.get("goal") or SCENARIO_DISTANCE)  # m to drive (corner scenarios set their own)
    return rows


def spread(counts: dict[str, int], n_workers: int) -> list[str]:
    """Workers per track for the suite: one each, then each next worker to the track whose workers have the most
    scenarios to play each, so that the busiest worker, which sets the eval's length, has as few as possible. Returns
    one track name per worker, grouped by track."""
    names = sorted(counts)
    workers = dict.fromkeys(names, 1)
    for _ in range(n_workers - len(names)):
        workers[max(names, key=lambda t: counts[t] / workers[t])] += 1
    return [t for t in names for _ in range(workers[t])]


class _Layout:
    """The suite's layout: workers spread over the tracks by `spread`, each working through its share in suite order.
    `start_fn` hands each env its next scenario (`running`, -1 = none left, then a grid start and the env is paused);
    `refill` queues the whole suite again."""

    def __init__(self, suite: list[dict], n_workers: int):
        counts = {}
        for r in suite:
            counts[r["track"]] = counts.get(r["track"], 0) + 1
        self.suite, self.tracks = suite, spread(counts, max(n_workers, len(counts)))
        self.plan: list[list[int]] = [[] for _ in self.tracks]
        self.goal = np.full(len(self.tracks), SCENARIO_DISTANCE)  # per worker: its current scenario's distance
        for t in counts:  # a track's scenarios, dealt round-robin to its workers
            mine = [w for w, wt in enumerate(self.tracks) if wt == t]
            for j, i in enumerate([i for i, r in enumerate(suite) if r["track"] == t]):
                self.plan[mine[j % len(mine)]].append(i)
        self.refill()

    def refill(self) -> None:
        self.queues = [list(q) for q in self.plan]
        self.running = np.full(len(self.tracks), -1)

    def start_fn(self, idx: np.ndarray) -> Starts:
        cols = {k: np.zeros(len(idx), np.int64 if k in ("kind", "kick_steps") else np.float64) for k in STARTS_COLUMNS}
        for j, e in enumerate(idx):
            self.running[e] = self.queues[e].pop(0) if self.queues[e] else -1
            if self.running[e] >= 0:
                self.goal[e] = self.suite[self.running[e]].get("goal", SCENARIO_DISTANCE)
                for k in STARTS_COLUMNS:
                    cols[k][j] = self.suite[self.running[e]][k]
        return Starts(**cols)

    def bad_launch(self, idx: np.ndarray, attempt: int) -> Starts:
        ids = [self.suite[self.running[e]]["id"] for e in idx if self.running[e] >= 0]
        raise RuntimeError(f"suite scenarios with a non-finite launch: {ids} (make_suite.py checks launches: regenerate)")


class Suite:
    """A scenario suite with its workers, which stay up between runs: starting 48 workers takes ~12 s, so train.py
    keeps one for all its evaluations. Each run plays every scenario once from a fresh race, so runs don't depend on
    what the workers played before."""

    def __init__(self, suite: list[dict], n_workers: int = 48):
        self.suite, self.layout = suite, _Layout(suite, n_workers)
        self.lengths = {t: Track.load(t).length for t in set(self.layout.tracks)}
        self.env = RacingEnv(self.layout.tracks, episode_seconds=SCENARIO_SECONDS, max_distance=self.layout.goal,
                             start_fn=self.layout.start_fn, retry_fn=self.layout.bad_launch, overlap_starts=True)

    def run(self, policy: Policy) -> list[dict]:
        """Play every scenario once; returns one result row per scenario, in suite order."""
        suite, lay, env = self.suite, self.layout, self.env
        lay.refill()
        results: list[dict | None] = [None] * len(suite)
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
                outcome = ("recovered" if e["lost_control"][i] else
                           "survived" if e["ever_off"][i] or e["scraped"][i] else "clean") \
                    if reached else "failed"
                results[before[i]] = {"id": sc["id"], "split": sc["split"], "track": sc["track"], "tier": sc["tier"],
                                      "s": sc["s"], "outcome": outcome, "end": end,
                                      "distance": round(float(e["distance"][i]), 1),
                                      "seconds": round(float(e["seconds"][i]), 2), "bot_seconds": sc["bot_time"],
                                      "pace": round(sc["bot_time"] / e["seconds"][i], 4) if reached else "",
                                      "end_s": round(float(np.mod(sc["s"] + e["distance"][i],
                                                                  self.lengths[sc["track"]])), 1),
                                      "lost_first": LOST_CAUSES[e["lost_first"][i]] if e["lost_first"][i] >= 0 else "",
                                      **{c: int(e[f"lost_{c}"][i]) for c in LOST_CAUSES},
                                      "damage": round(float(e["damage"][i]), 1),
                                      **{k: _r(e[k][i]) for k in ("lost_at", "lost_s", "lost_speed", "last_lost_at",
                                                                  "peak", "end_speed", "end_angle", "end_offset",
                                                                  "wall_damage", "ground_damage", "ground_at",
                                                                  "ground_off")},
                                      "end_wheels_off": int(e["end_wheels_off"][i])}
        assert all(r is not None for r in results)
        return results

    def close(self) -> None:
        self.env.close()


def _r(x: float, digits: int = 2):
    """A float for the CSV: rounded, empty when nan (e.g. never lost control)."""
    return "" if not np.isfinite(x) else round(float(x), digits)


def run_suite(policy: Policy, suite: list[dict], n_workers: int = 48) -> list[dict]:
    """Play every scenario once with workers of its own; returns one result row per scenario, in suite order."""
    s = Suite(suite, n_workers)
    try:
        return s.run(policy)
    finally:
        s.close()


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
        for c in LOST_CAUSES:  # nominal scenarios that entered this loss-of-control state (absent from old CSVs)
            if nom and c in nom[0]:
                m[f"{split}/nominal/lost_{c}"] = float(np.mean([int(r[c]) > 0 for r in nom]))
        if nom and "ground_at" in nom[0]:  # nominal scenarios that scraped the ground (CSVs from 2026-11-24 on)
            m[f"{split}/nominal/scraped"] = float(np.mean([r["ground_at"] != "" for r in nom]))
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


def lost_breakdown(rows: list[dict]) -> list[dict]:
    """Per split and tier: recovered scenarios by the first cause of losing control, and the median damage of the
    scenarios that hit something."""
    out = []
    for split in sorted({r["split"] for r in rows}):
        for tier in sorted({r["tier"] for r in rows if r["split"] == split}, key=KINDS.index):
            t = [r for r in rows if r["split"] == split and r["tier"] == tier]
            rec = [r for r in t if r["outcome"] == "recovered"]
            hit = [float(r["damage"]) for r in t if int(r["contact"]) > 0 or float(r["damage"]) > 0]
            out.append({"split": split, "tier": tier, "scenarios": len(t), "recovered": len(rec),
                        **{c: sum(r["lost_first"] == c for r in rec) for c in LOST_CAUSES},
                        "with damage": len(hit), "median damage": float(np.median(hit)) if hit else float("nan")})
    return out


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


def bot_standing(track: str) -> tuple[np.ndarray, np.ndarray]:
    """The bot's race from the grid (setup_tracks.py's reference laps, standing start included): metres raced since
    its start -> seconds. Interpolate with np.interp(distance, *bot_standing(track))."""
    data = np.loadtxt(f"{_prefix(track)}_laps.csv", ndmin=2)
    d, t = data[:, TRACE_COLUMNS.index("dist_raced")], data[:, TRACE_COLUMNS.index("time")]
    step = np.diff(d)  # the engine's dist_raced can jump by a lap for a tick at the line (Spa): drop lap-sized jumps,
    length = Track.load(track).length  # as reward.py does
    step -= length * np.rint(step / length)
    return np.maximum.accumulate(np.concatenate([[0.0], np.cumsum(step)])), t - t[0]


def grid_laps(policy: Policy, tracks: list[str], laps: int = 2) -> list[dict]:
    """The real task: `laps` laps from a standing start on the grid, one episode per track (the sim and the policy are
    deterministic, so more would be copies). Ends when the laps are done, or on a failure. Per track:
    - completion: share of the laps' distance driven before the episode ended (1 = all laps; a car that recovers
      from a spin and carries on keeps counting);
    - clean: share driven before the first loss of control (the suite's test: a wall, facing backwards, far off);
    - pace: the bot's time over that clean distance, from its own standing start, / the car's time over it
      (> 1 = faster than the bot). Only the clean stretch counts, so a crash doesn't leave a hole, and the spin
      that follows doesn't count as slowness.
    Times are the sim's clock, not steps x env.dt: the engine calls the robots every 0.020 or 0.022 s (float rounding
    of its 0.002 s physics steps), so an agent step lasts 0.040-0.044 s, ~0.042 on average."""
    def grid(idx):
        z = np.zeros(len(idx))
        return Starts(np.zeros(len(idx), np.int64), z, z, z, z, z, z, np.zeros(len(idx), np.int64))
    info_t = {t: Track.load(t) for t in tracks}
    env = RacingEnv(list(tracks), episode_seconds=laps * max(t.bot_lap for t in info_t.values()) / 0.6 + 30,
                    start_fn=grid)
    n = len(tracks)
    times: list[list[float]] = [[] for _ in tracks]
    end, distance = [""] * n, np.zeros(n)
    clean_d, clean_s = np.full(n, np.nan), np.full(n, np.nan)  # at the first loss of control
    try:
        obs, _ = env.reset()
        t0, now = env._raw[:, F["time"]].astype(np.float64), env._raw[:, F["time"]].astype(np.float64)
        active = np.ones(n, bool)
        prev_laps = env._raw[:, F["laps"]].copy()
        while active.any():
            obs, _, term, trunc, info = env.step(policy(obs), active)
            raw = env._raw
            for i in np.flatnonzero(active):
                done = term[i] or trunc[i]
                if done:  # the env has already moved on to its next start: read this episode from its stats
                    e = info["episode_end"]
                    distance[i] = e["distance"][i]
                    now[i] += env.dt  # its last step (the raw state is the next start's): within 0.004 s
                    if e["lost_control"][i] and np.isnan(clean_d[i]):
                        clean_d[i], clean_s[i] = e["lost_at"][i], now[i] - t0[i]
                else:
                    now[i] = raw[i, F["time"]]
                    if env._lost[i] and np.isnan(clean_d[i]):
                        clean_d[i], clean_s[i] = env._lost_at[i], now[i] - t0[i]
                    if raw[i, F["laps"]] > prev_laps[i] and raw[i, F["laps"]] >= 2:
                        times[i].append(float(raw[i, F["last_lap_time"]]))
                    distance[i] = env.reward.raced[i]
                prev_laps[i] = raw[i, F["laps"]]
                if len(times[i]) >= laps or done:
                    active[i] = False
                    end[i] = "laps_done" if len(times[i]) >= laps else ENDS[info["episode_end"]["end"][i]]
                    if np.isnan(clean_d[i]):  # never lost control
                        clean_d[i], clean_s[i] = distance[i], now[i] - t0[i]
    finally:
        env.close()
    out = []
    for i, t in enumerate(tracks):
        goal = laps * info_t[t].length
        bot_s = float(np.interp(max(clean_d[i], 0.0), *bot_standing(t)))
        out.append({"track": t, "end": end[i], "laps": len(times[i]), "lap_times": times[i],
                    "bot_lap": info_t[t].bot_lap,
                    "best_vs_bot": (min(times[i]) / info_t[t].bot_lap) if times[i] else float("nan"),
                    "distance": float(distance[i]), "clean_distance": float(clean_d[i]),
                    "clean_seconds": float(clean_s[i]), "bot_seconds": bot_s,
                    "completion": float(min(1.0, max(distance[i], 0.0) / goal)),
                    "clean": float(min(1.0, max(clean_d[i], 0.0) / goal)),
                    "pace": bot_s / clean_s[i] if clean_s[i] > 0 else float("nan")})
    return out


def summarize_laps(rows: list[dict], split_of: dict[str, str]) -> dict[str, float]:
    """grid_laps rows -> metrics per split ("{split}/lap/..."): completion and clean averaged over tracks (each track
    weighs the same), finished = share of tracks that did every lap, pace pooled = total bot seconds / total car
    seconds over the clean stretches (long clean stretches weigh more, a crash after 200 m adds little noise)."""
    m: dict[str, float] = {}
    for split in sorted({split_of[r["track"]] for r in rows}):
        sp = [r for r in rows if split_of[r["track"]] == split]
        m[f"{split}/lap/completion"] = float(np.mean([r["completion"] for r in sp]))
        m[f"{split}/lap/clean"] = float(np.mean([r["clean"] for r in sp]))
        m[f"{split}/lap/finished"] = float(np.mean([r["end"] == "laps_done" for r in sp]))
        car = sum(r["clean_seconds"] for r in sp if r["clean_seconds"] > 0)
        m[f"{split}/lap/pace"] = sum(r["bot_seconds"] for r in sp if r["clean_seconds"] > 0) / car if car else float("nan")
    return m

