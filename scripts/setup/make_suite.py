"""Generate the evaluation scenario suite once and write data/suite-v2.csv. Commit it; don't regenerate it after any
result has been compared with it (scenario ids would change meaning).

    python scripts/setup/make_suite.py [--force]

Start points along each track, every SPACING[split] m from 100 m after the start line, and one scenario per tier at
each (see rlpilot.starts): nominal, mild, perturbed, disturbed, crashed. Each scenario is drawn from its own seed (suite
seed, track, point, tier), so it doesn't depend on any other. Validation and test tracks are sampled densely; training
tracks more sparsely, to keep an eval short (they only show the train / unseen gap).
"""

import argparse
import csv
import zlib

import numpy as np

from rlpilot.evaluate import SCENARIO_DISTANCE, SUITE, check_launches
from rlpilot.starts import KINDS, SUITE_TIERS, draw
from rlpilot.tracks import Track, TrackTable, load_split

SEED = 20261122
SPACING = {"train": 600.0, "validation": 200.0, "test": 200.0}  # m between start points


def scenario(split: str, name: str, table: TrackTable, j: int, s: float, tier: str, bot_time: float, attempt: int) -> dict:
    """Scenario at start point j of a track, drawn from its own seed (attempt > 0: redrawn after a bad launch)."""
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode()), j, KINDS.index(tier), attempt])
    st = draw(np.array([KINDS.index(tier)]), np.zeros(1, np.int64), np.array([s]), table, rng)
    return {"id": f"{name}:{int(s)}:{tier}", "split": split, "track": name, "tier": tier, "s": round(s, 1),
            "lateral": round(float(st.lateral[0]), 3), "yaw": round(float(st.yaw[0]), 4),
            "speed": round(float(st.speed[0]), 3), "kick_steer": float(st.kick_steer[0]),
            "kick_pedal": float(st.kick_pedal[0]), "kick_steps": int(st.kick_steps[0]), "bot_time": round(bot_time, 3),
            "attempt": attempt}


def load_rows(rows: list[dict]) -> list[dict]:
    """Rows as load_suite gives them (typed, with the kind index)."""
    return [dict(r, kind=KINDS.index(r["tier"])) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="overwrite an existing suite (only before any v2 result)")
    a = p.parse_args()
    if SUITE.exists() and not a.force:
        raise SystemExit(f"{SUITE} exists: the suite is frozen")
    rows = []
    for split, tracks in load_split().items():
        for name in tracks:
            track = Track.load(name)
            table = TrackTable([track])
            points = np.arange(100.0, track.length - 1.0, SPACING[split])
            bot = track.bot_time_between(points, SCENARIO_DISTANCE)
            for j, s in enumerate(points):
                for tier in SUITE_TIERS:
                    rows.append(scenario(split, name, table, j, float(s), tier, float(bot[j]), attempt=0))
    for _ in range(5):  # a few launches come back non-finite from the engine: redraw those with the next seed
        bad = check_launches(load_rows(rows))
        print(f"{len(bad)} scenarios with a non-finite launch: {[rows[i]['id'] for i in bad]}")
        if not bad:
            break
        for i in bad:
            r = rows[i]
            j = int(np.flatnonzero(np.arange(100.0, Track.load(r["track"]).length - 1.0, SPACING[r["split"]]) == r["s"])[0])
            rows[i] = scenario(r["split"], r["track"], TrackTable([Track.load(r["track"])]), j, r["s"], r["tier"],
                               r["bot_time"], r["attempt"] + 1)
    else:
        raise SystemExit("launches still failing after 5 redraws")
    with open(SUITE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for split in SPACING:
        print(f"{split:10s} {sum(r['split'] == split for r in rows):5d} scenarios")


if __name__ == "__main__":
    main()
