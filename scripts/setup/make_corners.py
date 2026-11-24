"""The corners of the validation and train tracks (data/corners-v2.csv, see rlpilot.corners) and scenarios aimed at
them (data/corner-suite-v2.csv), for the failure map (scripts/analysis/corner_map.py).

    python scripts/setup/make_corners.py [--force]

The suite's start points (every 200 m on validation, 600 m on train) reach corners unevenly: some get several
scenarios per tier, some none, and on train tracks a corner is in ~1-2 per tier. Here every corner gets the same: a
nominal and a mild start LEADS m before its window (the bot's braking point, or 50 m before a flat-out corner), driven
to the window's end (EXIT m after the corner), so the scenario's pace is the corner's. Test tracks have none.

Diagnostic, not a benchmark: the scenarios can be regenerated after a change to the corner finder (their ids carry the
corner). Each is drawn from its own seed, as in make_suite.py, and launches are checked the same way.
"""

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np

from rlpilot import corners
from rlpilot.client import DATA
from rlpilot.evaluate import check_launches
from rlpilot.starts import KINDS, draw
from rlpilot.tracks import Track, TrackTable, load_split

SEED = 20261124
LEADS = (150.0, 300.0)       # m before the corner's window
TIERS = ("nominal", "mild")
MAX_GOAL = 800.0             # m: a long complex is cut here (60 s is the scenario's time limit)
CORNERS = DATA / "corners-v2.csv"
SCENARIOS = DATA / "corner-suite-v2.csv"


def scenario(split: str, track: Track, table: TrackTable, c: dict, lead: float, tier: str, attempt: int) -> dict:
    rng = np.random.default_rng([SEED, zlib.crc32(track.name.encode()), int(c["apex"]), int(lead), KINDS.index(tier),
                                 attempt])
    s = float(np.mod(c["window_start"] - lead, track.length))
    st = draw(np.array([KINDS.index(tier)]), np.zeros(1, np.int64), np.array([s]), table, rng)
    goal = min(lead + c["window_length"], MAX_GOAL)
    return {"id": f"{c['id']}:{int(lead)}:{tier}", "split": split, "track": track.name, "tier": tier, "s": round(s, 1),
            "lateral": round(float(st.lateral[0]), 3), "yaw": round(float(st.yaw[0]), 4),
            "speed": round(float(st.speed[0]), 3), "kick_steer": float(st.kick_steer[0]),
            "kick_pedal": float(st.kick_pedal[0]), "kick_steps": int(st.kick_steps[0]),
            "bot_time": round(float(track.bot_time_between(np.array([s]), goal)[0]), 3), "attempt": attempt,
            "corner": c["id"], "lead": lead, "goal": round(goal, 1)}


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    a = p.parse_args()
    if SCENARIOS.exists() and not a.force:
        raise SystemExit(f"{SCENARIOS} exists (--force to regenerate)")
    split = load_split()
    table_rows, rows, where = [], [], {}
    for sp in ("validation", "train"):
        for name in split[sp]:
            track = Track.load(name)
            table = TrackTable([track])
            for c in corners.find(track):
                table_rows.append({"split": sp, **c})
                for lead in LEADS:
                    for tier in TIERS:
                        where[len(rows)] = (sp, track, table, c, lead, tier)
                        rows.append(scenario(sp, track, table, c, lead, tier, 0))
    for _ in range(5):
        bad = check_launches([dict(r, kind=KINDS.index(r["tier"])) for r in rows])
        print(f"{len(bad)} scenarios with a non-finite launch: {[rows[i]['id'] for i in bad]}")
        if not bad:
            break
        for i in bad:
            rows[i] = scenario(*where[i], rows[i]["attempt"] + 1)
    else:
        raise SystemExit("launches still failing after 5 redraws")
    write(CORNERS, table_rows)
    write(SCENARIOS, rows)
    print(f"{len(table_rows)} corners ({sum(c['braking'] for c in table_rows)} braking zones) in {CORNERS}, "
          f"{len(rows)} scenarios in {SCENARIOS}")


if __name__ == "__main__":
    main()
