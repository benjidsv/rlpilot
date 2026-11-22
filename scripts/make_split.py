"""Draw the v2 track split once and write data/split-v2.json. Don't redraw: the test tracks must stay unseen.

    python scripts/make_split.py [--force]

Test: the Nordschleife (21 km, very varied; driving it never seen is the headline goal) + one short circuit + one short
road track, so that the test isn't decided by one track's character. Validation (selects checkpoints, guides the work):
three medium circuits + three medium road tracks. Train: everything else, including Spring and Longjour.
Short = under 3.6 km, medium = 3.6 to 7.5 km; categories from reference/tracks.csv (circuit, road, gprix).
"""

import argparse
import csv
import json
import random

from rlpilot.tracks import CACHE, SPLIT_FILE, all_tracks

SEED = 20261122


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="overwrite an existing split (only before any v2 result)")
    a = p.parse_args()
    if SPLIT_FILE.exists() and not a.force:
        raise SystemExit(f"{SPLIT_FILE} exists: the split is frozen")
    rows = {r["track"]: r for r in csv.DictReader(open(CACHE / "tracks.csv"))}
    tracks = all_tracks()
    assert set(tracks) <= set(rows), set(tracks) - set(rows)
    length = {t: float(rows[t]["length_m"]) for t in tracks}
    cat = {t: rows[t]["category"] for t in tracks}
    rng = random.Random(SEED)
    pool = lambda c, lo, hi: sorted(t for t in tracks if cat[t] == c and lo <= length[t] < hi and t != "nordschleife")
    test = ["nordschleife"] + rng.sample(pool("circuit", 0, 3600), 1) + rng.sample(pool("road", 0, 3600), 1)
    validation = rng.sample(pool("circuit", 3600, 7500), 3) + rng.sample(pool("road", 3600, 7500), 3)
    train = sorted(set(tracks) - set(test) - set(validation))
    split = {"train": train, "validation": sorted(validation), "test": sorted(test)}
    km = {k: round(sum(length[t] for t in v) / 1000, 1) for k, v in split.items()}
    SPLIT_FILE.write_text(json.dumps({"seed": SEED, "km": km, "tracks": split}, indent=2) + "\n")
    for k, v in split.items():
        print(f"{k:10s} {len(v):2d} tracks, {km[k]:6.1f} km: {', '.join(v)}")


if __name__ == "__main__":
    main()
