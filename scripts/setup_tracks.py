"""Create race configs for every installed track and run the built-in bot reference on each.

    python scripts/setup_tracks.py [--categories circuit road gprix] [--jobs 5]

For track t in sd-local/tracks/<category>/<t>/: data/raceman/rlpilot-<t>.xml (the agent) and rlref-<t>.xml (the
bot), copies of rlpilot.xml / rlref.xml with only the track changed. Then reference.load() runs the bot for
3 laps (cached in reference/). Tracks where the bot fails to complete 3 laps are reported, and their race
configs removed, since without a reference there is no lookahead and no lap-time yardstick.
Writes reference/tracks.csv: track, category, length, bot best lap.
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rlpilot import reference, workdirs
from rlpilot.track import track_of

RACEMAN = workdirs.DATA / "raceman"


def race_xml(template: str, track: str, category: str) -> str:
    text = (RACEMAN / f"{template}.xml").read_text()
    sec = re.search(r'(<section name="Tracks">.*?</section>\s*</section>)', text, re.S).group(1)
    new = re.sub(r'(name="name" val=")[^"]*', rf"\g<1>{track}", sec, count=1)
    new = re.sub(r'(name="category" val=")[^"]*', rf"\g<1>{category}", new, count=1)
    return text.replace(sec, new)


def setup(track: str, category: str) -> tuple[str, str, float | None, float | None, str]:
    if track == track_of("rlpilot"):
        race, ref = "rlpilot", "rlref"
    else:
        race, ref = f"rlpilot-{track}", f"rlref-{track}"
        (RACEMAN / f"{race}.xml").write_text(race_xml("rlpilot", track, category))
        (RACEMAN / f"{ref}.xml").write_text(race_xml("rlref", track, category))
    try:
        r = reference.load(ref, laps=3)
        if len(r.lap_times) < 3:
            raise RuntimeError(f"bot completed {len(r.lap_times)} of 3 laps")
        return track, category, r.track_length, float(r.lap_times[1:].min()), "ok"
    except Exception as e:  # noqa: BLE001 - report and drop the track
        if race != "rlpilot":
            for name in (race, ref):
                (RACEMAN / f"{name}.xml").unlink(missing_ok=True)
        return track, category, None, None, str(e).splitlines()[0][:120]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", default=["circuit", "road", "gprix"])
    p.add_argument("--jobs", type=int, default=5)
    a = p.parse_args()
    tracks = [(d.name, c) for c in a.categories for d in sorted((workdirs.TEMPLATE / "tracks" / c).glob("*"))
              if d.is_dir() and (d / f"{d.name}.xml").exists()]
    tracks.append((track_of("rlpilot"), "circuit"))  # Jarama is in the install dir
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda tc: setup(*tc), tracks))
    ok = [r for r in results if r[4] == "ok"]
    with open(reference.CACHE / "tracks.csv", "w") as f:
        f.write("track,category,length_m,bot_best_lap_s\n")
        for t, c, length, lap, _ in sorted(ok):
            f.write(f"{t},{c},{length:.0f},{lap:.2f}\n")
    for t, c, length, lap, status in sorted(results, key=lambda r: (r[4] != "ok", r[1], r[0])):
        print(f"{c:8s} {t:15s} " + (f"{length:6.0f} m  bot {lap:6.2f} s" if status == "ok" else f"FAILED: {status}"))
    print(f"{len(ok)}/{len(results)} tracks usable")


if __name__ == "__main__":
    main()
