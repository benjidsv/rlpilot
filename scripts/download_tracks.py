"""Download tracks from the Speed Dreams asset index into sd-local/tracks, like the in-game download manager
(same index, sha256 check, layout tracks/<category>/<directory>/ with a .revision file).

    python scripts/download_tracks.py --categories circuit road [--dry-run]

One track at a time (download, verify, extract, delete the archive), so the extra disk space is one archive.
Tracks already installed at the same revision are skipped.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "sd-local" / "tracks"
INDEX = "https://www.speed-dreams.net/assets.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", default=["circuit", "road"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    index = json.loads(subprocess.run(["curl", "-sSfL", "--max-time", "60", INDEX], capture_output=True,
                                      check=True).stdout)
    todo = [t for t in index["tracks"] if t["category"] in a.categories]
    for t in todo:
        dst = TRACKS / t["category"] / t["directory"]
        rev = dst / ".revision"
        if rev.exists() and rev.read_text().strip() == str(t["revision"]):
            print(f"skip  {t['category']}/{t['directory']} (installed)")
            continue
        free = shutil.disk_usage(TRACKS).free
        if free < 3 * int(t["size"]) + (1 << 30):
            sys.exit(f"stopping: only {free / 1e9:.1f} GB free")
        print(f"get   {t['category']}/{t['directory']} ({int(t['size']) / 1e6:.0f} MB)", flush=True)
        if a.dry_run:
            continue
        with tempfile.TemporaryDirectory(dir=TRACKS) as tmp:
            zpath = Path(tmp) / "track.zip"
            subprocess.run(["curl", "-sSfL", "--max-time", "900", "-o", str(zpath), t["url"]], check=True)
            if t["hashtype"] != "sha256" or sha256(zpath) != t["hash"]:
                print(f"  hash mismatch, skipped")
                continue
            with zipfile.ZipFile(zpath) as z:
                names = z.namelist()
                if any(n.startswith("/") or ".." in Path(n).parts for n in names):
                    print("  unsafe paths in archive, skipped")
                    continue
                z.extractall(Path(tmp) / "x")
            zpath.unlink()
            src = Path(tmp) / "x" / t["directory"]
            if not src.is_dir():
                print(f"  archive has no {t['directory']}/ directory, skipped")
                continue
            (src / ".revision").write_text(f"{t['revision']}\n")
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)


if __name__ == "__main__":
    main()
