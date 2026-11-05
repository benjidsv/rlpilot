"""Per-worker Speed Dreams local dirs (`-lc`).

The engine writes results and regenerated driver lists into its local dir on every (re)start, so
parallel workers must not share one. Each worker gets small copies of `config/` and `drivers/` from
the initialized template (sd-local) and symlinks to the heavy `cars/` and `tracks/` dirs. Race and
driver configs are re-copied from data/ every time because the engine rewrites them when loading.

    python -m rlpilot.workdirs <run_dir> <n_workers>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "sd-local"
DATA = ROOT / "data"
INSTALL_RACEMAN = ROOT / "sd-install/share/games/speed-dreams-2/config/raceman"
WORKER_BIN = ROOT / "sd-install/games/rlpilot-worker"
CLIENT_LIB = ROOT / f"sd-install/lib/librlpilot_client.{'dylib' if sys.platform == 'darwin' else 'so'}"


def prepare(run_dir: Path, n_workers: int, race: str = "rlpilot", only: int | None = None) -> list[Path]:
    """Create run_dir/w<i>/ for every worker (or only worker `only`); returns the dirs created."""
    if not (TEMPLATE / "config").is_dir():
        raise FileNotFoundError(f"{TEMPLATE} is not initialized: run the game once with -lc {TEMPLATE}/")
    race_xml = DATA / "raceman" / f"{race}.xml"
    # The race list is enumerated from the install dir, then loaded from the local dir.
    INSTALL_RACEMAN.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(race_xml, INSTALL_RACEMAN / race_xml.name)

    dirs = []
    for i in range(n_workers) if only is None else [only]:
        d = Path(run_dir) / f"w{i}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        shutil.copytree(TEMPLATE / "config", d / "config")
        shutil.copytree(TEMPLATE / "drivers", d / "drivers", ignore=shutil.ignore_patterns("rlpilot"))
        for heavy in ("cars", "tracks"):
            (d / heavy).symlink_to(TEMPLATE / heavy, target_is_directory=True)
        shutil.copytree(DATA / "drivers" / "rlpilot", d / "drivers" / "rlpilot")
        shutil.copyfile(race_xml, d / "config" / "raceman" / race_xml.name)
        dirs.append(d)
    return dirs


if __name__ == "__main__":
    for p in prepare(Path(sys.argv[1]), int(sys.argv[2])):
        print(p)
