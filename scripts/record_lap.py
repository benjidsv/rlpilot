"""Record a checkpoint driving, so that the rendered game can replay it (the policy only runs in the headless worker).

    python scripts/record_lap.py mlruns/1/<run>/artifacts/best --track jarama --laps 2
    python scripts/record_lap.py --prepare-only --track jarama   # new take: the engine rewrites the race file

The worker saves its exact car controls per robot tick. The script then prepares a settings dir with a rendered
race, with the driver-input HUD on, and prints the command that replays the recording in the normal game. The defaults match proto-6's observations.
"""

import argparse
import os
import re
from pathlib import Path

from PIL import Image
from stable_baselines3 import PPO

from rlpilot import EnvConfig, SpeedDreamsVecEnv, workdirs
from rlpilot.evaluate import rollout
from rlpilot.reference import lap_times_from_trace
from rlpilot.track import race_for

OUT = workdirs.ROOT / "replay"
INPUTS_SCALE = 3.5  # the driver-input widget is made for a game HUD: too small for a video
INPUTS_RIGHT_MARGIN = -620  # its distance from the right of the screen (HUD units)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", help="directory with model.zip")
    p.add_argument("--prepare-only", action="store_true", help="keep the recording, only reset the settings dir")
    p.add_argument("--track", default="jarama")
    p.add_argument("--laps", type=int, default=2)
    a = p.parse_args()
    if not a.prepare_only and not a.checkpoint:
        p.error("a checkpoint is required unless --prepare-only is set")

    race = race_for(a.track)
    controls = OUT / f"{a.track}.controls"
    OUT.mkdir(exist_ok=True)
    if not a.prepare_only:
        record(a.checkpoint, race, a.laps, controls)

    lc = workdirs.prepare(OUT / "lc", 1, race)[0]
    race_xml = lc / "config/raceman" / f"{race}.xml"
    race_xml.write_text(race_xml.read_text().replace('val="results only"', 'val="normal"'))
    show_inputs(lc / "config/osghudconfig.xml")
    print(f"RLP_REPLAY={controls} ./sd-install/games/speed-dreams-2 -lc {lc}/ -s {race} -nr")


def show_inputs(hud: Path) -> None:
    """Turn on the HUD's driver inputs, INPUTS_SCALE times bigger, in this replay's own config. The HUD sizes an element
    from its PNG times one global scale, so the widget gets upscaled PNGs and proportional offsets."""
    source = workdirs.INSTALL_RACEMAN.parents[1] / "data/img/osg-hud"
    bigger = source / "rlreplay"
    bigger.mkdir(exist_ok=True)
    for png in source.glob("driverinput-*.png"):
        image = Image.open(png)
        image.resize([round(n * INPUTS_SCALE) for n in image.size], Image.Resampling.LANCZOS).save(bigger / png.name)

    def element(m: re.Match) -> str:
        text = m.group(0).replace("data/img/osg-hud/", "data/img/osg-hud/rlreplay/")
        if m.group(1) == "driverinput-container":  # anchored to the screen: only move it, clear of the other widgets
            return re.sub(r'(position-horizontalModifier" val=")[-\d.]+', rf"\g<1>{INPUTS_RIGHT_MARGIN}", text)
        return re.sub(r'(position-(?:horizontal|vertical)Modifier" val=")([-\d.]+)',
                      lambda v: f"{v.group(1)}{float(v.group(2)) * INPUTS_SCALE:g}", text)

    text = hud.read_text()
    text, shown = re.subn(r'(<section name="driverinputWidget">\s*<attnum name="enabled" val=")0', r"\g<1>1", text)
    if not shown:
        raise RuntimeError(f"driverinputWidget not found in {hud}")
    text = re.sub(r'<section name="(driverinput-[\w-]+)">.*?</section>', element, text, flags=re.S)
    hud.write_text(text)


def record(checkpoint: str, race: str, laps: int, controls: Path) -> None:
    os.environ["RLP_RECORD_CONTROLS"] = str(controls)  # inherited by the worker

    model = PPO.load(Path(checkpoint) / "model.zip", device="cpu")
    cfg = EnvConfig(n_workers=1, race=race, gearbox="points", lookahead=True, lookahead_stat="peak", line_obs=True,
                    car_obs=True, laps_per_episode=laps, spin_us=50, recycle_resets=2**62, track_switch_steps=2**62)
    env = SpeedDreamsVecEnv(cfg)
    try:
        traces, _, stats = rollout(lambda o: model.predict(o, deterministic=True)[0], env)
    finally:
        env.close()
    print(f"episode ended by {stats[0]['end']}, lap times (s): {[round(float(t), 2) for t in lap_times_from_trace(traces[0])]}")


if __name__ == "__main__":
    main()
