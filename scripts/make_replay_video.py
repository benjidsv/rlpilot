"""Put a synced map of the replayed lap next to a screen recording of it (scripts/record_lap.py).

    python scripts/make_replay_video.py lap.mov --export

The recording must be cut to the lap: first frame on the line, last frame still on the lap. Writes replay/merged.mp4
(the full lap, under 10 MB for GitHub) and docs/img/lap_replay_clip.gif (the --start/--length clip).
"""

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

from make_readme_figures import DATA, IMG, LINE_SMOOTH_M, ROAD_FACTOR, ROOT, WideRoad

REPLAY = ROOT / "replay"
LAP_S = 108.27  # lap time of the recording in replay/, as shown by the game
END_S = 108.25  # where the video is cut: its last frame (108.258 s) already brakes, past the end of the recording
# The game's picture inside the macOS window recording is 2032x1184 at (120, 146). Keep its middle 50% and drop its top
# 15%: that leaves the car and the input widget, and removes the rest of the HUD (gear, speed and rpm move to the map).
CROP = "1016:1006:628:324"
MAP_W, CAM_W, HEIGHT = 640, 566, 560
FPS = 30
ARC_START, ARC_END = 225, -45  # degrees, like the game's rev counter
AGENT, BOT, INK, MUTED, SURFACE, ROAD = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
KEEP_COLORS = (AGENT, BOT, "#00ff00", "#ff0000")  # the map's cars, the HUD's throttle and brake bars


def export_lap() -> None:
    """The recorded lap and the bot's lap 2, with the track borders."""
    from rlpilot import reference

    ref = reference.load("rlref")
    bot = ref.lap_trace(2)
    c = np.fromfile(REPLAY / "jarama.controls", dtype=[(k, "f4") for k in ("steer", "accel", "brake", "clutch")]
                    + [("gear", "i4"), ("x", "f4"), ("y", "f4")])
    with open(REPLAY / "jarama.controls.check.csv") as f:  # written by the replay: time, x, y, x, y, speed, gear, rpm, redline
        check = np.array([[float(v) for v in row] for row in csv.reader(f)])
    t = check[:, 0]
    lap = t >= t[-1] - LAP_S
    xy = np.column_stack([c["x"][lap], c["y"][lap]])
    time = t[lap] - t[lap][0]
    np.savez_compressed(DATA / "replay_lap.npz", left=ref.track_left.astype(np.float32),
                        right=ref.track_right.astype(np.float32), agent_lap_s=LAP_S, bot_lap_s=ref.lap_times[1],
                        agent_time=time.astype(np.float32), agent_x=xy[:, 0], agent_y=xy[:, 1],
                        agent_speed=check[lap, 5].astype(np.float32), agent_gear=check[lap, 6].astype(np.int8),
                        agent_rpm_fraction=(check[lap, 7] / check[lap, 8]).astype(np.float32),
                        agent_rpm=(check[lap, 7] * 60 / (2 * np.pi)).astype(np.float32),
                        bot_time=(bot["time"] - bot["time"][0]).astype(np.float32), bot_x=bot["x"].astype(np.float32),
                        bot_y=bot["y"].astype(np.float32), bot_speed=bot["speed_x"].astype(np.float32))


def map_frames():
    """Yields one RGB frame of the map per video frame, in real time."""
    d = np.load(DATA / "replay_lap.npz")
    road = WideRoad(d["left"].astype(float), d["right"].astype(float), ROAD_FACTOR, LINE_SMOOTH_M)
    cars = {"agent": (AGENT, "agent"), "bot": (BOT, "built-in bot")}
    t = {n: d[f"{n}_time"].astype(float) for n in cars}
    xy = {n: road.line(np.column_stack([d[f"{n}_x"], d[f"{n}_y"]]).astype(float)) for n in cars}

    fig = Figure(figsize=(MAP_W / 100, HEIGHT / 100), dpi=100, facecolor=SURFACE)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.02, 0.1, 0.96, 0.56])
    ax.set_aspect("equal")
    ax.axis("off")
    n = len(road.left)
    quads = [[road.left[i], road.left[(i + 1) % n], road.right[(i + 1) % n], road.right[i]] for i in range(n)]
    ax.add_collection(PolyCollection(quads, facecolor=ROAD, edgecolor=ROAD, linewidth=0.5))
    ax.update_datalim(np.vstack([road.left, road.right]))
    ax.autoscale_view()
    fig.text(0.03, 0.9, "Flying lap on Jarama, real time\nBest checkpoint of proto-6 vs built-in bot",
             fontweight="bold", fontsize=15, color=INK, linespacing=1.35)
    fig.text(0.03, 0.025, f"Road and lines up to {ROAD_FACTOR:.0f}x wider, lines smoothed over {LINE_SMOOTH_M:.0f} m",
             fontsize=11, color=MUTED)
    trails, dots, labels = {}, {}, {}
    for i, (name, (color, title)) in enumerate(cars.items()):
        trails[name], = ax.plot([], [], color=color, lw=2.2, solid_capstyle="round", zorder=3 - i, animated=True)
        dots[name], = ax.plot([], [], "o", color=color, ms=10, mec=SURFACE, mew=1.5, zorder=5 - i, animated=True)
        x = 0.03 + 0.33 * i
        fig.add_artist(Line2D([x + 0.012], [0.8], marker="o", ms=12, color=color, mec=SURFACE, mew=1,
                              transform=fig.transFigure))
        fig.text(x + 0.035, 0.785, title, fontsize=14, color=INK)
        labels[name] = fig.text(x + 0.035, 0.73, "", fontsize=13, color=MUTED, family="DejaVu Sans Mono", animated=True)
    # Gear, speed and rpm of the agent, bottom right: the arc fills up to the redline.
    gauge = fig.add_axes([0.64, 0.075, 0.33, 0.33 * MAP_W / HEIGHT], facecolor="none")
    gauge.set_aspect("equal")
    gauge.axis("off")
    gauge.set_xlim(-1.05, 1.05)
    gauge.set_ylim(-1.05, 1.05)
    gauge.add_patch(Wedge((0, 0), 1, ARC_END, ARC_START, width=0.13, facecolor=ROAD, edgecolor="none"))
    rpm_arc = Wedge((0, 0), 1, ARC_START, ARC_START, width=0.13, facecolor=AGENT, edgecolor="none", animated=True)
    gauge.add_patch(rpm_arc)
    gear_text = gauge.text(0, 0.22, "", ha="center", va="center", fontsize=30, fontweight="bold", color=INK, animated=True)
    speed_text = gauge.text(0, -0.28, "", ha="center", va="center", fontsize=17, color=INK, animated=True)
    gauge.text(0, -0.56, "km/h", ha="center", va="center", fontsize=10, color=MUTED)
    rpm_text = gauge.text(0, -0.84, "", ha="center", va="center", fontsize=10, color=MUTED, animated=True)
    clock = fig.text(0.97, 0.76, "", ha="right", fontsize=13, color=MUTED, family="DejaVu Sans Mono", animated=True)
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)

    for now in np.arange(0, END_S, 1 / FPS):
        canvas.restore_region(background)
        for name in cars:
            when = min(now, t[name][-1])
            keep = (t[name] >= when - 8) & (t[name] <= when)
            trails[name].set_data(xy[name][keep, 0], xy[name][keep, 1])
            dots[name].set_data([np.interp(when, t[name], xy[name][:, 0])], [np.interp(when, t[name], xy[name][:, 1])])
            done = now >= t[name][-1]
            labels[name].set_text(f"{float(d[f'{name}_lap_s']):.2f} s" if done
                                  else f"{np.interp(when, t[name], d[f'{name}_speed']) * 3.6:3.0f} km/h")
            ax.draw_artist(trails[name])
            ax.draw_artist(dots[name])
            fig.draw_artist(labels[name])
        clock.set_text(f"t = {now:6.2f} s")
        fig.draw_artist(clock)
        when = min(now, t["agent"][-1])
        i = min(np.searchsorted(t["agent"], when), len(t["agent"]) - 1)
        fraction = min(float(d["agent_rpm_fraction"][i]), 1.0)
        rpm_arc.set_theta2(ARC_START)
        rpm_arc.set_theta1(ARC_START - fraction * (ARC_START - ARC_END))
        gear_text.set_text(str(int(d["agent_gear"][i])))
        speed_text.set_text(f"{float(d['agent_speed'][i]) * 3.6:.0f}")
        rpm_text.set_text(f"{float(d['agent_rpm'][i]):.0f} rpm")
        for artist in (rpm_arc, gear_text, speed_text, rpm_text):
            gauge.draw_artist(artist)
        canvas.blit(fig.bbox)
        yield np.asarray(canvas.buffer_rgba())[..., :3].tobytes()


def master(recording: Path, out: Path) -> None:
    """Map on the left, game on the right, near-lossless: the small files are encoded from this."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{MAP_W}x{HEIGHT}", "-r", str(FPS),
           "-i", "-", "-i", str(recording), "-filter_complex",
           f"[1:v]trim=end={END_S},setpts=PTS-STARTPTS,fps={FPS},crop={CROP},scale={CAM_W}:{HEIGHT}:flags=lanczos[cam];"
           "[0:v][cam]hstack=shortest=1,format=yuv420p[v]",
           "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "14", "-preset", "fast", str(out)]
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in map_frames():
        ffmpeg.stdin.write(frame)
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError("ffmpeg failed")


def full_lap_mp4(source: Path, out: Path, target_mb: float = 9.5, width: int = 1200) -> None:
    kbps = int(target_mb * 8000 / END_S)
    base = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vf", f"scale={width}:-2:flags=lanczos", "-an",
            "-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    log = str(REPLAY / "x264-pass")
    subprocess.run(base + ["-pass", "1", "-passlogfile", log, "-f", "null", "/dev/null"], check=True)
    subprocess.run(base + ["-pass", "2", "-passlogfile", log, str(out)], check=True)


def clip_gif(source: Path, out: Path, start: float, length: float, width: int, fps: int, colors: int) -> None:
    """Few colours, no dithering: scenery can wash out, but the inputs and the cars keep exact palette slots."""
    from PIL import Image

    clip = ["-ss", str(start), "-t", str(length), "-i", str(source)]
    scale = f"fps={fps},scale={width}:-2:flags=lanczos"
    palette = REPLAY / "palette.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *clip, "-vf",
                    f"{scale},palettegen=max_colors={colors - len(KEEP_COLORS)}:stats_mode=full:reserve_transparent=0",
                    str(palette)], check=True)
    pixels = np.array(Image.open(palette).convert("RGB")).reshape(-1, 3)
    pixels[-len(KEEP_COLORS):] = [list(bytes.fromhex(c[1:])) for c in KEEP_COLORS]  # palettegen fills the first slots
    Image.fromarray(pixels.reshape(16, 16, 3)).save(palette)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *clip, "-i", str(palette), "-filter_complex",
                    f"[0:v]{scale}[v];[v][1:v]paletteuse=dither=none:diff_mode=rectangle", "-loop", "0", str(out)],
                   check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("recording", nargs="?", type=Path, default=ROOT / "lap.mov")
    p.add_argument("--start", type=float, default=42.0)
    p.add_argument("--length", type=float, default=40.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--colors", type=int, default=32)
    p.add_argument("--export", action="store_true", help="re-export docs/data/replay_lap.npz from replay/")
    p.add_argument("--gif-only", action="store_true", help="reuse replay/merged-master.mp4")
    a = p.parse_args()
    source = REPLAY / "merged-master.mp4"
    if not a.gif_only:
        if a.export or not (DATA / "replay_lap.npz").exists():
            export_lap()
        master(a.recording, source)
        full_lap_mp4(source, REPLAY / "merged.mp4")
    clip_gif(source, IMG / "lap_replay_clip.gif", a.start, a.length, a.width, a.fps, a.colors)
    for f in (source, REPLAY / "merged.mp4", IMG / "lap_replay_clip.gif"):
        if f.exists():
            print(f"{f.relative_to(ROOT)}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
