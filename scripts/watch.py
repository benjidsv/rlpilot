"""Watch a checkpoint drive, in the rendered game, next to live telemetry against the built-in bot.

    uv run python scripts/watch.py sw10m-fix-learnedstd torino             # the run's last checkpoint, 2 laps
    uv run python scripts/watch.py runs/<run> torino --best --laps 3       # its best/ checkpoint instead
    uv run python scripts/watch.py runs/<run>/checkpoints/0004000000.pt torino --no-game   # lap times + summary only
    uv run python scripts/watch.py sw10m-colored torino --shift hazard     # the gear choice read another way (eval.Driver)
    uv run python scripts/watch.py sw10m-colored torino --from-start       # lap 1 in real time (default: fast-forward)
    uv run python scripts/watch.py sw10m-colored --scenario spring:1200:crashed   # one scenario of the eval suite
    uv run python scripts/watch.py sw10m-colored --failed crashed          # list the run's failed crashed scenarios

The run is a run dir, a checkpoint file, or an experiment name (runs/<name>__<seed>__*, --seed when there are several).
The policy only runs in the headless worker, so the laps are driven there first (a few seconds). The worker records the
car's controls on every engine tick, and watch.py records, at each policy step, the car's state, the controls, the
critic's value and the policy's std. Then:
- a summary image (runs/<run>/watch/<track>-<checkpoint>.png): the map coloured by time gained or lost against the bot,
  speed and delta against the bot per lap;
- the game replays the recorded controls in a window (the engine is deterministic: it's the same drive), with the
  driver inputs on the HUD, and the HUD's delta and best lap set to the bot's fastest lap;
- a telemetry window follows the game (it reads the replay's race time as the game writes it, so it follows pauses):
  the map, a zoomed map with the bot's line and ghost (the bot at the same lap time), the delta to the bot, and the last
  300 m of speed, pedals, steering, delta, value and std against the bot's at the same place on the track.
A scenario (--scenario, an id of data/suite-v2.csv as in the eval's CSVs) starts where the suite launches it and drives
its distance (500 m); the game launches the replayed car there too (RLP_REPLAY_LAUNCH, robot/rlpilot.cpp), and the
delta and ghost count from the scenario's start. After the drive the car stays braked; quit the game to finish. Closing the game prints how far the replay strayed from
the recording (it should be 0). Test tracks need --test.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure

from rlpilot.client import A, F, ROOT, prepare_dir
from rlpilot.env import ENDS, RacingEnv
from rlpilot.evaluate import SCENARIO_SECONDS, load_suite
from rlpilot.starts import Starts, write_launch
from rlpilot.tracks import TRACE_COLUMNS, Track, TrackTable, _prefix, lap_times, load_split, race_for

sys.path.insert(0, str(Path(__file__).parent))
from eval import SHIFT_MODES, Driver, load_agent  # noqa: E402

GAME = ROOT / "sd-install/games/speed-dreams-2"
AGENT, BOT, ROAD, MUTED = "#2a78d6", "#eb6834", "#d9d8d2", "#8a8984"
BEHIND, AHEAD = 300.0, 100.0  # the telemetry window: metres behind the car, and of the bot's ahead of it
# Per policy step: the car's state when the policy decided (raw fields), then the controls it applied, then the policy
STATE = ("time", "cur_lap_time", "laps", "dist_from_start", "dist_raced", "speed_x", "pos_x", "pos_y", "gear", "damage")
COLUMNS = STATE + ("steer", "accel", "brake", "value", "std_steer", "std_pedal")


# ---- the checkpoint, the drive ----

def find_checkpoint(arg: str, seed: int | None, best: bool) -> Path:
    """A checkpoint file as is; a run dir's best.pt or last numbered checkpoint; an experiment name's run."""
    p = Path(arg)
    if p.is_file():
        return p
    if not p.is_dir():
        runs = sorted(glob.glob(f"runs/{arg}__{seed if seed is not None else '*'}__*"))
        if len(runs) != 1:
            found = "\n  ".join(runs) or "none"
            raise SystemExit(f"{arg!r}: not a file or a run dir, and {len(runs)} runs match runs/{arg}__*: pass a run dir"
                             f" or --seed\n  {found}")
        p = Path(runs[0])
    if best:
        return p / "checkpoints" / "best.pt"
    numbered = sorted(c for c in (p / "checkpoints").glob("*.pt") if c.stem.isdigit())
    if not numbered:
        raise SystemExit(f"no checkpoint in {p / 'checkpoints'} yet")
    return numbered[-1]


def scenario_start(sc: dict) -> Starts:
    return Starts(np.array([sc["kind"]]), *(np.array([sc[k]], np.float64) for k in ("s", "lateral", "yaw", "speed",
                  "kick_steer", "kick_pedal")), np.array([sc["kick_steps"]], np.int64))


def launch_spec(sc: dict) -> str:
    """RLP_REPLAY_LAUNCH for the game: the launch slots the worker got (starts.write_launch)."""
    act = np.zeros((1, A["launch_speed"] + 1), np.float32)
    write_launch(scenario_start(sc), np.zeros(1, np.int64), TrackTable([Track.load(sc["track"])]), act)
    return ",".join(f"{v:.9g}" for v in act[0, A["launch_pos"]:A["launch_speed"] + 1])


def drive(checkpoint: Path, track: str, laps: int, controls: Path, shift: str = "argmax", seed: int = 0,
          scenario: dict | None = None) -> dict:
    """`laps` laps from the grid (or the suite `scenario`: its start, then its distance) in a headless worker that
    writes its controls, tick by tick, to `controls`; returns the drive per policy step (COLUMNS) and how it went. As
    evaluate.grid_laps (or the suite), plus the policy's outputs. `shift`: how the gear choice is read from the policy
    (eval.Driver); `seed` for shift = sample."""
    agent, normalize, ckpt = load_agent(checkpoint)
    driver = Driver(agent, shift, seed)
    bot_lap = Track.load(track).bot_lap
    what = f"scenario {scenario['id']}" if scenario else f"{laps} laps from the grid"
    print(f"{checkpoint} (step {ckpt['step']:,}) on {track}, {what}, shift: {shift}")

    def grid(idx):
        z = np.zeros(len(idx))
        return Starts(np.zeros(len(idx), np.int64), z, z, z, z, z, z, np.zeros(len(idx), np.int64))

    os.environ["RLP_RECORD_CONTROLS"] = str(controls)  # the worker inherits it; it records its first episode
    if scenario:
        env = RacingEnv([track], episode_seconds=SCENARIO_SECONDS, max_distance=scenario["goal"],
                        start_fn=lambda idx: scenario_start(scenario))
        laps = 10**6  # only the scenario's end stops it
    else:
        env = RacingEnv([track], episode_seconds=laps * bot_lap / 0.6 + 30, start_fn=grid)
    del os.environ["RLP_RECORD_CONTROLS"]
    rows, times = [], []
    try:
        obs, _ = env.reset()
        prev_laps = env._raw[0, F["laps"]]
        while True:
            state = [env._raw[0, F[k]] for k in STATE]
            x = torch.as_tensor(normalize(obs))
            with torch.no_grad():
                _, _, _, value = agent.get_action_and_value(x)  # sets agent.last_std
            std = agent.last_std[0].numpy()
            action = driver(x, env.controls.gear)
            obs, _, term, trunc, info = env.step(action)
            act = env._act[0]
            rows.append(state + [act[A["steer"]], act[A["accel"]], act[A["brake"]], float(value[0, 0]), *std])
            done = bool(term[0] or trunc[0])
            raw = env._raw[0]
            if not done and raw[F["laps"]] > prev_laps and raw[F["laps"]] >= 2:
                times.append(float(raw[F["last_lap_time"]]))
            prev_laps = raw[F["laps"]]
            if len(times) >= laps or done:
                end = "laps_done" if len(times) >= laps else ENDS[info["episode_end"]["end"][0]]
                break
    finally:
        env.close()
    d = {k: np.array(v, np.float64) for k, v in zip(COLUMNS, np.array(rows).T)}
    d |= {"lap_times": np.array(times), "end": end, "step": ckpt["step"], "bot_lap": bot_lap}
    vs_bot = f", best / bot {min(times) / bot_lap:.3f}" if times else ""
    print(f"  ended: {end}, lap times (s): {', '.join(f'{t:.2f}' for t in times) or 'none'} (bot {bot_lap:.2f}{vs_bot})")
    return d


# ---- the bot ----

class Bot:
    """The built-in bot's fastest lap (the one Track.bot_lap times), by distance from the start line and by lap time,
    and the track's edges."""

    def __init__(self, track: str):
        prefix = _prefix(track)
        t = Track.load(track)
        self.length, self.lap = t.length, t.bot_lap
        data = np.loadtxt(f"{prefix}_laps.csv", ndmin=2)
        trace = {c: data[:, i] for i, c in enumerate(TRACE_COLUMNS)}
        best = int(np.argmin(lap_times(trace))) + 1
        lap = {k: v[trace["laps"] == best] for k, v in trace.items()}
        order = np.argsort(lap["dist_from_start"])
        self.s = lap["dist_from_start"][order]
        self.by_s = {k: lap[k][order] for k in ("x", "y", "speed_x", "steer", "accel", "brake")}
        self.by_s["time"] = (lap["time"] - lap["time"][0])[order]
        self.t = lap["time"] - lap["time"][0]  # in time order, for the ghost
        self.xy = np.column_stack([lap["x"], lap["y"]])
        self.edges = np.loadtxt(f"{prefix}_track.csv", ndmin=2)  # left x, left y, right x, right y

    def at(self, key: str, s) -> np.ndarray:
        """The bot's `key` where it was `s` metres from the start line. Its lap time doesn't wrap round the lap: 0 at the
        line, the lap time at its end."""
        if key == "time":
            return np.interp(np.mod(s, self.length), self.s, self.by_s[key])
        return np.interp(np.mod(s, self.length), self.s, self.by_s[key], period=self.length)

    def ghost(self, lap_time: float) -> tuple[float, float]:
        """Where the bot was `lap_time` seconds into its lap."""
        lap_time = min(lap_time, self.t[-1])
        return float(np.interp(lap_time, self.t, self.xy[:, 0])), float(np.interp(lap_time, self.t, self.xy[:, 1]))

    def write_times(self, path: Path) -> None:
        """For the replay robot (RLP_REPLAY_BOT): the lap time, then the time at each metre, as the engine counts them."""
        metres = np.arange(int(self.length + 1))
        np.concatenate([[self.lap], np.interp(metres, self.s, self.by_s["time"])]).astype(np.float32).tofile(path)


def add_delta(d: dict, bot: Bot, s0: float | None = None) -> None:
    """d["delta"]: the lap time minus the bot's at the same place, from the first crossing of the start line (nan
    before: the run up from the grid isn't a lap). From a scenario's start s0: the time since the start minus the
    bot's from s0 to the same place, and cur_lap_time made the bot's lap time there (for the ghost)."""
    if s0 is None:
        d["delta"] = np.where(d["laps"] >= 1, d["cur_lap_time"] - bot.at("time", d["dist_from_start"]), np.nan)
        return
    elapsed = d["time"] - d["time"][0]
    x = s0 + d["dist_raced"] - d["dist_raced"][0]
    bot_elapsed = bot.at("time", x) + np.floor(x / bot.length) * bot.lap - bot.at("time", s0)
    d["delta"] = elapsed - bot_elapsed
    d["elapsed"] = elapsed
    d["cur_lap_time"] = np.mod(bot.at("time", s0) + elapsed, bot.lap)
    d["laps"] = np.ones_like(d["laps"])


def quads(bot: Bot) -> np.ndarray:
    """[n, 4, 2]: the road between consecutive edge samples."""
    e = bot.edges
    nxt = np.roll(e, -1, axis=0)
    return np.stack([e[:, :2], nxt[:, :2], nxt[:, 2:], e[:, 2:]], 1)


def road(ax, bot: Bot, **kw) -> PolyCollection:
    c = ax.add_collection(PolyCollection(quads(bot), facecolor=ROAD, edgecolor=ROAD, linewidth=0.6, **kw))
    ax.set_aspect("equal")
    ax.axis("off")
    return c


# ---- the summary image ----

def summary(d: dict, bot: Bot, title: str, path: Path) -> None:
    """The map coloured by time gained (green) or lost (red) against the bot, and per lap: speed and delta against the
    bot, pedals against the bot's on the last lap."""
    fig = Figure(figsize=(16, 10), dpi=100, layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.2])
    ax = fig.add_subplot(grid[0:2, 0])
    road(ax, bot)
    # the delta's slope (s per 100 m), smoothed over ~25 steps, cut at each start line crossing
    rate = np.gradient(d["delta"], d["dist_raced"]) * 100
    rate[np.r_[False, np.diff(d["laps"]) != 0]] = np.nan
    ok = np.isfinite(rate)
    smooth = np.convolve(np.where(ok, rate, 0), np.ones(25), "same") / np.maximum(np.convolve(ok, np.ones(25), "same"), 1)
    lim = max(np.nanpercentile(np.abs(smooth[ok]), 95), 1e-3) if ok.any() else 1
    pts = np.column_stack([d["pos_x"], d["pos_y"]])
    lines = LineCollection(np.stack([pts[:-1], pts[1:]], 1), cmap="RdYlGn_r", linewidth=2.5)
    lines.set_array(np.where(ok, smooth, 0)[:-1])
    lines.set_clim(-lim, lim)
    ax.add_collection(lines)
    ax.autoscale_view()
    cb = fig.colorbar(lines, ax=ax, orientation="horizontal", shrink=0.6, pad=0.01)
    cb.set_label("s lost (+) or gained (-) per 100 m against the bot")
    if d["end"] not in ("laps_done", "distance"):
        ax.plot(*pts[-1], "x", color="k", ms=12, mew=3)
        ax.annotate(d["end"], pts[-1], textcoords="offset points", xytext=(8, 8), fontsize=11)
    ax.plot(*bot.edges[0, :2], "s", color="k", ms=5)  # the start line

    s_bot = np.linspace(0, bot.length, 2000)
    axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, :])]
    axes[0].plot(s_bot, bot.at("speed_x", s_bot) * 3.6, color=BOT, lw=1.2, label=f"bot ({bot.lap:.2f} s)")
    axes[2].plot(s_bot, bot.at("accel", s_bot) - bot.at("brake", s_bot), color=BOT, lw=1, label="bot")
    laps = [int(n) for n in np.unique(d["laps"]) if n >= 1]
    for i, n in enumerate(laps):
        m = d["laps"] == n
        shade = 0.35 + 0.65 * (i + 1) / len(laps)
        done = f"{d['lap_times'][i]:.2f} s" if i < len(d["lap_times"]) else "unfinished"
        if "elapsed" in d:
            done = f"scenario, {d['elapsed'][-1]:.2f} s"
        axes[0].plot(d["dist_from_start"][m], d["speed_x"][m] * 3.6, color=AGENT, alpha=shade, lw=1.2,
                     label=f"lap {n} ({done})")
        axes[1].plot(d["dist_from_start"][m], d["delta"][m], color=AGENT, alpha=shade, lw=1.2, label=f"lap {n}")
    last = d["laps"] == laps[-1] if laps else np.zeros(len(d["laps"]), bool)
    axes[2].plot(d["dist_from_start"][last], d["accel"][last] - d["brake"][last], color=AGENT, lw=1,
                 label=f"lap {laps[-1] if laps else '-'}")
    axes[1].axhline(0, color=MUTED, lw=0.8)
    for a, label in zip(axes, ("speed (km/h)", "delta to the bot (s)", "throttle + / brake -")):
        a.set_ylabel(label)
        a.set_xlim(0, bot.length)
        a.grid(alpha=0.3)
        a.legend(loc="lower right", fontsize=8)
    axes[2].set_xlabel("distance from the start line (m)")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path)
    print(f"  summary: {path}")


# ---- the live window ----

GAME_W, GAME_H = 960, 540  # the game's window (points), set in its screen.xml
TITLE_BAR = 28             # a macOS window's title bar (points)


def screen_area() -> tuple[int, int, int, int]:
    """The main screen's usable area in points, without the menu bar and the dock: x, y (from the top left), w, h."""
    script = """use framework "AppKit"
set f to current application's NSScreen's mainScreen()'s visibleFrame()
set s to current application's NSScreen's mainScreen()'s frame()
return {item 1 of item 1 of f, item 2 of item 1 of f, item 1 of item 2 of f, item 2 of item 2 of f, item 2 of item 2 of s}"""
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5).stdout
        x, y, w, h, screen_h = (float(v) for v in out.split(","))
        return int(x), int(screen_h - y - h), int(w), int(h)  # AppKit counts y from the bottom
    except Exception:
        return 0, 33, 1440, 900


def place_window(pid: int, x: int, y: int, w: int | None = None, h: int | None = None, front: bool = False) -> str:
    """Move (and size) process `pid`'s first window with System Events: "" if done, else the error. Needs the terminal
    to be allowed in System Settings > Privacy & Security > Accessibility."""
    lines = ['tell application "System Events"', f"set p to first process whose unix id is {pid}",
             f"set position of window 1 of p to {{{x}, {y}}}"]
    if w is not None:
        lines.append(f"set size of window 1 of p to {{{w}, {h}}}")
    if front:
        lines.append("set frontmost of p to true")
    try:
        r = subprocess.run(["osascript", "-e", "\n".join(lines + ["end tell"])], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return "timed out"
    return "" if r.returncode == 0 else (r.stderr.strip() or f"osascript exited with {r.returncode}")


class Telemetry:
    """The window around the game: it fills the screen and leaves the game's window a hole at its top left. Around it:
    a small map and the lap info, the zoomed map, and the last BEHIND metres of telemetry against the bot's at the same
    place (with the bot's next AHEAD metres). update(race time) shows the drive at that time."""

    def __init__(self, d: dict, bot: Bot, name: str, header: str, area: tuple[int, int, int, int]):
        import matplotlib as mpl
        mpl.rcParams["toolbar"] = "None"  # the toolbar would take the bottom of the window
        import matplotlib.pyplot as plt

        self.plt, self.d, self.bot, self.header = plt, d, bot, header
        _, _, W, H = area
        H -= TITLE_BAR
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, num=f"watch: {name}")
        hole_w, hole_h = GAME_W, GAME_H + TITLE_BAR  # the game's window, title bar included

        def box(x, y, w, h):  # points from the window's top left -> an axes
            return self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

        # the right of the game: the small map and the lap info, then the zoomed map
        x0 = hole_w + 10
        R = W - x0 - 8
        self.full = box(x0, 6, 0.42 * R, 150)
        mid = 0.5 * (bot.edges[:, :2] + bot.edges[:, 2:])
        self.full.plot(*np.vstack([mid, mid[:1]]).T, color=ROAD, lw=3, solid_capstyle="round")  # cheaper than quads
        self.full.set_aspect("equal")
        self.full.axis("off")
        self.info = box(x0 + 0.46 * R, 6, 0.54 * R, 150)
        self.info.axis("off")
        self.text = self.info.text(0, 1, "", va="top", family="monospace", fontsize=9)
        self.zoom = box(x0, 162, R, hole_h - 166)
        self.quads, self.mid = quads(bot), mid
        self.near = road(self.zoom, bot)  # redrawn each frame with only the road near the car
        self.zoom.set_adjustable("datalim")  # equal aspect, filling the box
        self.zoom.plot(bot.xy[:, 0], bot.xy[:, 1], color=BOT, lw=1, alpha=0.8)  # the bot's line
        self.trail, = self.zoom.plot([], [], color=AGENT, lw=2)
        self.dots = [(a.plot([], [], "o", color=AGENT, ms=ms, mec="w", zorder=5)[0],
                      a.plot([], [], "o", color=BOT, ms=ms, mec="w", zorder=4)[0]) for a, ms in ((self.full, 6),
                                                                                                 (self.zoom, 10))]

        valid = np.isfinite(d["delta"])
        dmax = max(np.nanmax(np.abs(d["delta"])) if valid.any() else 1, 0.5) * 1.1
        vmax = max(d["speed_x"].max(), bot.by_s["speed_x"].max()) * 3.6 * 1.1
        smax = max(np.abs(d["steer"]).max(), np.abs(bot.by_s["steer"]).max(), 0.05) * 1.1
        span = lambda v: (v.min() - 0.05 * np.ptp(v) - 1e-3, v.max() + 0.05 * np.ptp(v) + 1e-3)
        stds = np.concatenate([d["std_steer"], d["std_pedal"]])
        panels = [("speed (km/h)", (0, vmax), True), ("throttle + / brake -", (-1.05, 1.05), True),
                  ("steer", (-smax, smax), True), ("delta (s)", (-dmax, dmax), False), ("value", span(d["value"]), False),
                  ("std", (0, stds.max() * 1.15), False)]
        # under the game: 3 columns x 2 rows, read down each column
        top, foot, left = hole_h + 12, 34, 52  # where the band starts, room for the x labels, for the y labels
        cw, rh = (W - 8) / 3, (H - top - foot) / 2
        self.tele, self.lines = [], {}
        for i, (label, ylim, has_bot) in enumerate(panels):
            col, row = divmod(i, 2)
            a = box(col * cw + left, top + row * rh, cw - left - 10, rh - 10)
            a.set_xlim(-BEHIND, AHEAD)
            a.set_ylim(*ylim)
            a.set_ylabel(label, fontsize=8)
            a.tick_params(labelsize=7, labelbottom=row == 1)
            a.axvline(0, color=MUTED, lw=0.8)
            a.grid(alpha=0.3)
            self.lines[label] = [a.plot([], [], color=AGENT, lw=1.3)[0]]
            if has_bot:
                self.lines[label].append(a.plot([], [], color=BOT, lw=1.1)[0])
            self.tele.append(a)
        self.lines["std"].append(self.tele[-1].plot([], [], color=AGENT, lw=1.3, ls="--")[0])
        self.tele[-1].legend(self.lines["std"], ["steer", "pedal"], fontsize=7, loc="upper left")
        self.tele[0].legend(self.lines["speed (km/h)"], ["policy", "bot (fastest lap)"], fontsize=7, loc="lower left")
        self.fig.text(0.5, 6 / H, "m from the car (the bot ahead of it: what it did at those places)", ha="center",
                      fontsize=8, color=MUTED)
        self.ahead = np.linspace(-BEHIND, AHEAD, 400)
        # Blitting: everything that moves is drawn on a saved background of the rest (axes, ticks, labels: most of a
        # full draw's time). The zoomed map moves its limits, so its road and the bot's line are redrawn too.
        # (in drawing order: the road, the lines, then the cars on top)
        self.moving = [self.near, self.zoom.lines[0], self.trail] + [a for pair in self.dots for a in pair[::-1]]
        self.moving += [self.text]
        self.moving += [line for lines in self.lines.values() for line in lines]
        for a in self.moving:
            a.set_animated(True)
        self.background = None
        self.fig.canvas.mpl_connect("draw_event", self._saved)
        plt.show(block=False)
        self.fig.canvas.draw()

    def _saved(self, _event) -> None:
        """After a full draw (the first, a resize): the new background, with the moving artists on it."""
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_moving()

    def _draw_moving(self) -> None:
        for a in self.moving:
            a.axes.draw_artist(a) if a.axes else self.fig.draw_artist(a)

    def alive(self) -> bool:
        return self.plt.fignum_exists(self.fig.number)

    def wait(self, seconds: float) -> None:
        """Handle the window's events for `seconds`, without the full redraw plt.pause would do."""
        self.fig.canvas.flush_events()
        time.sleep(seconds)

    def update(self, race_time: float) -> None:
        d, bot = self.d, self.bot
        k = int(np.clip(np.searchsorted(d["time"], race_time, "right") - 1, 0, len(d["time"]) - 1))
        x, y, s = d["pos_x"][k], d["pos_y"][k], d["dist_from_start"][k]
        on_lap = d["laps"][k] >= 1
        gx, gy = bot.ghost(d["cur_lap_time"][k]) if on_lap else (np.nan, np.nan)
        for car, ghost in self.dots:
            car.set_data([x], [y])
            ghost.set_data([gx], [gy])
        recent = slice(max(0, k - 250), k + 1)  # ~10 s
        self.trail.set_data(d["pos_x"][recent], d["pos_y"][recent])
        self.zoom.set_xlim(x - 120, x + 120)
        self.zoom.set_ylim(y - 120, y + 120)
        self.near.set_verts(self.quads[np.hypot(*(self.mid - [x, y]).T) < 250])

        done = "  ".join(f"{t:.2f}" for t in d["lap_times"][: max(0, int(d["laps"][k]) - 1)])
        timing = (f"time    {d['elapsed'][k]:7.2f} s\ndelta   {d['delta'][k]:+7.2f} s\nend     {d['end']}\n"
                  if "elapsed" in d else
                  f"lap {int(d['laps'][k])}     {d['cur_lap_time'][k]:7.2f} s\n"
                  f"bot lap   {bot.lap:7.2f} s\n"
                  f"done      {done or '-'}\n")
        self.text.set_text(
            f"{self.header}\n\n"
            f"{timing}"
            f"speed  {d['speed_x'][k] * 3.6:4.0f} km/h (bot {bot.at('speed_x', s) * 3.6:3.0f})\n"
            f"gear {int(d['gear'][k])}   damage {int(d['damage'][k])}")

        back = slice(int(np.searchsorted(d["dist_raced"][: k + 1], d["dist_raced"][k] - BEHIND)), k + 1)
        rel = d["dist_raced"][back] - d["dist_raced"][k]
        ahead_s = s + self.ahead
        mine = {"speed (km/h)": d["speed_x"][back] * 3.6, "throttle + / brake -": d["accel"][back] - d["brake"][back],
                "steer": d["steer"][back], "delta (s)": d["delta"][back], "value": d["value"][back],
                "std": d["std_steer"][back]}
        bots = {"speed (km/h)": bot.at("speed_x", ahead_s) * 3.6,
                "throttle + / brake -": bot.at("accel", ahead_s) - bot.at("brake", ahead_s),
                "steer": bot.at("steer", ahead_s), "std": None}
        for label, lines in self.lines.items():
            lines[0].set_data(rel, mine[label])
            if label == "std":
                lines[1].set_data(rel, d["std_pedal"][back])
            elif len(lines) > 1:
                lines[1].set_data(self.ahead, bots[label])
        canvas = self.fig.canvas
        if self.background is None:
            canvas.draw()  # saves the background (_saved)
            return
        canvas.restore_region(self.background)
        self._draw_moving()
        canvas.blit(self.fig.bbox)


def arrange(game_pid: int, area: tuple[int, int, int, int]) -> None:
    """The telemetry window over the screen's usable area, the game's window on the hole at its top left. The game's
    window shows up a few seconds after launch, and it's placed again for a few seconds in case the game moves it."""
    x, y, w, h = area
    hint = ("drag the game onto the blank area at the top left. To let watch.py place it, allow your terminal app in "
            "System Settings > Privacy & Security > Accessibility")
    for _ in range(10):  # our own window registers with System Events a moment after it opens
        err = place_window(os.getpid(), x, y, w, h)
        if not err or ("-1719" not in err and "-1728" not in err):
            break
        time.sleep(1)
    if err:
        print(f"  couldn't place the telemetry window ({err}): {hint}")
        return
    first, deadline = None, time.time() + 40
    while time.time() < deadline:
        err = place_window(game_pid, x, y + TITLE_BAR, front=True)
        if not err:
            first = first or time.time()
            if time.time() - first > 6:
                return
        elif "-1719" not in err and "-1728" not in err:  # anything but "no window (yet)": no permission, ...
            print(f"  couldn't place the game's window ({err}): {hint}")
            return
        time.sleep(1)
    if first is None:
        print(f"  the game's window never showed up to be placed: {hint}")


def play(d: dict, bot: Bot, track: str, controls: Path, name: str, header: str, launch: str | None = None,
         fast_until: float | None = None) -> None:
    """The game, in a window, on a fresh local dir: the race file switched to a rendered race, a GAME_W x GAME_H
    window, the HUD's driver inputs on, the robot replaying `controls` and showing the delta to the bot; the telemetry
    window around it. `fast_until`: the game fast-forwards (x16) up to that race time (RLP_REPLAY_FAST_UNTIL, the
    patched race engine), same drive."""
    race = race_for(track)
    check_path = Path(f"{controls}.check.csv")
    check_path.unlink(missing_ok=True)  # the robot writes a new one when the replay starts
    bot_times = controls.with_suffix(".bot")
    bot.write_times(bot_times)
    area = screen_area()
    tele = Telemetry(d, bot, name, header, area)
    with tempfile.TemporaryDirectory(prefix="rlpilot-watch-") as tmp:
        lc = prepare_dir(Path(tmp), 0, race)
        race_xml = lc / "config/raceman" / f"{race}.xml"
        race_xml.write_text(race_xml.read_text().replace('val="results only"', 'val="normal"'))
        hud = lc / "config/osghudconfig.xml"
        text, shown = re.subn(r'(<section name="driverinputWidget">\s*<attnum name="enabled" val=")0', r"\g<1>1",
                              hud.read_text())
        if shown:
            hud.write_text(text)
        screen = lc / "config/screen.xml"
        text = re.sub(r'(name="window width" val=")\d+', rf"\g<1>{GAME_W}", screen.read_text())
        screen.write_text(re.sub(r'(name="window height" val=")\d+', rf"\g<1>{GAME_H}", text))
        print("  opening the game (quit it to finish)")
        log = open(controls.with_suffix(".game.log"), "wb")
        game = subprocess.Popen([str(GAME), "-lc", f"{lc}/", "-s", race, "-nr"], cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT,
                                env=os.environ | {"RLP_REPLAY": str(controls)}
                                | ({"RLP_REPLAY_LAUNCH": launch} if launch else {"RLP_REPLAY_BOT": str(bot_times)})
                                | ({"RLP_REPLAY_FAST_UNTIL": f"{fast_until:.3f}"} if fast_until else {}))
        threading.Thread(target=arrange, args=(game.pid, area), daemon=True).start()
        f, pending, last = None, "", None
        try:
            while game.poll() is None:
                if f is None and check_path.exists():
                    f = open(check_path)
                if f is not None:  # the robot appends a line per replayed tick: race time first
                    *lines, pending = (pending + f.read()).split("\n")
                    if lines:
                        last = float(lines[-1].split(",")[0])
                if tele.alive():
                    if last is not None:
                        tele.update(last)
                    tele.wait(0.03)
                else:  # the telemetry window was closed: just wait for the game
                    time.sleep(0.1)
        finally:
            if game.poll() is None:
                game.terminate()
            log.close()
            if f is not None:
                f.close()
    if tele.alive():
        tele.plt.close(tele.fig)
    if game.returncode:
        print(f"  the game exited with {game.returncode}, see {controls.with_suffix('.game.log')}")
    check(controls)


def check(controls: Path) -> None:
    """The robot logs its position next to the recorded one on each replayed tick: they should be the same."""
    path = Path(f"{controls}.check.csv")
    recorded = controls.stat().st_size // 28  # rlp_tick: 7 x 4 bytes
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [[float(v) for v in row[1:5]] for row in csv.reader(f) if len(row) >= 5]
    if not rows:
        print("  the replay didn't start (no check lines)")
        return
    rows = np.array(rows)
    gap = np.hypot(rows[:, 0] - rows[:, 2], rows[:, 1] - rows[:, 3])
    worst = int(np.argmax(gap))
    print(f"  replayed {len(rows)} of {recorded} ticks; largest gap to the recording {gap[worst]:.2f} m "
          f"(tick {worst})" + ("" if gap[worst] < 0.5 else "  <- the replay strayed: it isn't the policy's drive"))


def suite_eval(checkpoint: Path) -> Path | None:
    """The suite eval of this checkpoint's step (runs/<run>/eval/<step>.csv), else the run's last one."""
    evals = sorted(p for p in (checkpoint.parent.parent / "eval").glob("*.csv") if p.stem.isdigit())
    same = [p for p in evals if p.stem == checkpoint.stem]
    return (same or evals[-1:] or [None])[0]


def eval_outcome(checkpoint: Path, scenario_id: str) -> str:
    path = suite_eval(checkpoint)
    if path is None:
        return ""
    with open(path) as f:
        row = next((r for r in csv.DictReader(f) if r.get("id") == scenario_id), None)
    return f"{row['outcome']}, {row['end']} in {path.name}" if row else ""


def list_failed(checkpoint: Path, tier: str) -> None:
    path = suite_eval(checkpoint)
    if path is None:
        raise SystemExit(f"no suite eval in {checkpoint.parent.parent / 'eval'}")
    suite = {r["id"]: r for r in load_suite(("train", "validation", "test"))}
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if "tier" in r and r["outcome"] == "failed" and tier in ("all", r["tier"])]
    print(f"{len(rows)} failed in {path}  (yaw: the start's heading against the road, deg)")
    for r in rows:
        yaw = np.degrees(suite[r["id"]]["yaw"]) if r["id"] in suite else float("nan")
        print(f"  {r['id']:36s} {r['split']:10s} {r['end']:12s} {float(r['distance']):6.0f} m  yaw {yaw:+5.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", help="run dir, checkpoint file, or experiment name")
    p.add_argument("track", nargs="?", help="a track (laps from the grid); not with --scenario")
    p.add_argument("--scenario", help="a suite scenario id instead, e.g. spring:1200:crashed (track:s:tier)")
    p.add_argument("--failed", metavar="TIER", help="list the checkpoint's failed scenarios of TIER (or 'all') in its "
                   "last suite eval, and stop")
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--best", action="store_true", help="the run's best.pt instead of its last checkpoint")
    p.add_argument("--seed", type=int, help="with an experiment name that has several seeds")
    p.add_argument("--no-game", action="store_true", help="only drive the laps: lap times and the summary image")
    p.add_argument("--test", action="store_true", help="allow a test track (what you see there can bias later choices)")
    p.add_argument("--shift", choices=SHIFT_MODES, default="argmax",
                   help="the gear choice: argmax (as the in-run eval), sample (as in training), hazard (deterministic, "
                        "the median of the sampled timing); see eval.Driver")
    p.add_argument("--shift-seed", type=int, default=0, help="with --shift sample")
    p.add_argument("--from-start", action="store_true", help="watch lap 1 in real time too (by default the game "
                   "fast-forwards to --lead seconds before the end of lap 1, so the flying lap starts right away)")
    p.add_argument("--lead", type=float, default=10.0, help="seconds of lap 1 shown in real time before the flying lap")
    a = p.parse_args()
    split = {t: s for s, ts in load_split().items() for t in ts}
    checkpoint = find_checkpoint(a.run, a.seed, a.best)
    if a.failed:
        list_failed(checkpoint, a.failed)
        return
    scenario = None
    if a.scenario:
        found = [r for r in load_suite(tuple(set(split.values()))) if r["id"] == a.scenario]
        if not found:
            raise SystemExit(f"no scenario {a.scenario!r} in the suite (ids: track:s:tier, see --failed)")
        scenario = found[0]
        a.track = scenario["track"]
    if a.track is None:
        raise SystemExit("give a track, or --scenario")
    if a.track not in split:
        raise SystemExit(f"unknown track {a.track!r}; tracks: {', '.join(sorted(split))}")
    if split[a.track] == "test" and not a.test:
        raise SystemExit(f"{a.track} is a test track: pass --test to watch it anyway")
    out = checkpoint.parent.parent / "watch"
    out.mkdir(exist_ok=True)
    mode = "" if a.shift == "argmax" else f"-{a.shift}" + (f"{a.shift_seed}" if a.shift == "sample" else "")
    name = a.scenario.replace(":", "-") if scenario else a.track
    stem = (out / f"{name}-{checkpoint.stem}{mode}").resolve()  # absolute: the worker and the game chdir
    controls = stem.with_suffix(".controls")
    d = drive(checkpoint, a.track, a.laps, controls, a.shift, a.shift_seed, scenario)
    bot = Bot(a.track)
    add_delta(d, bot, scenario["s"] if scenario else None)
    np.savez_compressed(stem.with_suffix(".npz"), **d)
    times = ", ".join(f"{t:.2f}" for t in d["lap_times"]) or "no lap"
    if scenario:
        times = f"{a.scenario}, {d['elapsed'][-1]:.2f}"
        in_eval = eval_outcome(checkpoint, a.scenario)
        print(f"  scenario: {d['end']} after {d['elapsed'][-1]:.1f} s, {d['dist_raced'][-1] - d['dist_raced'][0]:.0f} m"
              + (f" (the suite eval: {in_eval})" if in_eval else ""))
    title = (f"{checkpoint.parent.parent.name} {checkpoint.stem} (step {d['step']:,}) on {a.track} ({split[a.track]}): "
             f"{times} s, bot {bot.lap:.2f} s, {d['end']}, shift {a.shift}")
    summary(d, bot, title, stem.with_suffix(".png"))
    if not a.no_game:
        run = checkpoint.parent.parent.name
        exp, seed = (run.split("__") + ["?"])[:2]  # runs/<exp>__<seed>__<time>
        where = a.scenario if scenario else a.track
        header = f"{exp} seed {seed}\n{checkpoint.stem}, step {d['step'] / 1e6:.2f}M\n{where} ({split[a.track]}), shift {a.shift}"
        fast_until = None
        if not scenario and not a.from_start and len(d["lap_times"]):  # lap 1 ends when the lap count reaches 2
            lap1_end = float(d["time"][np.argmax(d["laps"] >= 2)])
            fast_until = lap1_end - a.lead
            hits = d["time"][1:][(np.diff(d["damage"]) > 0) & (d["time"][1:] < lap1_end)]
            if len(hits) and hits[-1] > fast_until - 3.0:  # don't open on a wall contact: skip past it
                fast_until = min(float(hits[-1]) + 3.0, lap1_end - 0.5)
            print(f"  fast-forwarding to {fast_until:.1f} s ({lap1_end - fast_until:.1f} s before the flying lap; "
                  "--from-start to watch lap 1)")
        play(d, bot, a.track, controls, run, header, launch_spec(scenario) if scenario else None, fast_until)


if __name__ == "__main__":
    main()
