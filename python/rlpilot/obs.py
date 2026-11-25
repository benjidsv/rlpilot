"""What the policy sees: what a driver gets. NAMES lists every input in order, with its scale (105 inputs).

- The car (21): its motion and what the driver feels (speed, yaw rate, accelerations with gravity in them), where it
  is on the road, the engine and the gearbox, and the controls applied at the previous step.
- The road's shape (84), from the track map (a driver who knows the track, or reads the pace notes): both edges at
  POINTS along the centreline, in the car's frame, from 20 m behind (the mirrors, for backing out) to 300 m ahead (the
  longest braking zones), densest around the car.

How this was chosen (3M tests on a 10M schedule, 2026-11-22/23; the variants' code is in
runs/obs-variants-2026-11-23.py):
- No rays: they only measured the road's edge, and read 0 as soon as the car was off the road. Without them the car
  learned about as well (validation nominal survival 0.58 at 3M, against verify's 0.61 at 5M) and completed far more
  crashed starts (0.64 against 0.38).
- Extras at every point (height, runoff width, grass / loose; 273 inputs) cost most of the driving (0.23): they drown
  the edges in the first layer. relu or a smaller first-layer init barely helped.
- 24 compact extras at fixed scales (129 inputs) trained better but evaluated worse than this (0.46): fewer clean
  scenarios, more lost control, slower. Left for later ablations. The track table still has all of it (tracks.py).

Optional, `configure(height=...)` (train.py --obs-height; saved in checkpoints): the road's height at the same POINTS,
against the road's height at the car, in m / 10. "centre": the centreline's (21 inputs: slopes, crests and dips);
"edges": each edge's (42: also the banking). The failure map (docs/v2.md) found a downhill approach makes a corner
harder, and the policy can't see slopes otherwise. Off by default.

FIXED marks inputs already at a fixed scale, which the trainer's running normalization must skip (it would blow up
the noise of near-constant inputs). Off by default. `configure(fixed_gears=True)` (train.py --fixed-gear-inputs, the
mask is saved in checkpoints as `obs_fixed`): the gear one-hots and the shift cooldown pass through as 0 / 1. Under the
running normalization a gear the car rarely uses reads as a huge input when it engages (2026-11-24, baseline s1 at 10M:
5th +7.2 sd, 6th +7.7, R +13.3 and clipped to 10, against +0.6 for 3rd): a jump at the very upshifts the car avoids.
"""

from __future__ import annotations

import numpy as np

from .client import F
from .controls import SHIFT_COOLDOWN, TOP_GEAR, Controls
from .tracks import TrackTable

POINTS = np.array([-20, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30, 40, 55, 70, 90, 115, 145, 180, 220, 260, 300],
                  np.float64)  # m along the centreline from the car
GEARS = (-1,) + tuple(range(1, TOP_GEAR + 1))

_BASE_NAMES = (
    ["speed_x / 50", "speed_y / 10", "yaw_rate / 2", "accel_x / 10", "accel_y / 10", "accel_z / 10"]  # car frame
    + ["angle / pi", "to_middle / half width"]  # heading to the road (+ = road points left), offset (+ = left)
    + ["rpm / redline", "rpm_up / redline", "rpm_down / redline"]
    + [f"gear {g}" for g in ("R",) + GEARS[1:]] + ["shift cooldown / max"]
    + ["steer command", "pedal"]  # the controls applied at the previous step
    + [f"{side} {xy} @{d:g}m" for side in ("left", "right") for xy in ("x / 100", "y / 50") for d in POINTS]
)
HEIGHTS = ("none", "centre", "edges")
HEIGHT = "none"
NAMES, DIM, FIXED = _BASE_NAMES, len(_BASE_NAMES), np.zeros(len(_BASE_NAMES), bool)


def configure(height: str = "none", fixed_gears: bool = False) -> None:
    """Set the optional inputs (before making envs or a policy): NAMES, DIM and FIXED follow. fixed_gears: the gear
    one-hots and the shift cooldown skip the running normalization."""
    global HEIGHT, NAMES, DIM, FIXED
    if height not in HEIGHTS:
        raise ValueError(f"height: one of {HEIGHTS}, not {height!r}")
    extra = {"none": [], "centre": [f"height @{d:g}m / 10" for d in POINTS],
             "edges": [f"{side} height @{d:g}m / 10" for side in ("left", "right") for d in POINTS]}[height]
    HEIGHT, NAMES = height, _BASE_NAMES + extra
    DIM = len(NAMES)
    FIXED = np.array([fixed_gears and (n.startswith("gear ") or n.startswith("shift cooldown")) for n in NAMES], bool)


def observe(raw: np.ndarray, track: np.ndarray, controls: Controls, table: TrackTable) -> np.ndarray:
    """raw [n, OBS_DIM] (shm rows), track [n] (index in table.tracks) -> observations [n, DIM] (float32)."""
    n = len(raw)
    red = np.maximum(raw[:, F["rpm_redline"]], 1.0)
    hw = np.maximum(raw[:, F["track_width"]], 1.0) * 0.5
    car = np.stack([raw[:, F["speed_x"]] / 50, raw[:, F["speed_y"]] / 10, raw[:, F["yaw_rate"]] / 2,
                    raw[:, F["accel_x"]] / 10, raw[:, F["accel_y"]] / 10, raw[:, F["accel_z"]] / 10,
                    raw[:, F["angle"]] / np.pi, raw[:, F["to_middle"]] / hw,
                    raw[:, F["rpm"]] / red, raw[:, F["rpm_up"]] / red, raw[:, F["rpm_down"]] / red], 1)
    gear = (controls.gear[:, None] == np.array(GEARS)[None, :]).astype(np.float32)
    shift = (controls.cooldown / SHIFT_COOLDOWN)[:, None]
    prev = np.stack([controls.steer, controls.pedal], 1)

    s = raw[:, F["dist_from_start"]].astype(np.float64)[:, None] + POINTS[None, :]
    world = table.at(track, s, ("left_x", "left_y", "right_x", "right_y", "left_z", "right_z"))  # [n, k, 6]
    x, y, yaw = (raw[:, F[k]].astype(np.float64)[:, None] for k in ("pos_x", "pos_y", "yaw"))
    c, si = np.cos(yaw), np.sin(yaw)
    edges = []
    for o in (0, 2):  # left, right
        dx, dy = world[..., o] - x, world[..., o + 1] - y
        edges += [(c * dx + si * dy) / 100, (c * dy - si * dx) / 50]  # forward, left
    if HEIGHT != "none":
        z0 = 0.5 * (world[:, POINTS == 0, 4] + world[:, POINTS == 0, 5])  # [n, 1]: the road's height at the car
        if HEIGHT == "centre":
            edges.append((0.5 * (world[..., 4] + world[..., 5]) - z0) / 10)
        else:
            edges += [(world[..., 4] - z0) / 10, (world[..., 5] - z0) / 10]
    out = np.concatenate([car, gear, shift, prev, *edges], 1).astype(np.float32)
    assert out.shape == (n, DIM)
    return out
