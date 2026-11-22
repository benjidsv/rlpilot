"""What the policy sees: what a driver gets. NAMES lists every input in order, with its scale.

- The car: its motion and what the driver feels (speed, yaw rate, accelerations with gravity in them), where it is on
  the road, the engine and the gearbox, and the controls applied at the previous step.
- The road, from the track map (a driver who knows the track, or reads the pace notes): both edges at POINTS along the
  centreline, from 20 m behind (the mirrors, for backing out) to 300 m ahead (the longest braking zones), densest
  around the car. Each point gives the edge's position in the car's frame, its height against the road at the car
  (crests, dips and banking), and what lies beyond the edge: the width before the barrier, and whether it's grass or
  loose (sand, gravel, dirt) rather than paved. Surfaces come as kinds, as a driver sees them, not as exact grip.

No rays: they only measured the road's edge, like the edge points, and read 0 as soon as the car was off the road.
"""

from __future__ import annotations

import numpy as np

from .client import F
from .controls import SHIFT_COOLDOWN, TOP_GEAR, Controls
from .tracks import TrackTable

POINTS = np.array([-20, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30, 40, 55, 70, 90, 115, 145, 180, 220, 260, 300],
                  np.float64)  # m along the centreline from the car
RUNOFF_CAP = 20.0  # m: beyond this the runoff is "plenty"
GEARS = (-1,) + tuple(range(1, TOP_GEAR + 1))
_CAR = POINTS.tolist().index(0)
EDGE_INPUTS = ("x / 100", "y / 50", "z / 10", f"runoff / {RUNOFF_CAP:g}", "grass", "loose")

NAMES = (
    ["speed_x / 50", "speed_y / 10", "yaw_rate / 2", "accel_x / 10", "accel_y / 10", "accel_z / 10"]  # car frame
    + ["angle / pi", "to_middle / half width"]                # heading to the road (+ = road points left), offset (+ = left)
    + ["rpm / redline", "rpm_up / redline", "rpm_down / redline"]
    + [f"gear {g}" for g in ("R",) + GEARS[1:]] + ["shift cooldown / max"]
    + ["steer command", "pedal"]                              # the controls applied at the previous step
    # x forward, y left (car frame); z against the road's centre at the car
    + [f"{side} {what} @{d:g}m" for side in ("left", "right") for what in EDGE_INPUTS for d in POINTS]
)
DIM = len(NAMES)
_COLUMNS = ("left_x", "left_y", "left_z", "side_left", "grass_left", "loose_left",
            "right_x", "right_y", "right_z", "side_right", "grass_right", "loose_right")


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
    g = table.at(track, s, _COLUMNS)  # [n, k, 12]
    x, y, yaw = (raw[:, F[k]].astype(np.float64)[:, None] for k in ("pos_x", "pos_y", "yaw"))
    c, si = np.cos(yaw), np.sin(yaw)
    z0 = 0.5 * (g[:, _CAR, 2] + g[:, _CAR, 8])[:, None]  # the road's height at the car
    edges = []
    for o in (0, 6):  # left, right
        dx, dy = g[..., o] - x, g[..., o + 1] - y
        edges += [(c * dx + si * dy) / 100, (c * dy - si * dx) / 50, (g[..., o + 2] - z0) / 10,
                  np.minimum(g[..., o + 3], RUNOFF_CAP) / RUNOFF_CAP, g[..., o + 4], g[..., o + 5]]
    out = np.concatenate([car, gear, shift, prev, *edges], 1).astype(np.float32)
    assert out.shape == (n, DIM)
    return out
