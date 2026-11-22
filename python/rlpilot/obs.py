"""What the policy sees. PLACEHOLDER for phase 1: a plain, readable set of inputs, to be redesigned in phase 2.

Rule of thumb: what a driver gets. The car's motion and what it feels, the engine and gearbox, where the car is on
the road, the road ahead from the track map, and the walls around it (rays, for now). NAMES lists every input in
order, with its scale.
"""

from __future__ import annotations

import numpy as np

from .client import F, N_FIELDS, N_RAYS, RAY_RANGE
from .controls import SHIFT_COOLDOWN, TOP_GEAR, Controls
from .tracks import TrackTable

EDGE_DISTANCES = np.array([0, 5, 10, 20, 35, 50, 75, 100, 150, 200], np.float64)  # m along the centreline
GEARS = (-1,) + tuple(range(1, TOP_GEAR + 1))

NAMES = (
    ["speed_x / 50", "speed_y / 10", "yaw_rate / 2", "accel_x / 10", "accel_y / 10"]   # car frame: m/s, rad/s, m/s^2
    + ["angle / pi", "to_middle / half width"]                # heading to the road (+ = road points left), offset (+ = left)
    + ["rpm / redline", "rpm_up / redline", "rpm_down / redline"]
    + [f"gear {g}" for g in ("R",) + GEARS[1:]] + ["shift cooldown / max"]
    + ["steer command", "pedal"]                              # the controls applied at the previous step
    + [f"edge_{side}_{xy}_{d:g}m" for side in ("left", "right") for xy in ("x", "y") for d in EDGE_DISTANCES]  # car frame, x/100, y/50
    + [f"ray_{i} / {RAY_RANGE:g}" for i in range(N_RAYS)]     # distance to the road's edge, -90 to +90 deg, + = left
)
DIM = len(NAMES)


def observe(raw: np.ndarray, track: np.ndarray, controls: Controls, table: TrackTable) -> np.ndarray:
    """raw [n, OBS_DIM] (shm rows), track [n] (index in table.tracks) -> observations [n, DIM] (float32)."""
    n = len(raw)
    red = np.maximum(raw[:, F["rpm_redline"]], 1.0)
    hw = np.maximum(raw[:, F["track_width"]], 1.0) * 0.5
    car = np.stack([raw[:, F["speed_x"]] / 50, raw[:, F["speed_y"]] / 10, raw[:, F["yaw_rate"]] / 2,
                    raw[:, F["accel_x"]] / 10, raw[:, F["accel_y"]] / 10,
                    raw[:, F["angle"]] / np.pi, raw[:, F["to_middle"]] / hw,
                    raw[:, F["rpm"]] / red, raw[:, F["rpm_up"]] / red, raw[:, F["rpm_down"]] / red], 1)
    gear = (controls.gear[:, None] == np.array(GEARS)[None, :]).astype(np.float32)
    shift = (controls.cooldown / SHIFT_COOLDOWN)[:, None]
    prev = np.stack([controls.steer, controls.pedal], 1)

    s = raw[:, F["dist_from_start"]].astype(np.float64)[:, None] + EDGE_DISTANCES[None, :]
    world = table.at(track, s, ("left_x", "left_y", "right_x", "right_y"))  # [n, k, 4]
    x, y, yaw = (raw[:, F[k]].astype(np.float64)[:, None] for k in ("pos_x", "pos_y", "yaw"))
    c, si = np.cos(yaw), np.sin(yaw)
    edges = []
    for side in (0, 2):  # left, right
        dx, dy = world[..., side] - x, world[..., side + 1] - y
        edges += [(c * dx + si * dy) / 100, (c * dy - si * dx) / 50]  # forward, left
    rays = raw[:, N_FIELDS:N_FIELDS + N_RAYS] / RAY_RANGE
    out = np.concatenate([car, gear, shift, prev, *edges, rays], 1).astype(np.float32)
    assert out.shape == (n, DIM)
    return out
