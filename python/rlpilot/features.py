"""Raw shared-memory fields -> normalized policy observations and agent actions -> car controls.
All functions are vectorized over the batch of envs."""

from __future__ import annotations

import numpy as np

from ._native import ACT_DIM, A, F, N_FIELDS, N_RAYS, RAY_RANGE

WHEEL_RADIUS = 0.33  # m, only used to scale wheel spin to ~m/s

OBS_NAMES = (["speed_x", "speed_y", "speed_z", "yaw_rate", "angle", "to_middle", "rpm", "gear"]
             + [f"wheel_spin{i}" for i in range(4)] + [f"ray{i}" for i in range(N_RAYS)])


def obs_dim(action_dim: int) -> int:
    return len(OBS_NAMES) + action_dim  # + previous action


def build_obs(raw: np.ndarray, prev_action: np.ndarray, out: np.ndarray) -> np.ndarray:
    """raw [n, OBS_DIM] (shm layout), prev_action [n, action_dim] -> out [n, obs_dim], float32."""
    half_width = np.maximum(raw[:, F["track_width"]], 1.0) * 0.5
    out[:, 0] = raw[:, F["speed_x"]] / 50.0
    out[:, 1] = raw[:, F["speed_y"]] / 10.0
    out[:, 2] = raw[:, F["speed_z"]] / 10.0
    out[:, 3] = raw[:, F["yaw_rate"]] / 2.0
    out[:, 4] = raw[:, F["angle"]] / np.pi
    out[:, 5] = raw[:, F["to_middle"]] / half_width
    out[:, 6] = raw[:, F["rpm"]] / np.maximum(raw[:, F["rpm_redline"]], 1.0)
    out[:, 7] = raw[:, F["gear"]] / 6.0
    out[:, 8:12] = raw[:, F["wheel_spin0"]:F["wheel_spin0"] + 4] * (WHEEL_RADIUS / 50.0)
    out[:, 12:12 + N_RAYS] = raw[:, N_FIELDS:N_FIELDS + N_RAYS] / RAY_RANGE
    out[:, 12 + N_RAYS:] = prev_action
    return out


def write_actions(action: np.ndarray, act: np.ndarray, separate_pedals: bool) -> None:
    """Agent actions in [-1, 1] -> shm act [n, ACT_DIM] (steer, accel, brake, gear)."""
    a = np.clip(action, -1.0, 1.0)
    act[:, A["steer"]] = a[:, 0]
    if separate_pedals:
        act[:, A["accel"]] = (a[:, 1] + 1) * 0.5
        act[:, A["brake"]] = (a[:, 2] + 1) * 0.5
    else:
        act[:, A["accel"]] = np.maximum(a[:, 1], 0)
        act[:, A["brake"]] = np.maximum(-a[:, 1], 0)
    act[:, A["gear"]] = 0


assert ACT_DIM == 6
