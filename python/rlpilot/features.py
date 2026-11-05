"""Raw shared-memory fields -> normalized policy observations and agent actions -> car controls.

ObsLayout is the single source of truth for what the policy sees: names, sizes and offsets all derive from its ordered
groups, and assemble() fills an observation group by group, so they can't drift apart. Tests, plots and probes read
observations through it, with no hardcoded index. Each group's `scale` states its frame, sign convention and units.

The groups up to `car` keep the order of proto-3..6, whose checkpoints still load when the new groups are off.
All functions are vectorized over the batch of envs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._native import ACT_DIM, A, F, N_FIELDS, N_RAYS, RAY_RANGE

WHEEL_RADIUS = 0.33  # m, only used to scale wheel spin to ~m/s
CAR_FACTORS = ("car_grip", "car_mass", "car_power")  # EnvConfig ranges = shm act slots, in this order
CAR_OBS_SCALE = 10.0  # observation = (factor - 1) x this
# Observation = metres / this. Not clipped. Over the bot's laps on all 41 tracks at 30 points, |x| reaches 337 m at p99
# and 464 m at most, |y| 146 m and 245 m, so inputs stay within ~5 and are mostly under 3. y gets the finer scale because
# the lateral shape of the road is the signal: a 10 m bend reads 0.2.
EDGE_X_SCALE = 100.0
EDGE_Y_SCALE = 50.0
# Observation = metres / this. On the train and validation tracks at 60 m/s (6 s = 360 m ahead), the road differs by at most
# 50 m from the car's height, so inputs stay within about 5, like the edges.
HEIGHT_SCALE = 10.0
EDGE_GROUPS = ("edge_left_x", "edge_left_y", "edge_right_x", "edge_right_y")  # in the order of EdgeTable.car_frame


@dataclass(frozen=True)
class ObsGroup:
    name: str
    names: tuple[str, ...]  # one per observation slot
    scale: str              # frame, sign convention, units and normalization


class ObsLayout:
    """Ordered observation groups: the only place that knows where anything is in an observation vector."""

    def __init__(self, groups: tuple[ObsGroup, ...]):
        self.groups = tuple(groups)
        self.names = tuple(n for g in self.groups for n in g.names)
        self.dim = len(self.names)
        if len(set(self.names)) != self.dim or len({g.name for g in self.groups}) != len(self.groups):
            raise ValueError("observation names must be unique")
        self.slices: dict[str, slice] = {}
        a = 0
        for g in self.groups:
            self.slices[g.name] = slice(a, a + len(g.names))
            a += len(g.names)
        self._index = {n: i for i, n in enumerate(self.names)}

    def index(self, name: str) -> int:
        """Column of one observation slot, by name."""
        return self._index[name]

    def group(self, obs: np.ndarray, name: str) -> np.ndarray:
        """The columns of a named group of observations [..., dim]."""
        return obs[..., self.slices[name]]

    def get(self, obs: np.ndarray, name: str) -> np.ndarray:
        """One named slot of observations [..., dim]."""
        return obs[..., self._index[name]]

    def __contains__(self, group_name: str) -> bool:
        return group_name in self.slices


def make_layout(action_names: tuple[str, ...], n_lookahead: int = 0, n_line: int = 0, car_obs: bool = False,
                steer_history: int = 0, edge_points: int = 0, height_points: int = 0) -> ObsLayout:
    named = lambda prefix, n: tuple(f"{prefix}{i}" for i in range(n))
    groups = [
        ObsGroup("speed", ("speed_x", "speed_y", "speed_z"), "car frame, m/s: x / 50 (+ = forward), y / 10, z / 10"),
        ObsGroup("yaw_rate", ("yaw_rate",), "rad/s / 2"),
        ObsGroup("angle", ("angle",), "track tangent minus car yaw, rad / pi, + = the road points left of the car"),
        ObsGroup("to_middle", ("to_middle",), "offset from the centreline / half width, + = left of it"),
        ObsGroup("rpm", ("rpm",), "engine speed / redline"),
        ObsGroup("gear", ("gear",), "gear / 6"),
        ObsGroup("wheel_spin", named("wheel_spin", 4), "rad/s x wheel radius / 50 (~ m/s / 50), FR FL RR RL"),
        ObsGroup("rays", named("ray", N_RAYS), f"distance to the road's edge, m / {RAY_RANGE:g}, -90 to +90 deg, + = left"),
        ObsGroup("prev_action", tuple(f"prev_{a}" for a in action_names),
                 "controls of the previous step, 0 on an episode's first observation. prev_steer is the steering "
                 "command in [-1, 1] (1 = 45 deg of wheel, + = left), in both steering modes"),
    ]
    if n_lookahead:
        groups.append(ObsGroup("lookahead", named("lookahead", n_lookahead),
                               "centreline curvature in windows ahead, 1/m x 25, + = turning left, clipped to +-3"))
    if n_line:
        groups.append(ObsGroup("line", named("line", n_line),
                               "the bot's racing line relative to the car, m / half width, + = left, clipped to +-3"))
    if car_obs:
        groups.append(ObsGroup("car", CAR_FACTORS, f"(factor - 1) x {CAR_OBS_SCALE:g}"))
    if steer_history:
        groups.append(ObsGroup("steer_history", named("steer_change", steer_history),
                               "applied steering change of the last steps, most recent first, command units / steer_limit "
                               "(so in [-1, 1]), + = towards the left, 0 before an episode's first steps"))
    if edge_points:
        for side in ("left", "right"):
            groups.append(ObsGroup(f"edge_{side}_x", named(f"edge_{side}_x", edge_points),
                                   f"{side} road edge at equal centreline spacing over max(horizon x speed, floor) from "
                                   f"abreast of the car (index 0), car frame, m / {EDGE_X_SCALE:g}, + = ahead, not clipped"))
            groups.append(ObsGroup(f"edge_{side}_y", named(f"edge_{side}_y", edge_points),
                                   f"same points, m / {EDGE_Y_SCALE:g}, + = left of the car, not clipped"))
    if height_points:
        groups.append(ObsGroup("height", named("height", height_points),
                               f"road height ahead minus at the car, along the centreline, over the edges' span, m / {HEIGHT_SCALE:g}, "
                               "+ = above, not clipped"))
    return ObsLayout(tuple(groups))


def base_parts(raw: np.ndarray, prev_action: np.ndarray) -> dict[str, np.ndarray]:
    """raw [n, OBS_DIM] (shm layout), prev_action [n, action_dim] -> the groups every layout has, per group name."""
    half_width = np.maximum(raw[:, F["track_width"]], 1.0) * 0.5
    return {
        "speed": raw[:, [F["speed_x"], F["speed_y"], F["speed_z"]]] / np.array([50.0, 10.0, 10.0], np.float32),
        "yaw_rate": raw[:, F["yaw_rate"]] / 2.0,
        "angle": raw[:, F["angle"]] / np.pi,
        "to_middle": raw[:, F["to_middle"]] / half_width,
        "rpm": raw[:, F["rpm"]] / np.maximum(raw[:, F["rpm_redline"]], 1.0),
        "gear": raw[:, F["gear"]] / 6.0,
        "wheel_spin": raw[:, F["wheel_spin0"]:F["wheel_spin0"] + 4] * (WHEEL_RADIUS / 50.0),
        "rays": raw[:, N_FIELDS:N_FIELDS + N_RAYS] / RAY_RANGE,
        "prev_action": prev_action,
    }


def edge_parts(car_frame: np.ndarray) -> dict[str, np.ndarray]:
    """EdgeTable.car_frame output [n, points, 4] -> the normalized edge groups."""
    return {name: car_frame[..., i] / (EDGE_Y_SCALE if name.endswith("y") else EDGE_X_SCALE)
            for i, name in enumerate(EDGE_GROUPS)}


def decode_edges(layout: ObsLayout, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observations [..., dim] -> left, right edge points in the car's frame (m, [..., points, 2]: x forward, y left)."""
    def side(s):
        return np.stack([layout.group(obs, f"edge_{s}_x") * EDGE_X_SCALE,
                         layout.group(obs, f"edge_{s}_y") * EDGE_Y_SCALE], -1)
    return side("left"), side("right")


def assemble(layout: ObsLayout, parts: dict[str, np.ndarray]) -> np.ndarray:
    """Per-group values -> observations [n, layout.dim]. Each layout group must be there, with exactly its width (a
    reshape, so that a wrong width raises rather than broadcasts), and nothing else may be: no slot can stay unfilled."""
    if len(parts) != len(layout.groups):
        raise ValueError(f"expected groups {list(layout.slices)}, got {list(parts)}")
    n = len(parts["prev_action"])
    out = np.empty((n, layout.dim), np.float32)
    for g in layout.groups:
        s = layout.slices[g.name]
        out[:, s] = parts[g.name].reshape(n, s.stop - s.start)
    return out


def write_actions(action: np.ndarray, act: np.ndarray, separate_pedals: bool) -> None:
    """Controls in [-1, 1] (first column: the steering command) -> shm act [n, ACT_DIM] (steer, accel, brake, gear)."""
    a = np.clip(action, -1.0, 1.0)
    act[:, A["steer"]] = a[:, 0]
    if separate_pedals:
        act[:, A["accel"]] = (a[:, 1] + 1) * 0.5
        act[:, A["brake"]] = (a[:, 2] + 1) * 0.5
    else:
        act[:, A["accel"]] = np.maximum(a[:, 1], 0)
        act[:, A["brake"]] = np.maximum(-a[:, 1], 0)
    act[:, A["gear"]] = 0


assert ACT_DIM == 13
