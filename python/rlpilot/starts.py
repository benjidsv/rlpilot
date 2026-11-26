"""Where and how an episode starts. One draw function for training and for the eval suite, so that both see the
same kinds of situation.

Kinds (training draws them by TRAIN_SHARES; the suite has one scenario of each tier per start point):
- grid:      a standing start on the grid.
- nominal:   on the bot's line at the bot's speed there (+-10%), facing down the road. "Can it drive this road?"
- mild:      (eval only) a bit off the line, a few degrees of heading, 80-110% of the bot's speed.
- perturbed: anywhere across 80% of the road, 0 to 1.2x the bot's speed, and a heading error that shrinks with speed
             (up to 30 deg when slow, a few degrees at 1.2x): big errors only where they can be survived.
- disturbed: a nominal start, then a fraction of a second of forced bad input (a steering jerk, a brake stab, or a
             throttle stab while steering) before the agent takes over: real slides, yaw rates and weight transfer.
             The forced steps run inside the reset: the policy and the rollout buffer never see them.
- crashed:   nearly stopped, facing any way, anywhere across the road or a little beyond its edge (inside the
             barriers): "get going again".

Launches are physically bounded so that nominal starts are feasible by construction: the bot drove that spot at that
speed. Every launch starts with the wheels straight (the engine can't launch with steering on).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .client import A
from .tracks import TrackTable, half_width

KINDS = ("grid", "nominal", "mild", "perturbed", "disturbed", "crashed")
K = {k: i for i, k in enumerate(KINDS)}
TRAIN_SHARES = {"nominal": 0.50, "perturbed": 0.25, "disturbed": 0.12, "crashed": 0.08, "grid": 0.05}
SUITE_TIERS = ("nominal", "mild", "perturbed", "disturbed", "crashed")
CAR_MARGIN = 2.5   # m kept between a crashed car's centre and the barrier (it may face any way: half its length + room)
OFF_ROAD_MAX = 3.0  # m beyond the road's edge a crashed car may start


@dataclass
class Starts:
    """A batch of starts, one per row."""
    kind: np.ndarray        # index in KINDS
    s: np.ndarray           # m from the start line (unused on the grid)
    lateral: np.ndarray     # m from the centreline, + = left
    yaw: np.ndarray         # rad from the road's direction, + = left
    speed: np.ndarray       # m/s, along the car
    kick_steer: np.ndarray  # forced action during the kick (disturbed only), in action units
    kick_pedal: np.ndarray
    kick_steps: np.ndarray  # agent steps of forced action (0: none)

    def __len__(self) -> int:
        return len(self.kind)

    def __getitem__(self, idx) -> "Starts":
        return Starts(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def to_columns(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def draw(kinds: np.ndarray, track: np.ndarray, s: np.ndarray, table: TrackTable, rng: np.random.Generator) -> Starts:
    """kinds [n] (indices in KINDS), track [n] (indices in table.tracks), s [n] (m along the lap) -> Starts."""
    n = len(kinds)
    look = table.at(track, s[:, None], ("left_x", "left_y", "right_x", "right_y", "side_left", "side_right",
                                        "bot_offset", "bot_speed"))[:, 0]
    hw = half_width(look[:, 0:2], look[:, 2:4])
    side_l, side_r, line, v_bot = look[:, 4], look[:, 5], look[:, 6], look[:, 7]
    u = lambda lo, hi: rng.uniform(lo, hi, n)  # every draw is made for every row, so a row's values don't depend on
    sign = rng.choice([-1.0, 1.0], n)          # which kinds the other rows are

    lateral = line + u(-0.5, 0.5)
    yaw = u(-0.02, 0.02)
    speed = v_bot * u(0.9, 1.1)
    kick_steer, kick_pedal, kick_steps = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)

    mild = kinds == K["mild"]
    lateral = np.where(mild, line + u(-0.3, 0.3) * hw, lateral)
    yaw = np.where(mild, u(-1, 1) * np.radians(5), yaw)
    speed = np.where(mild, v_bot * u(0.8, 1.1), speed)

    pert = kinds == K["perturbed"]
    f = u(0, 1.2)
    yaw_max = np.radians(30) * np.clip((1.3 - f) / 0.8, 0.1, 1.0)
    lateral = np.where(pert, u(-0.8, 0.8) * hw, lateral)
    yaw = np.where(pert, u(-1, 1) * yaw_max, yaw)
    speed = np.where(pert, f * v_bot, speed)

    dist = kinds == K["disturbed"]
    kick = rng.integers(0, 3, n)  # 0 steering jerk, 1 brake stab, 2 throttle stab, each with some steering
    kick_steer = np.where(dist, sign * np.select([kick == 0], [1.0], 0.5), kick_steer)
    kick_pedal = np.where(dist, np.select([kick == 1, kick == 2], [-1.0, 1.0], 0.0), kick_pedal)
    kick_steps = np.where(dist, rng.integers(8, 21, n), kick_steps)  # 0.32 to 0.8 s

    crash = kinds == K["crashed"]
    lo = np.minimum(-(hw + np.minimum(side_r, OFF_ROAD_MAX) - CAR_MARGIN), -0.3 * hw)
    hi = np.maximum(hw + np.minimum(side_l, OFF_ROAD_MAX) - CAR_MARGIN, 0.3 * hw)
    lateral = np.where(crash, lo + u(0, 1) * (hi - lo), lateral)
    yaw = np.where(crash, u(-np.pi, np.pi), yaw)
    speed = np.where(crash, u(0, 3), speed)

    return Starts(kinds.astype(np.int64), s.astype(np.float64), lateral, yaw, np.maximum(speed, 0.0), kick_steer,
                  kick_pedal, kick_steps)


def write_launch(starts: Starts, track: np.ndarray, table: TrackTable, act: np.ndarray) -> None:
    """Starts [n] -> the launch slots of act [n, ACT_DIM]; the worker reads them when its race restarts."""
    look = table.at(track, starts.s[:, None], ("left_x", "left_y", "right_x", "right_y"))[:, 0]
    hw = np.maximum(half_width(look[:, 0:2], look[:, 2:4]), 0.5)
    length = table.length[track]
    grid = starts.kind == K["grid"]
    pos = np.clip(np.mod(starts.s, length) / length, 1e-4, 1 - 1e-4)
    slots = np.stack([pos, (starts.lateral + hw) / (2 * hw), starts.yaw, starts.speed], 1)
    act[:, A["launch_pos"]:A["launch_speed"] + 1] = np.where(grid[:, None], 0.0, slots).astype(np.float32)


def train_kinds(n: int, rng: np.random.Generator, crashed_share: float = 0.0) -> np.ndarray:
    """n kinds drawn by TRAIN_SHARES. crashed_share > 0: crashed starts get that share instead, the difference taken
    from nominal (2026-11-25: backward-facing crashed starts, ~4% of training, are a skill learnt then lost)."""
    shares = dict(TRAIN_SHARES)
    if crashed_share > 0:
        shares["nominal"] -= crashed_share - shares["crashed"]
        shares["crashed"] = crashed_share
        if shares["nominal"] < 0:
            raise ValueError(f"crashed_share {crashed_share} leaves no nominal starts")
    names = list(shares)
    p = np.array([shares[k] for k in names])
    return np.array([K[k] for k in names])[rng.choice(len(names), n, p=p / p.sum())]
