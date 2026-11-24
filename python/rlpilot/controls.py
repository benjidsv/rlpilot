"""Agent actions -> car controls: steering, pedals and the gearbox. The only place that knows how a car is driven.

An action is [steer, pedal, shift] per env:
- steer in [-1, 1]: a change of the steering command, at most STEER_RATE per step (+ = left). The command itself is in
  [-1, 1], 1 = full lock (placeholder parametrization: v1's, which drove smoothly; to revisit in phase 2).
- pedal in [-1, 1]: > 0 throttle, < 0 brake.
- shift in {0, 1, 2}: down, hold, up (a categorical choice, carried as a float in the action array).

The gearbox is sequential, like a racing car's paddles: R <-> 1 <-> 2 ... <-> TOP_GEAR, one gear per shift, and no shift
until SHIFT_COOLDOWN steps after the last one (the simulator takes 0.1 s to engage a gear and 0.1 s to release the
clutch on this car). Two refusals, like a real car's electronics: a downshift that would take the engine past its
redline, and any move between 1st and reverse unless the car is nearly stopped. The worker only applies the gear it
is sent, and slips the clutch when pulling away in 1st or reverse.

configure("auto") (a diagnostic, --gearbox auto): while the car moves, auto_shift picks the shift and the policy's
choice is ignored; when it is nearly stopped, the policy still chooses (1st <-> reverse, getting out of a stuck spot).
"""

from __future__ import annotations

import numpy as np

from .client import A, F

STEER_RATE = 0.0625    # command units per step: 70 deg/s of wheel at 25 Hz
TOP_GEAR = 6           # sc-cavallo-360 (the worker clamps to the car's own gears anyway)
SHIFT_COOLDOWN = 5     # agent steps (0.2 s at 25 Hz)
STOPPED_SPEED = 1.0    # m/s: below this, 1st <-> reverse is allowed
DOWN, HOLD, UP = 0, 1, 2
GEARBOXES = ("manual", "auto")
GEARBOX = "manual"


def configure(gearbox: str = "manual") -> None:
    """Who shifts while the car moves (before making envs): the policy ("manual") or auto_shift ("auto")."""
    global GEARBOX
    if gearbox not in GEARBOXES:
        raise ValueError(f"gearbox: one of {GEARBOXES}, not {gearbox!r}")
    GEARBOX = gearbox


def policy_shifts(speed: np.ndarray) -> np.ndarray:
    """[n] bool: whether the policy's shift choice is used at this speed (m/s, signed or not)."""
    return np.ones(np.shape(speed), bool) if GEARBOX == "manual" else np.abs(speed) < STOPPED_SPEED


class Controls:
    """Per-env control state: the steering command, the selected gear and the shift cooldown."""

    def __init__(self, n: int):
        self.steer = np.zeros(n, np.float32)
        self.pedal = np.zeros(n, np.float32)
        self.gear = np.ones(n, np.int64)
        self.cooldown = np.zeros(n, np.int64)
        self.shifted = np.zeros(n, bool)   # this step
        self.refused = np.zeros(n, bool)   # this step: a shift the gearbox didn't allow

    def reset(self, idx: np.ndarray, raw: np.ndarray) -> None:
        """Envs idx start an episode: wheels straight, in the gear the worker launched them in."""
        self.steer[idx] = self.pedal[idx] = 0
        g = np.rint(raw[idx, F["gear"]]).astype(np.int64)
        self.gear[idx] = np.where(g < 0, -1, np.maximum(g, 1))
        self.cooldown[idx] = 0
        self.shifted[idx] = self.refused[idx] = False

    def apply(self, action: np.ndarray, raw: np.ndarray, act: np.ndarray, active: np.ndarray | None = None) -> None:
        """action [n, 3], raw [n, OBS_DIM] (the state the action responds to) -> act [n, ACT_DIM] (shm). Envs not
        `active` keep their state."""
        on = np.ones(len(action), bool) if active is None else active
        steer = np.clip(self.steer + np.clip(action[:, 0], -1, 1) * STEER_RATE, -1, 1)
        self.steer = np.where(on, steer, self.steer).astype(np.float32)
        self.pedal = np.where(on, np.clip(action[:, 1], -1, 1), self.pedal).astype(np.float32)
        cooldown = np.maximum(self.cooldown - 1, 0)  # a shift at step t allows the next one at t + SHIFT_COOLDOWN
        choice = np.rint(action[:, 2]).astype(np.int64)
        if GEARBOX == "auto":
            choice = np.where(policy_shifts(raw[:, F["speed_x"]]), choice, auto_shift(self.gear, raw))
        gear, shifted, refused = shift_gear(self.gear, choice, cooldown, raw)
        self.gear = np.where(on, gear, self.gear)
        self.shifted, self.refused = shifted & on, refused & on
        self.cooldown = np.where(on, np.where(shifted, SHIFT_COOLDOWN, cooldown), self.cooldown)
        act[:, A["steer"]] = self.steer
        act[:, A["accel"]] = np.maximum(self.pedal, 0)
        act[:, A["brake"]] = np.maximum(-self.pedal, 0)
        act[:, A["gear"]] = self.gear


def shift_gear(gear: np.ndarray, choice: np.ndarray, cooldown: np.ndarray, raw: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The gearbox's rules. gear [n] selected (-1 = reverse, 1..TOP_GEAR), choice [n] DOWN / HOLD / UP -> new gear, and
    whether a shift happened or was refused. Up in top gear or down in reverse does nothing."""
    ready = cooldown <= 0
    up, down = ready & (choice == UP), ready & (choice == DOWN)
    stopped = np.abs(raw[:, F["speed_x"]]) < STOPPED_SPEED
    over_rev = raw[:, F["rpm_down"]] > raw[:, F["rpm_redline"]]
    new = gear.copy()
    new = np.where(up & (gear >= 1) & (gear < TOP_GEAR), gear + 1, new)
    new = np.where(up & (gear == -1) & stopped, 1, new)
    new = np.where(down & (gear >= 2) & ~over_rev, gear - 1, new)
    new = np.where(down & (gear == 1) & stopped, -1, new)
    refused = (up & (gear == -1) & ~stopped) | (down & (gear >= 2) & over_rev) | (down & (gear == 1) & ~stopped)
    return new, new != gear, refused


def auto_shift(gear: np.ndarray, raw: np.ndarray, up_at: float = 0.92, down_below: float = 0.7) -> np.ndarray:
    """A simple automatic gearbox, as shift choices: up at `up_at` of the redline, down when the lower gear would turn
    below `down_below` of it. A baseline for scripted drivers and tests, not used in training."""
    red = np.maximum(raw[:, F["rpm_redline"]], 1.0)
    up = (gear >= 1) & (raw[:, F["rpm"]] >= up_at * red)
    down = (gear >= 2) & (raw[:, F["rpm_down"]] < down_below * red)
    return np.where(up, UP, np.where(down, DOWN, HOLD))
