import numpy as np

from rlpilot import obs as O
from rlpilot.controls import DOWN, HOLD, STEER_RATE, TOP_GEAR, UP

I = {n: i for i, n in enumerate(O.NAMES)}


def scripted(obs: np.ndarray) -> np.ndarray:
    """A crude driver from observations only: steer to the road's direction and middle, slow down for walls ahead,
    shift on rpm. Deterministic; for tests, not a baseline worth beating."""
    angle, mid = obs[:, I["angle / pi"]] * np.pi, obs[:, I["to_middle / half width"]]
    target = np.clip(1.2 * angle - 0.3 * mid, -1, 1)
    steer = np.clip((target - obs[:, I["steer command"]]) / STEER_RATE, -1, 1)
    v, ahead = obs[:, I["speed_x / 50"]] * 50, obs[:, I["ray_9 / 200"]] * 200
    pedal = np.where(v < 8 + 0.3 * ahead, 0.7, -0.6)
    rpm, down = obs[:, I["rpm / redline"]], obs[:, I["rpm_down / redline"]]
    gear = np.array([-1] + list(range(1, TOP_GEAR + 1)))[np.argmax(obs[:, I["gear R"]:I["gear R"] + TOP_GEAR + 1], 1)]
    shift = np.where((rpm > 0.92) & (gear >= 1), UP, np.where((gear >= 2) & (down < 0.7), DOWN, HOLD))
    return np.stack([steer, pedal, shift], 1).astype(np.float32)
