import numpy as np

from rlpilot import obs as O
from rlpilot.controls import DOWN, HOLD, STEER_RATE, TOP_GEAR, UP

I = {n: i for i, n in enumerate(O.NAMES)}


def scripted(obs: np.ndarray) -> np.ndarray:
    """A crude driver from observations only: steer to the road's direction and middle, slow down when the road ahead bends,
    shift on rpm. Deterministic; for tests, not a baseline worth beating."""
    angle, mid = obs[:, I["angle / pi"]] * np.pi, obs[:, I["to_middle / half width"]]
    target = np.clip(1.2 * angle - 0.3 * mid, -1, 1)
    steer = np.clip((target - obs[:, I["steer command"]]) / STEER_RATE, -1, 1)
    # how far the road runs straight ahead: the last point whose centre is within 3 m of the car's heading
    ahead_points = [d for d in O.POINTS if d >= 0]
    mid = np.stack([obs[:, I[f"left y / 50 @{d:g}m"]] + obs[:, I[f"right y / 50 @{d:g}m"]] for d in ahead_points], 1) * 25
    straight = np.cumprod(np.abs(mid) < 3, axis=1)
    v, ahead = obs[:, I["speed_x / 50"]] * 50, np.minimum(np.max(straight * np.array(ahead_points), axis=1), 200)
    pedal = np.where(v < 8 + 0.3 * ahead, 0.7, -0.6)
    rpm, down = obs[:, I["rpm / redline"]], obs[:, I["rpm_down / redline"]]
    gear = np.array([-1] + list(range(1, TOP_GEAR + 1)))[np.argmax(obs[:, I["gear R"]:I["gear R"] + TOP_GEAR + 1], 1)]
    shift = np.where((rpm > 0.92) & (gear >= 1), UP, np.where((gear >= 2) & (down < 0.7), DOWN, HOLD))
    return np.stack([steer, pedal, shift], 1).astype(np.float32)
