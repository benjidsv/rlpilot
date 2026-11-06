"""Acceleration observations (--accel-obs), checked against the engine's own speeds.

Measured 2026-11-06 on Jarama: under steady throttle and braking, accel_x vs the backward difference of speed_x has a
correlation of 0.99; over a lap of driving, accel_y vs speed_x x yaw_rate (the centripetal acceleration) 0.97-0.98, slope
0.91-0.94. With the test controller's pedal, which flips between steps, accel_x is spikier (it's the last tick's, not the
step's mean), so the longitudinal check uses steady pedals.
"""

import numpy as np

from rlpilot import EnvConfig, SpeedDreamsVecEnv
from rlpilot._native import F
from rlpilot.features import ACCEL_SCALE, make_layout
from rlpilot.track import race_for

LIMIT = 0.0625


def env_for(track: str, **kw) -> SpeedDreamsVecEnv:
    return SpeedDreamsVecEnv(EnvConfig(n_workers=1, races=(race_for(track),), gearbox="points", steer_limit=LIMIT,
                                       accel_obs=True, spin_us=50, **kw))


def hold_the_road(env, pedal: float, offset_gain: float = 0.5) -> np.ndarray:
    raw = env._raw
    want = 1.5 * (raw[:, F["angle"]] - offset_gain * raw[:, F["to_middle"]] / np.maximum(raw[:, F["track_width"]], 1))
    a = np.zeros((1, env.act_dim), np.float32)
    a[0, 0] = np.clip((want[0] - env._prev_action[0, 0]) / LIMIT, -1, 1)
    a[0, 1] = pedal
    return a


def test_layout_adds_three_slots_at_the_end():
    base = make_layout(("steer", "pedal"), height_points=29)
    with_accel = make_layout(("steer", "pedal"), height_points=29, accel=True)
    assert with_accel.dim == base.dim + 3
    assert with_accel.names[:base.dim] == base.names
    assert with_accel.names[-3:] == ("accel_x", "accel_y", "accel_z")


def test_longitudinal_matches_the_change_of_speed_and_the_obs_is_scaled():
    env = env_for("jarama", max_episode_seconds=60)
    try:
        env.reset()
        raws, accel = [], []
        for step in range(260):  # full throttle, then full braking, then a little throttle
            pedal = 1.0 if step < 150 else (-1.0 if step < 200 else 0.3)
            obs, *_ = env.step(hold_the_road(env, pedal))
            raws.append(env._raw[0].copy())
            accel.append(env.obs_layout.group(obs[0], "accel"))
    finally:
        env.close()
    r, accel = np.array(raws), np.array(accel)
    np.testing.assert_allclose(accel, r[:, [F["accel_x"], F["accel_y"], F["accel_z"]]] / ACCEL_SCALE, rtol=1e-6)
    ax, dv = r[1:, F["accel_x"]], np.diff(r[:, F["speed_x"]]) / env.cfg.step_seconds
    assert np.corrcoef(ax, dv)[0, 1] > 0.97
    assert ax[20:140].mean() > 2 and ax[155:195].mean() < -8  # + = forward: positive under throttle, negative braking
    assert np.abs(r[:, F["accel_z"]]).max() < 15


def test_lateral_is_the_centripetal_acceleration_with_left_positive():
    env = env_for("corkscrew", max_episode_seconds=80)
    try:
        env.reset()
        raws = []
        for step in range(1500):
            speed = env._raw[0, F["speed_x"]]
            env.step(hold_the_road(env, 0.6 if speed < 22 else -0.3))
            raws.append(env._raw[0].copy())
    finally:
        env.close()
    r = np.array(raws)
    centripetal = r[:, F["speed_x"]] * r[:, F["yaw_rate"]]  # turning left (+ yaw rate) pulls the car to the left (+)
    assert np.abs(centripetal).max() > 5
    assert np.corrcoef(r[:, F["accel_y"]], centripetal)[0, 1] > 0.9
    assert 0.8 < np.polyfit(centripetal, r[:, F["accel_y"]], 1)[0] < 1.1
