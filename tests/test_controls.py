"""The gearbox and steering rules, without the simulator."""
import numpy as np

from rlpilot.client import ACT_DIM, OBS_DIM, A, F
from rlpilot.controls import DOWN, HOLD, SHIFT_COOLDOWN, STEER_RATE, TOP_GEAR, UP, Controls, shift_gear


def raw(speed=20.0, rpm_down=500.0, redline=900.0, n=1):
    r = np.zeros((n, OBS_DIM), np.float32)
    r[:, F["speed_x"]], r[:, F["rpm_down"]], r[:, F["rpm_redline"]] = speed, rpm_down, redline
    return r


def shift(gear, choice, **kw):
    g, shifted, refused = shift_gear(np.array([gear]), np.array([choice]), np.array([0]), raw(**kw))
    return int(g[0]), bool(shifted[0]), bool(refused[0])


def test_sequential_shifts():
    assert shift(3, UP) == (4, True, False)
    assert shift(3, DOWN) == (2, True, False)
    assert shift(3, HOLD) == (3, False, False)
    assert shift(TOP_GEAR, UP) == (TOP_GEAR, False, False)  # nothing above top: a no-op, not a refusal


def test_over_rev_downshift_refused():
    assert shift(3, DOWN, rpm_down=950.0, redline=900.0) == (3, False, True)


def test_reverse_only_when_stopped():
    assert shift(1, DOWN, speed=0.5) == (-1, True, False)
    assert shift(1, DOWN, speed=5.0) == (1, False, True)
    assert shift(-1, UP, speed=-0.5) == (1, True, False)
    assert shift(-1, UP, speed=-3.0) == (-1, False, True)
    assert shift(-1, DOWN, speed=0.0) == (-1, False, False)


def test_cooldown_and_steer_rate():
    c, act = Controls(1), np.zeros((1, ACT_DIM), np.float32)
    c.reset(np.array([0]), raw())
    c.gear[:] = 3
    c.apply(np.array([[1.0, 0.5, UP]]), raw(), act)
    assert c.gear[0] == 4 and c.cooldown[0] == SHIFT_COOLDOWN
    assert act[0, A["gear"]] == 4 and act[0, A["accel"]] == 0.5 and act[0, A["brake"]] == 0
    assert np.isclose(act[0, A["steer"]], STEER_RATE)
    for _ in range(SHIFT_COOLDOWN - 1):  # no shift until the cooldown has run out
        c.apply(np.array([[0.0, -1.0, UP]]), raw(), act)
        assert c.gear[0] == 4
    assert act[0, A["brake"]] == 1 and act[0, A["accel"]] == 0
    c.apply(np.array([[0.0, 0.0, UP]]), raw(), act)
    assert c.gear[0] == 5


def test_paused_envs_keep_their_state():
    c, act = Controls(2), np.zeros((2, ACT_DIM), np.float32)
    c.reset(np.array([0, 1]), raw(n=2))
    c.gear[:] = 2
    c.apply(np.array([[1.0, 1.0, UP]] * 2), raw(n=2), act, active=np.array([True, False]))
    assert list(c.gear) == [3, 2] and c.steer[1] == 0 and c.cooldown[1] == 0
