"""Termination clocks as observations (--timer-obs), checked on the real engine against the rules that end episodes."""

import numpy as np

from rlpilot import EnvConfig, RewardConfig, SpeedDreamsVecEnv
from rlpilot._native import F
from rlpilot.features import TIMER_NAMES, make_layout
from rlpilot.reward import TERM_REASONS
from rlpilot.track import race_for

LIMIT = 0.0625


def test_layout_adds_the_timers_last():
    base = make_layout(("steer", "pedal"), accel=True)
    lay = make_layout(("steer", "pedal"), accel=True, timers=True)
    assert lay.names[:base.dim] == base.names and lay.names[-3:] == TIMER_NAMES


def test_offtrack_clock_counts_up_resets_and_reaches_1_when_the_episode_ends():
    patience = 2.0
    env = SpeedDreamsVecEnv(EnvConfig(n_workers=1, races=(race_for("jarama"),), gearbox="points", steer_limit=LIMIT,
                                      timer_obs=True, spin_us=50, max_episode_seconds=60,
                                      reward=RewardConfig(offtrack_patience=patience)))
    dt = env.cfg.step_seconds
    try:
        obs, _ = env.reset()
        assert np.all(env.obs_layout.group(obs[0], "timers") == 0)
        stuck = [env.obs_layout.get(obs[0], "stuck_timer")]
        clock, off, ended = [], [], None
        for step in range(1500):
            a = np.zeros((1, env.act_dim), np.float32)
            a[0, 1] = 0.5
            if step > 100:  # at speed, steer off the road to the left and keep going
                a[0, 0] = 1.0
            obs, _, term, trunc, info = env.step(a)
            if step < 10:
                stuck.append(env.obs_layout.get(obs[0], "stuck_timer"))
            if term[0] or trunc[0]:
                ended = int(info["stats"]["term_reason"][0])
                final = info["final_obs"][0]
                clock.append(env.obs_layout.get(final, "offtrack_timer"))
                off.append(info["final_raw"][0][F["wheels_off"]] >= 4)
                break
            clock.append(env.obs_layout.get(obs[0], "offtrack_timer"))
            off.append(env._raw[0, F["wheels_off"]] >= 4)
    finally:
        env.close()
    clock, off = np.array(clock), np.array(off)
    assert ended == TERM_REASONS.index("offtrack")
    assert clock[-1] == 1.0
    # the clock is exactly the run of consecutive off-road steps so far, over the patience
    run = np.zeros(len(off))
    for i in range(len(off)):
        run[i] = (run[i - 1] + 1 if i else 1) if off[i] else 0
    np.testing.assert_allclose(clock, np.minimum(run * dt / patience, 1.0), atol=1e-6)
    assert np.all(clock[~off] == 0) and np.all(clock[(run > 0) & (clock < 1)] > 0)
    # standing still on the grid runs the stuck clock: 4 s of patience, one step is 0.01
    assert stuck[1] > 0 and np.isclose(stuck[1], dt / 4.0)
