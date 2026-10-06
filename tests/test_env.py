"""Integration tests: need the native build installed (cmake --install native/build)."""

import numpy as np
import pytest

from rlpilot import EnvConfig, SpeedDreamsVecEnv
from rlpilot._native import F, N_FIELDS, N_RAYS


def follow_centreline(env):
    raw = env._raw
    steer = 1.5 * (raw[:, F["angle"]] - 0.5 * raw[:, F["to_middle"]] / np.maximum(raw[:, F["track_width"]], 1))
    target = 10 + 0.18 * raw[:, N_FIELDS + N_RAYS // 2]  # slow enough to complete laps
    pedal = np.where(raw[:, F["speed_x"]] < target, 1.0, np.where(raw[:, F["speed_x"]] > target + 3, -0.6, 0))
    return np.stack([steer, pedal], 1).astype(np.float32)


@pytest.fixture(scope="module")
def env():
    e = SpeedDreamsVecEnv(EnvConfig(n_workers=2, max_episode_seconds=20))
    yield e
    e.close()


def test_spaces_and_finite(env):
    obs, _ = env.reset()
    assert obs.shape == (2, env.single_observation_space.shape[0])
    for _ in range(50):
        obs, rew, term, trunc, _ = env.step(follow_centreline(env))
        assert np.isfinite(obs).all() and np.isfinite(rew).all()


def test_deterministic_across_workers_and_resets(env):
    def rollout():
        env.reset()
        traj = []
        for _ in range(300):
            obs, *_ = env.step(follow_centreline(env))
            traj.append(obs.copy())
        return np.stack(traj)

    a, b = rollout(), rollout()
    np.testing.assert_array_equal(a, b)              # same after a reset
    np.testing.assert_array_equal(a[:, 0], a[:, 1])  # same in both workers


def test_rays_match_lateral_distances(env):
    env.reset()
    for _ in range(200):
        env.step(follow_centreline(env))
    raw = env._raw
    aligned = np.abs(raw[:, F["angle"]]) < 0.02
    assert aligned.any()
    left, right = raw[:, N_FIELDS + N_RAYS - 1], raw[:, N_FIELDS]  # +90 and -90 degrees
    np.testing.assert_allclose(left[aligned], raw[aligned, F["to_left"]], atol=0.3)
    np.testing.assert_allclose(right[aligned], raw[aligned, F["to_right"]], atol=0.3)


def test_truncation_and_autoreset(env):
    env.reset()
    steps = int(round(20 / env.cfg.step_seconds))
    for i in range(steps):
        obs, _, term, trunc, info = env.step(follow_centreline(env))
    assert trunc.all() and not term.any()
    assert info["_final_obs"].all() and (info["episode"]["l"] == steps).all()
    assert (env._raw[:, F["dist_raced"]] < 0).all()  # back on the grid, behind the start line


def test_reference_bot():
    from rlpilot import reference
    ref = reference.load("rlref")
    assert len(ref.lap_times) == 3 and 80 < ref.best_lap < 150
    assert 3000 < ref.track_length < 5000
    lap = ref.lap_trace()
    assert lap["dist_from_start"].max() > 0.98 * ref.track_length


def test_lap_done_and_reward_components():
    from rlpilot.reward import REWARD_COMPONENTS
    e = SpeedDreamsVecEnv(EnvConfig(n_workers=1, laps_per_episode=1, max_episode_seconds=400))
    try:
        e.reset()
        for _ in range(int(400 / e.cfg.step_seconds)):
            _, _, term, trunc, info = e.step(follow_centreline(e))
            if info:
                break
        st = info["stats"]
        assert term[0] and e.term_reasons[st["term_reason"][0]] == "lap_done"
        assert st["laps"][0] == 1 and abs(st["progress"][0] - 1.0) < 0.02
        assert st["reward_terminal"][0] == 0  # finishing is not penalized
        total = sum(st[f"reward_{c}"][0] for c in REWARD_COMPONENTS)
        assert abs(total - info["episode"]["r"][0]) < 1e-3
        assert 0 <= st["full_throttle_frac"][0] <= 1 and st["mean_speed"][0] > 10
    finally:
        e.close()


def test_worker_recycling_is_equivalent_to_reset():
    def episodes(recycle_resets):
        e = SpeedDreamsVecEnv(EnvConfig(n_workers=2, max_episode_seconds=4, recycle_resets=recycle_resets))
        try:
            e.reset()
            pids0 = [w.pid for w in e.client.workers]
            out = []
            for _ in range(4 * int(4 / e.cfg.step_seconds)):  # 4 episodes
                obs, *_ = e.step(follow_centreline(e))
                out.append(obs.copy())
            return np.stack(out), pids0 != [w.pid for w in e.client.workers]
        finally:
            e.close()

    normal, respawned_normal = episodes(10_000)
    recycled, respawned = episodes(2)
    assert respawned and not respawned_normal
    np.testing.assert_array_equal(normal, recycled)


def test_shift_points_gearbox():
    """gearbox="points": the two extra actions move the upshift point (early vs at the limiter)."""
    e = SpeedDreamsVecEnv(EnvConfig(n_workers=2, gearbox="points", max_episode_seconds=30))
    try:
        assert e.action_names == ("steer", "pedal", "shift_up", "shift_down")
        e.reset()
        red = e._raw[0, F["rpm_redline"]]
        points = np.array([[-1, -1], [1, -1]], np.float32)  # upshift at 70% / 97% of redline
        prev_gear, prev_rpm = e._raw[:, F["gear"]].copy(), e._raw[:, F["rpm"]].copy()
        at_upshift = [[], []]
        for _ in range(400):
            e.step(np.concatenate([follow_centreline(e), points], 1))
            gear = e._raw[:, F["gear"]]
            for i in np.flatnonzero(gear > prev_gear):
                at_upshift[i].append(prev_rpm[i] / red)
            prev_gear, prev_rpm = gear.copy(), e._raw[:, F["rpm"]].copy()
        assert at_upshift[0] and at_upshift[1]
        assert max(at_upshift[0]) < 0.8 and min(at_upshift[1]) > 0.9
    finally:
        e.close()
