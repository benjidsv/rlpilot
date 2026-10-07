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


@pytest.mark.parametrize("stat", ["mean", "peak"])
def test_track_profile_on_a_circle(stat):
    from rlpilot.track import CURVATURE_SCALE, TrackProfile
    r, n = 100.0, 2000
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)  # counter-clockwise = turning left
    circle = lambda rad: np.stack([rad * np.cos(a), rad * np.sin(a)], 1)
    p = TrackProfile(circle(r + 5), circle(r - 5), length=2 * np.pi * r, stat=stat)
    assert abs(p.total_turn - 2 * np.pi) < 1e-6
    look = p.lookahead(np.array([0.0, 300.0, 600.0]))
    np.testing.assert_allclose(look, CURVATURE_SCALE / r, rtol=1e-3)
    np.testing.assert_allclose(p.table[::50], CURVATURE_SCALE / r, rtol=1e-3)


def test_track_profile_peak_sees_short_corners():
    """A rounded square (200 m straights, 20 m radius corners): in the 50 m windows far ahead, the mean dilutes
    the 31 m corner while the peak reads its full curvature."""
    from rlpilot.track import CURVATURE_SCALE, TrackProfile
    r, side = 20.0, 200.0
    pts = []
    for q in range(4):  # counter-clockwise: straight along the heading, then a left quarter circle
        h = q * np.pi / 2
        d, c = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
        start = pts[-1] + d if pts else np.zeros(2)
        pts += [start + d * t for t in np.arange(0, side, 1.0)]
        centre = pts[-1] + d + c * r
        pts += [centre - c * r * np.cos(u) + d * r * np.sin(u) for u in np.linspace(0, np.pi / 2, 32)[1:-1]]
    mid = np.array(pts)
    seg = np.diff(np.vstack([mid, mid[:1]]), axis=0)
    normal = np.stack([-seg[:, 1], seg[:, 0]], 1) / np.hypot(seg[:, 0], seg[:, 1])[:, None]
    length = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
    peak, mean = (TrackProfile(mid + 5 * normal, mid - 5 * normal, length, stat=s) for s in ("peak", "mean"))
    s0 = np.array([side - 275.0 + length])  # the first corner is 250-300 m ahead
    assert peak.lookahead(s0)[0, -1] > 0.9 * CURVATURE_SCALE / r
    assert mean.lookahead(s0)[0, -1] < 0.7 * CURVATURE_SCALE / r
    assert np.abs(peak.lookahead(np.array([side / 2]))[0, :4]).max() < 0.05  # the next 75 m are straight


def test_steer_rate_penalty():
    from rlpilot.config import RewardConfig
    from rlpilot.reward import REWARD_COMPONENTS, RewardTracker
    raw = np.zeros((2, N_FIELDS + N_RAYS), np.float32)
    raw[:, F["speed_x"]] = 20.0
    t = RewardTracker(2, RewardConfig(steer_rate=0.1), step_seconds=0.04)
    t.reset(raw)
    t.step(raw, np.array([0.0, 0.5], np.float32))
    steer = t.components[REWARD_COMPONENTS.index("steer")]
    np.testing.assert_allclose(steer, [0.0, -0.025])  # 0.1 x 0.5^2


def test_multi_track_switching_and_lookahead():
    from rlpilot.track import race_for
    races = (race_for("jarama"), race_for("braga"))
    e = SpeedDreamsVecEnv(EnvConfig(n_workers=2, races=races, lookahead=True, max_episode_seconds=2,
                                    track_switch_steps=1))
    try:
        assert e.single_observation_space.shape[0] == 31 + 2 + 9
        e.reset()
        seen = []
        for _ in range(6 * int(2 / e.cfg.step_seconds)):
            obs, *_ = e.step(follow_centreline(e))
            assert np.isfinite(obs).all()
            seen.append(tuple(e.client.races))
        assert len(set(seen)) > 1  # workers changed track
        # the lookahead follows each worker's current track
        assert set(r for s in seen for r in s) <= set(races)
        np.testing.assert_array_equal(obs[:, -9:], e._lookahead())
    finally:
        e.close()


def test_car_factors_change_physics_and_obs():
    def speed_after(**kw):  # full throttle from the grid, 4 s
        e = SpeedDreamsVecEnv(EnvConfig(n_workers=1, max_episode_seconds=10, **kw))
        try:
            e.reset()
            for _ in range(100):
                obs, *_ = e.step(np.array([[0.0, 1.0]], np.float32))
            return e._raw[0, F["speed_x"]], obs, e
        finally:
            e.close()

    stock, _, _ = speed_after()
    heavy, _, _ = speed_after(car_mass=(1.2, 1.2))
    weak, _, _ = speed_after(car_power=(0.8, 0.8))
    assert heavy < stock - 1 and weak < stock - 1
    _, obs, e = speed_after(car_mass=(1.1, 1.1), car_obs=True)
    np.testing.assert_allclose(obs[0, -3:], [0.0, 1.0, 0.0], atol=1e-5)  # (factor - 1) x 10
