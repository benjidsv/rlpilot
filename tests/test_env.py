"""Integration tests: need the native build installed (cmake --install native/build)."""

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rlpilot import EnvConfig, SpeedDreamsVecEnv
from rlpilot import tracking
from rlpilot._native import CMD_STEP, A, MAX_WORKERS, Client, F, N_FIELDS, N_RAYS
from rlpilot.evaluate import eval_config
from rlpilot.sb3 import resume_ppo


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
        assert st["abs_offset"][0] < 0.3  # follow_centreline stays near the middle
        assert 0 < st["line_dev"][0] < 6  # ... so it is off the bot's line, by less than a track width
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
        assert e.single_observation_space.shape[0] == e.obs_layout.dim
        look = e.obs_layout.slices["lookahead"]
        e.reset()
        seen = []
        for _ in range(6 * int(2 / e.cfg.step_seconds)):
            obs, *_ = e.step(follow_centreline(e))
            assert np.isfinite(obs).all()
            seen.append(tuple(e.client.races))
        assert len(set(seen)) > 1  # workers changed track
        # the lookahead follows each worker's current track
        assert set(r for s in seen for r in s) <= set(races)
        np.testing.assert_array_equal(obs[:, look], e._look(e._raw[:, F["dist_from_start"]], e._look_off, e._look_size))
    finally:
        e.close()


def test_car_factors_change_physics_and_obs():
    def speed_after(**kw):  # full throttle from the grid, 4 s
        e = SpeedDreamsVecEnv(EnvConfig(n_workers=1, max_episode_seconds=10, **kw))
        try:
            e.reset()
            for _ in range(100):
                obs, *_ = e.step(np.array([[0.0, 1.0]], np.float32))
            return e._raw[0, F["speed_x"]], e.obs_layout.group(obs, "car") if kw.get("car_obs") else None, e
        finally:
            e.close()

    stock, _, _ = speed_after()
    heavy, _, _ = speed_after(car_mass=(1.2, 1.2))
    weak, _, _ = speed_after(car_power=(0.8, 0.8))
    assert heavy < stock - 1 and weak < stock - 1
    _, obs, e = speed_after(car_mass=(1.1, 1.1), car_obs=True)
    np.testing.assert_allclose(obs[0], [0.0, 1.0, 0.0], atol=1e-5)  # (factor - 1) x 10


def test_paused_envs_are_not_simulated():
    """step(active=...): left-out envs keep their state and cannot end (evaluation pauses finished episodes)."""
    e = SpeedDreamsVecEnv(EnvConfig(n_workers=2, max_episode_seconds=1))
    try:
        e.reset()
        active = np.array([True, False])
        frozen = e._raw[1].copy()
        ended = np.zeros(2, bool)
        for _ in range(3 * int(1 / e.cfg.step_seconds)):  # three time limits
            _, _, term, trunc, _ = e.step(follow_centreline(e), active)
            ended |= term | trunc
        assert ended[0] and not ended[1]
        np.testing.assert_array_equal(e._raw[1], frozen)
    finally:
        e.close()


def test_upshift_hold_stops_flicked_shift_points():
    """proto-5's pattern: the upshift point flicked between floor and ceiling with the downshift point at its max
    upshifts early, then downshifts straight back under throttle. With the hold, an upshift stays until the car
    slows down."""
    def downshifts_while_accelerating(gap):
        e = SpeedDreamsVecEnv(EnvConfig(n_workers=1, gearbox="points", max_episode_seconds=30, upshift_hold_gap=gap))
        try:
            e.reset()
            n = 0
            for t in range(500):
                gear, speed = e._raw[0, F["gear"]], e._raw[0, F["speed_x"]]
                flick = np.array([[-1.0 if t % 2 else 1.0, 1.0]], np.float32)
                e.step(np.concatenate([follow_centreline(e), flick], 1))
                n += e._raw[0, F["gear"]] < gear and e._raw[0, F["speed_x"]] > speed
            return n
        finally:
            e.close()

    assert downshifts_while_accelerating(None) >= 5
    assert downshifts_while_accelerating(0.05) == 0


def test_racing_line_matches_engine_lateral_offset():
    """RacingLine.lateral (from x, y) agrees with the engine's to_middle, sign included, so the bot's line and the
    car's offset are in the same frame."""
    from rlpilot.track import RacingLine, race_for
    e = SpeedDreamsVecEnv(EnvConfig(n_workers=1, max_episode_seconds=60))
    try:
        e.reset()
        line = RacingLine.for_race(race_for("jarama"))
        errs, offsets = [], []
        for t in range(600):
            a = follow_centreline(e)
            a[:, 0] += 0.15 * np.sin(t / 15)  # weave so the offset spans both sides
            e.step(a)
            r = e._raw[0]
            if r[F["speed_x"]] > 5:
                lat = line.lateral(r[F["pos_x"]], r[F["pos_y"]], r[F["dist_from_start"]])
                errs.append(lat - r[F["to_middle"]])
                offsets.append(r[F["to_middle"]])
        assert min(offsets) < -0.5 and max(offsets) > 0.5
        assert np.percentile(np.abs(errs), 95) < 0.3
    finally:
        e.close()


def test_line_shaping_telescopes():
    """Potential-based shaping sums to -Phi(start) over an episode whatever the path (gamma 1, terminal Phi = 0)."""
    from rlpilot.config import RewardConfig
    from rlpilot.reward import REWARD_COMPONENTS, RewardTracker
    k, cap = 0.5, 3.0
    raw = np.zeros((1, N_FIELDS + N_RAYS), np.float32)
    raw[:, F["speed_x"]] = 20.0
    for path in ([1.0, 2.0, 1.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 4.0, 4.0, 2.5, 6.0]):
        t = RewardTracker(1, RewardConfig(line_shaping=k, line_cap=cap, gamma=1.0, max_damage=1e9), 0.04)
        t.reset(raw, line_dev=np.array([path[0]], np.float32))
        total = 0.0
        for i, dev in enumerate(path[1:]):
            last = i == len(path) - 2
            raw[:, F["race_over"]] = float(last)  # ends the episode (terminated) on the last step
            t.step(raw, 0.0, np.array([dev], np.float32))
            total += t.components[REWARD_COMPONENTS.index("line")][0]
        raw[:, F["race_over"]] = 0
        np.testing.assert_allclose(total, k * min(path[0], cap), atol=1e-6)


def test_line_obs_follows_track_switches():
    from rlpilot.track import LINE_POINTS, race_for
    races = (race_for("jarama"), race_for("braga"))
    e = SpeedDreamsVecEnv(EnvConfig(n_workers=2, races=races, line_obs=True, max_episode_seconds=2,
                                    track_switch_steps=1))
    try:
        assert e.obs_layout.group(np.zeros(e.obs_layout.dim), "line").shape == (len(LINE_POINTS),)
        e.reset()
        for _ in range(4 * int(2 / e.cfg.step_seconds)):
            obs, *_ = e.step(follow_centreline(e))
            assert np.isfinite(obs).all()
        np.testing.assert_array_equal(e.obs_layout.group(obs, "line"), e._line_obs())
        for i, race in enumerate(e.client.races):
            assert e._line_off[i] == e._line.offset[race]
    finally:
        e.close()


def test_client_rejects_more_workers_than_exchange_commands():
    """rlp_exchange only commands the first 256 workers, and the others would silently never step."""
    with pytest.raises(ValueError, match="at most 256"):
        Client(MAX_WORKERS + 1)


class _FakeMlflow:
    def __init__(self):
        self.best = []  # step of the eval saved to best/, at each save

    def log_artifacts(self, path, artifact_path):
        assert artifact_path == "best"
        self.best.append(int(Path(path).name))

    def log_metrics(self, *args, **kwargs):
        pass

    def set_tag(self, *args):
        pass


def _select(monkeypatch, tmp_path, scores, smooth=3):
    """The steps saved to best/ over evals with these scores (steps 1, 2, ...), and the callback."""
    fake = _FakeMlflow()
    monkeypatch.setattr(tracking, "mlflow", fake)
    monkeypatch.setattr(tracking, "_save_local", lambda model, d: d.mkdir(parents=True))
    cb = tracking.EvalCallback(EnvConfig(), smooth=smooth)
    cb.model, cb._tmp = None, SimpleNamespace(name=str(tmp_path))
    for step, score in enumerate(scores, 1):
        cb._select_best(step, score)
    return fake.best, cb


def test_a_lucky_early_eval_cant_hold_best(monkeypatch, tmp_path):
    # 0.9 alone would be a window of 1 beating every full window; the later plateau has the best full window (0.4)
    saved, cb = _select(monkeypatch, tmp_path, [0.9, 0.1, 0.1, 0.4, 0.4, 0.4])
    assert saved == [1, 1, 1, 4]  # provisional, provisional, first full window, plateau
    assert cb.best_score == pytest.approx(0.4)


def test_first_full_window_replaces_a_better_provisional_best(monkeypatch, tmp_path):
    saved, cb = _select(monkeypatch, tmp_path, [0.2, 0.9, 0.1, 0.1])
    assert saved == [1, 2, 2]  # the last eval's window (0.37) doesn't beat the first full one (0.4)
    assert cb.best_score == pytest.approx(1.2 / 3)


def test_run_shorter_than_the_window_still_has_a_best(monkeypatch, tmp_path):
    saved, cb = _select(monkeypatch, tmp_path, [0.2, 0.5])
    assert saved == [1, 2]
    assert cb.best_score == -np.inf


def test_window_of_one_keeps_every_new_best(monkeypatch, tmp_path):
    assert _select(monkeypatch, tmp_path, [0.2, 0.1, 0.5], smooth=1)[0] == [1, 3]


def test_track_metrics_layout(monkeypatch):
    monkeypatch.setattr(tracking, "track_of", lambda race: race.split("-")[1])
    cb = tracking.EvalCallback(EnvConfig(races=("rlpilot-a", "rlpilot-b"), n_workers=2), heldout=("rlpilot-c",))
    cb.refs = {r: SimpleNamespace(best_lap=100.0) for r in cb.races}
    base = {"eval/progress": 1.0, "eval/completion_rate": 1.0, "eval/laps_completed": 3.0, "eval/completion_strict": 1.0,
            "eval/laps_completed_strict": 3.0, "eval/behavior/line_dev": 2.0, "eval/behavior/abs_offset": 0.5}
    flying = base | {"eval/flying_lap_time_median": 110.0, "eval/flying_lap_time_median_strict": 110.0}
    crashed = base | {"eval/progress": 0.5, "eval/fail_dist_from_start": 42.0}
    out, scores = cb._track_metrics({"rlpilot-a": flying, "rlpilot-b": crashed, "rlpilot-c": flying})
    assert out["track_gap_pct/a"] == pytest.approx(10.0) and "track_gap_pct/b" not in out
    assert out["track_fail_dist/b"] == 42.0 and out["track_progress/b"] == 0.5
    assert out["dashboard/train_progress"] == pytest.approx(0.75) and out["dashboard/train_gap_pct"] == pytest.approx(10.0)
    assert out["dashboard/heldout_flying_lap_tracks"] == 1.0 and out["eval/train_flying_lap_tracks"] == 0.5
    assert out["eval/heldout_line_dev"] == 2.0
    assert out["dashboard/train_completion_strict"] == 1.0
    assert len(scores["train"]) == 2 and len(scores["heldout"]) == 1
    assert {k.split("/")[0] for k in out} == {"dashboard", "eval", "track_progress", "track_gap_pct", "track_fail_dist"}


def test_sb3_records_are_renamed_or_dropped(monkeypatch):
    logged = {}
    monkeypatch.setattr(tracking, "mlflow", SimpleNamespace(log_metrics=lambda m, step: logged.update(m)))
    tracking.MLflowOutputFormat().write({"train/entropy_loss": -2.0, "train/value_loss": 1.0, "time/fps": 5,
                                         "rollout/ep_rew_mean": 1.0, "train/loss": 1.0, "train/actor_loss": 3.0}, {})
    assert logged == {"dashboard/ppo_entropy": 2.0, "ppo/value_loss": 1.0, "sb3/train/actor_loss": 3.0}


def _dummy_venv():
    return DummyVecEnv([lambda: gym.make("Pendulum-v1")])


def test_resume_uses_the_settings_it_logs(tmp_path):
    saved = dict(n_steps=64, batch_size=32, n_epochs=2, gamma=0.995, gae_lambda=0.95, ent_coef=0.003, target_kl=0.02,
                 learning_rate=3e-4, clip_range=0.2, policy_kwargs=dict(net_arch=[16, 16]))
    venv = VecNormalize(_dummy_venv(), norm_obs=False, norm_reward=True, gamma=saved["gamma"])
    PPO("MlpPolicy", venv, **saved).save(tmp_path / "model.zip")
    venv.save(str(tmp_path / "vecnormalize.pkl"))

    new = saved | dict(n_steps=128, batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.9, ent_coef=0.01, target_kl=None,
                       learning_rate=1e-4, clip_range=0.1)
    model = resume_ppo(tmp_path, _dummy_venv(), "cpu", new)
    for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "target_kl"):
        assert getattr(model, k) == new[k], k
    assert model.rollout_buffer.buffer_size == 128
    assert model.rollout_buffer.gamma == 0.99
    assert model.lr_schedule(1.0) == pytest.approx(1e-4)
    assert model.clip_range(1.0) == pytest.approx(0.1)
    assert isinstance(model.get_env(), VecNormalize)
    assert model.get_env().gamma == 0.99
    assert model.get_env().training
    model.learn(128)  # the new sizes actually run


def test_resume_refuses_a_different_network(tmp_path):
    settings = dict(n_steps=64, batch_size=32, gamma=0.99, learning_rate=3e-4, clip_range=0.2, policy_kwargs=dict(net_arch=[16, 16]))
    venv = VecNormalize(_dummy_venv(), norm_obs=False, norm_reward=True, gamma=0.99)
    PPO("MlpPolicy", venv, **settings).save(tmp_path / "model.zip")
    venv.save(str(tmp_path / "vecnormalize.pkl"))
    with pytest.raises(ValueError, match="policy kwargs"):
        resume_ppo(tmp_path, _dummy_venv(), "cpu", settings | {"policy_kwargs": dict(net_arch=[32, 32])})


# ---- random launches ----

LAUNCH_SLOTS = [A[f"launch_{k}"] for k in ("pos", "lateral", "yaw", "speed")]


def _launch_env(**kw):
    return SpeedDreamsVecEnv(EnvConfig(**{"n_workers": 1, "race": "rlpilot-braga", "gearbox": "points", "steer_limit": 0.0625,
                                          "recycle_resets": 2**62} | kw))


def _trace(env, steps, pedal=0.5):
    """Scripted controls: the same fingerprint for any env that starts from the same state."""
    rows = []
    for _ in range(steps):
        a = np.zeros((env.num_envs, env.act_dim), np.float32)
        a[:, 1] = pedal
        _, r, *_ = env.step(a)
        rows.append(np.concatenate([env._raw[:, [F[k] for k in ("pos_x", "pos_y", "speed_x", "gear")]], r[:, None]], 1).copy())
    return np.stack(rows)


def test_launch_places_the_car():
    env = _launch_env(n_workers=2, races=("rlpilot-braga", "rlpilot-brno"), race="rlpilot-braga")
    try:
        length = env._raw[:, F["track_length"]].copy()
        env._act[:, LAUNCH_SLOTS] = [[0.3, 0.75, 0.05, 25.0], [0.6, 0.25, -0.05, 15.0]]
        env.reset()
        raw = env._raw
        assert abs(raw[0, F["dist_from_start"]] - 0.3 * length[0]) < 2 and abs(raw[1, F["dist_from_start"]] - 0.6 * length[1]) < 2
        assert 24 < raw[0, F["speed_x"]] <= 25 and 14 < raw[1, F["speed_x"]] <= 15
        assert np.abs(raw[:, F["speed_y"]]).max() < 0.1
        width = raw[:, F["track_width"]]  # to_middle: + = left; lateral counts from the right edge
        assert abs(raw[0, F["to_middle"]] - 0.25 * width[0]) < 0.5 and abs(raw[1, F["to_middle"]] + 0.25 * width[1]) < 0.5
        assert raw[0, F["gear"]] > 1  # a gear for the speed, not first
        assert raw[:, F["race_over"]].max() == 0
    finally:
        env.close()


def test_launch_angle_sign_matches_the_yaw_slot():
    env = _launch_env(n_workers=2)
    try:
        env._act[:, LAUNCH_SLOTS] = [[0.5, 0.5, 0.05, 0.0], [0.5, 0.5, -0.05, 0.0]]
        env.reset()
        # angle = track tangent - yaw: a car turned left of the road sees the road to its right
        assert env._raw[0, F["angle"]] == pytest.approx(-0.05, abs=0.01)
        assert env._raw[1, F["angle"]] == pytest.approx(0.05, abs=0.01)
    finally:
        env.close()


def test_grid_is_exact_after_launches():
    env = _launch_env()
    try:
        env.reset()
        first = _trace(env, 200)
        env._act[0, LAUNCH_SLOTS] = [0.4, 0.5, 0.0, 20.0]
        env.reset()
        assert env._raw[0, F["speed_x"]] > 15
        _trace(env, 50)
        env._act[0, LAUNCH_SLOTS] = 0
        env.reset()
        assert np.array_equal(_trace(env, 200), first)  # nothing of the launch survives the restart
    finally:
        env.close()


def test_launch_is_deterministic_and_changes_a_fresh_race():
    env = _launch_env()
    try:
        env.reset()
        grid_dist = env._raw[0, F["dist_from_start"]]
        env._act[0, LAUNCH_SLOTS] = [0.4, 0.5, 0.0, 20.0]
        env.reset()  # no step since the last start: a reset still has to restart, as the launch changed
        assert env._raw[0, F["dist_from_start"]] != grid_dist
        a = _trace(env, 100)
        env.reset()
        assert np.array_equal(_trace(env, 100), a)
    finally:
        env.close()


def test_launch_progress_counts_from_the_start_and_across_the_line():
    env = _launch_env()
    try:
        length = float(env._raw[0, F["track_length"]])
        env._act[0, LAUNCH_SLOTS] = [1 - 60 / length, 0.5, 0.0, 25.0]  # 60 m before the line, facing it
        env._launched[0] = True  # share 0, so that reset keeps these slots
        env.reset()
        assert env._rew.raced[0] == 0
        assert env._raw[0, F["laps"]] == 0
        total, laps = 0.0, []
        for _ in range(100):
            a = np.zeros((1, env.act_dim), np.float32)
            a[:, 1] = 0.5
            _, r, *_ = env.step(a)
            assert 0 <= env._rew.components[0, 0] < 0.5  # at most 12 m per step: no lap-sized jump at the line
            total += env._rew.components[0, 0]
            laps.append(env._raw[0, F["laps"]])
        assert laps[-1] == 1  # crossed it
        assert total == pytest.approx(0.1 * env._rew.raced[0], rel=1e-3)
        assert 60 < env._rew.raced[0] < 150
    finally:
        env.close()


def test_launch_share_follows_samples_per_worker():
    env = _launch_env(n_workers=4, races=("rlpilot-braga", "rlpilot-brno"), race="rlpilot-braga", launch_share=0.7,
                      max_episode_seconds=4.0)
    try:
        env.reset()
        starts = []
        for _ in range(3000):
            a = np.zeros((4, env.act_dim), np.float32)
            a[:, 1] = 0.3
            _, _, term, trunc, info = env.step(a)
            if info:
                starts += list(info["stats"]["launched"][term | trunc])
        share = env._launch_steps / env._mode_steps
        assert np.abs(share - 0.7).max() < 0.05  # every worker, so every track
        assert 0 < np.mean(starts) < 1
    finally:
        env.close()


def test_launch_share_zero_draws_nothing():
    env = _launch_env()
    try:
        before = env._rng.bit_generator.state
        env.reset()
        _trace(env, 10)
        assert env._rng.bit_generator.state == before
        assert (env._act[0, LAUNCH_SLOTS] == 0).all()
    finally:
        env.close()


def test_launch_share_one_is_never_the_grid_and_ranges_hold():
    env = _launch_env(n_workers=4, launch_share=1.0, launch_speed=(5.0, 20.0), max_episode_seconds=1.0)
    try:
        length = float(env._raw[0, F["track_length"]])
        env.reset()
        seen = []
        for _ in range(100):
            a = np.zeros((4, env.act_dim), np.float32)
            obs, *_, info = env.step(a)
            assert np.isfinite(obs).all()
            if info:
                assert (info["stats"]["launched"][info["_stats"]] == 1).all()
            seen.append(env._act[:, LAUNCH_SLOTS].copy())
        slots = np.concatenate(seen)
        assert (slots[:, 0] > 0).all() and (slots[:, 0] < 1).all()
        assert slots[:, 1].min() >= 0.1 and slots[:, 1].max() <= 0.9
        assert np.abs(slots[:, 2]).max() <= 0.1
        assert slots[:, 3].min() >= 5 and slots[:, 3].max() <= 20
        assert len(np.unique(slots[:, 0])) > 4 and length > 0
    finally:
        env.close()


def test_recycling_a_launched_worker_is_equivalent_to_a_reset():
    env = _launch_env()
    try:
        env._act[0, LAUNCH_SLOTS] = [0.4, 0.5, 0.02, 20.0]
        env.reset()
        a = _trace(env, 100)
        env.client.respawn(0)
        env._start_rewards(np.array([0]))
        env._clear(np.array([0]), 0.0)
        assert np.array_equal(_trace(env, 100), a)
    finally:
        env.close()


def test_eval_never_launches():
    cfg = EnvConfig(launch_share=0.7)
    assert eval_config(cfg, ("rlpilot", ), 2, 2).launch_share == 0


def test_a_launched_episodes_stats_say_so():
    env = _launch_env(launch_share=1.0, max_episode_seconds=1.0)
    try:
        env.reset()
        info = {}
        while not info:
            *_, info = env.step(np.zeros((1, env.act_dim), np.float32))
        s = info["stats"]
        assert s["launched"][0] == 1 and s["track"][0] == 0
        assert s["seconds"][0] == pytest.approx(1.0)
        assert 0 <= s["end_dist"][0] < env._raw[0, F["track_length"]]
        assert s["progress"][0] >= 0
    finally:
        env.close()


def test_throttle_and_brake_metrics_match_a_scripted_pedal():
    env = _launch_env(max_episode_seconds=0.24)  # 6 steps
    try:
        env.reset()
        pedals = [0.5, 0.5, -0.5, -0.05, -0.4, 0.2]
        for p in pedals:
            _, _, term, trunc, info = env.step(np.array([[0, p, 0, 0]], np.float32))
        assert trunc[0] and not term[0]
        st = info["stats"]
        n = len(pedals)
        assert st["throttle_delta"][0] == pytest.approx((0.5 + 0 + 0.5 + 0 + 0 + 0.2) / n)
        assert st["brake_delta"][0] == pytest.approx((0 + 0 + 0.5 + 0.45 + 0.35 + 0.4) / n)
        assert st["brake_frac"][0] == pytest.approx(2 / n)                 # -0.05 is below BRAKE_ON
        assert st["brake_onsets_per_s"][0] == pytest.approx(2 / (n * 0.04))  # applied twice
        env.reset()  # a new episode starts from clean counters, with the pedals at 0
        for _ in range(n):
            _, _, _, _, info = env.step(np.array([[0, 0.25, 0, 0]], np.float32))
        assert info["stats"]["throttle_delta"][0] == pytest.approx(0.25 / n)
        assert info["stats"]["brake_frac"][0] == 0 and info["stats"]["brake_onsets_per_s"][0] == 0
    finally:
        env.close()


BAD_LAUNCH = [0.989797, 0.100086, -0.094747, 2.55208]  # proto-8's crash at 130.9M: Racepark3, any speed


def _bad_launches(env, n):
    """The next n draws of env are the launch that comes back non-finite; then the normal draws."""
    draw, left = env._draw_start, [n]

    def patched(idx):
        if left[0] > 0:
            left[0] -= 1
            env._launched[idx] = True
            env._act[idx, A["launch_pos"]:A["launch_speed"] + 1] = BAD_LAUNCH
        else:
            draw(idx)
    env._draw_start = patched


@pytest.mark.parametrize("bad_draws, launched", [(1, True), (99, False)])
def test_a_non_finite_launch_is_redrawn_then_falls_back_to_the_grid(bad_draws, launched):
    env = _launch_env(race="rlpilot-racepark3", launch_share=1.0)
    try:
        assert env.reset()[0].shape  # a normal draw first, with a worker that has run
        env.start_retries = 0
        _bad_launches(env, bad_draws)
        obs, _ = env.reset()
        assert np.isfinite(obs).all() and np.isfinite(env._raw).all()
        assert env.start_retries == (1 if bad_draws == 1 else 5)
        assert env._launched[0] == launched
        obs, reward, *_ = env.step(np.zeros((1, 4), np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
    finally:
        env.close()


def test_autoreset_recovers_from_a_non_finite_launch():
    env = _launch_env(race="rlpilot-racepark3", launch_share=1.0, max_episode_seconds=0.08)
    try:
        env.reset()
        env.start_retries = 0
        _bad_launches(env, 1)
        env.step(np.zeros((1, 4), np.float32))
        obs, _, _, trunc, _ = env.step(np.zeros((1, 4), np.float32))
        assert trunc[0] and env.start_retries == 1 and np.isfinite(obs).all() and np.isfinite(env._raw).all()
        assert env._raw[0, F["time"]] < 0.1  # the new episode, on a fresh draw
    finally:
        env.close()


def test_launches_stay_clear_of_the_failing_edge():
    assert EnvConfig().launch_lateral == 0.7


def test_loose_rules_cant_raise_the_score():
    from rlpilot.tracking import track_score
    clean = {"eval/laps_completed_strict": 2.0, "eval/flying_lap_time_median_strict": 100.0}
    # finished both laps, but past an excursion the strict rules would have ended on lap 1
    loose = {"eval/laps_completed": 2.0, "eval/flying_lap_time_median": 100.0, "eval/laps_completed_strict": 0.0}
    assert track_score(loose, 100.0, 2) < track_score(clean, 100.0, 2)


def _nan_after(env, step_no, worker=0):
    """The engine state of `worker` turns NaN on the step_no-th step after this call, as proto-9's at 104.9M."""
    exchange, count = env.client.exchange, [0]

    def patched(cmds):
        exchange(cmds)
        if not np.isscalar(cmds) or cmds == CMD_STEP:
            count[0] += 1
            if count[0] == step_no:
                env._raw[worker, F["pos_x"]] = np.nan
                env._raw[worker, F["dist_from_start"]] = np.nan
    env.client.exchange = patched


def test_a_non_finite_state_mid_episode_ends_only_that_episode():
    env = _launch_env(n_workers=2, race="rlpilot-racepark3", recycle_resets=1000)
    try:
        env.reset()
        for _ in range(5):
            env.step(np.zeros((2, 4), np.float32))
        pid, kept = env.client.procs[0].pid, env._raw[0].copy()
        _nan_after(env, 1)
        obs, reward, term, trunc, info = env.step(np.zeros((2, 4), np.float32))
        assert term[0] and not term[1] and not trunc.any()
        assert env.term_reasons[info["stats"]["term_reason"][0]] == "sim_error"
        assert reward[0] == 0.0 and np.isfinite(reward).all() and np.isfinite(obs).all()
        assert np.isfinite(info["final_obs"][0]).all() and np.isfinite(info["final_raw"][0]).all()
        assert info["final_raw"][0][F["pos_x"]] == kept[F["pos_x"]]  # the last finite state
        assert all(np.isfinite(v[0]) for k, v in info["stats"].items())
        assert env.client.procs[0].pid != pid and env._raw[0, F["time"]] < 0.1  # a fresh worker, a new episode
        (record,) = env.sim_errors
        assert record["worker"] == 0 and set(record["nonfinite_fields"]) == {"pos_x", "dist_from_start"}
        assert record["episode_steps"] == 5 and record["last_finite"]["pos_x"] == float(kept[F["pos_x"]])
        for _ in range(3):
            obs, reward, *_ = env.step(np.zeros((2, 4), np.float32))
            assert np.isfinite(obs).all() and np.isfinite(reward).all()
    finally:
        env.close()


def test_vecnormalize_stays_finite_through_a_non_finite_state():
    env = _launch_env(n_workers=2, race="rlpilot-racepark3", recycle_resets=1000)
    try:
        from rlpilot.sb3 import SB3VecEnv
        venv = VecNormalize(SB3VecEnv(env), norm_obs=False, norm_reward=True, gamma=0.995)
        venv.reset()
        _nan_after(env, 3, worker=1)
        for _ in range(6):
            _, reward, done, infos = venv.step(np.zeros((2, 4), np.float32))
            assert np.isfinite(reward).all()
        assert np.isfinite(venv.ret_rms.mean) and np.isfinite(venv.ret_rms.var) and len(env.sim_errors) == 1
    finally:
        env.close()
