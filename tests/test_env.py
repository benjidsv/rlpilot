"""The env against the real simulator: starts, steps, autoreset, termination, the gearbox in the engine."""
import numpy as np
import pytest

from conftest import scripted
from rlpilot import obs as O
from rlpilot.client import F
from rlpilot.controls import HOLD, UP
from rlpilot.env import ENDS, RacingEnv
from rlpilot.reward import NO_PROGRESS_SECONDS
from rlpilot.starts import KINDS, K, draw
from rlpilot.tracks import all_tracks


def fixed(env, kind, s=None):
    """A start_fn giving every env the same kind of start, at s (default: random)."""
    rng = np.random.default_rng(0)

    def fn(idx):
        where = rng.uniform(0, env.table.length[env.track[idx]]) if s is None else np.full(len(idx), s)
        return draw(np.full(len(idx), K[kind]), env.track[idx], where, env.table, rng)
    return fn


def test_every_kind_of_start_on_every_track():
    env = RacingEnv(all_tracks())
    try:
        for kind in KINDS:
            env.start_fn = fixed(env, kind)
            obs, _ = env.reset()
            assert obs.shape == (env.num_envs, O.DIM) and np.isfinite(obs).all(), kind
            assert (env.starts.kind == K[kind]).sum() >= env.num_envs - env.start_retries
        assert env.start_retries <= 3, f"{env.start_retries} bad launches out of {env.num_envs * len(KINDS)}"
    finally:
        env.close()


@pytest.fixture(scope="module")
def env():
    e = RacingEnv(["jarama"] * 2 + ["Imula"] * 2, episode_seconds=8.0)
    yield e
    e.close()


def test_launch_speed_and_autoreset(env):
    env.start_fn = fixed(env, "nominal")
    obs, _ = env.reset()
    assert np.allclose(env._raw[:, F["speed_x"]], env.starts.speed, rtol=0.05, atol=0.5)
    for _ in range(int(8.0 / env.dt) + 5):
        obs, reward, term, trunc, info = env.step(scripted(obs))
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        if "episode_end" in info:
            break
    done = info["_episode_end"]
    assert done.any() and (trunc | term)[done].all()
    e = info["episode_end"]
    assert all(ENDS[c] in ("time_limit", "no_progress", "damage") for c in e["end"][done])
    assert not np.allclose(info["final_obs"][done], obs[done])  # the next episode's first observation was returned


def test_no_progress_ends_the_episode(env):
    env.start_fn = fixed(env, "grid")
    env.max_steps = 10**6  # this env's 8 s episodes would be cut before the rule triggers
    obs, _ = env.reset()
    brake = np.tile([0.0, -1.0, HOLD], (env.num_envs, 1)).astype(np.float32)
    for t in range(int(NO_PROGRESS_SECONDS / env.dt) + 5):
        obs, _, term, _, info = env.step(brake)
        if term.any():
            break
    # envs 0, 1 are on Jarama, where the braked car stays put (on Imola's grid it creeps at 0.1 m/s, which is progress)
    assert term[:2].all() and (info["episode_end"]["end"][:2] == ENDS.index("no_progress")).all()
    assert abs(t + 1 - NO_PROGRESS_SECONDS / env.dt) <= 1
    env.max_steps = int(8.0 / env.dt)


def test_shift_reaches_the_engine(env):
    env.start_fn = fixed(env, "nominal", s=500.0)
    obs, _ = env.reset()
    gear0 = env.controls.gear.copy()
    a = np.tile([0.0, 0.3, UP], (env.num_envs, 1)).astype(np.float32)
    obs, *_ = env.step(a)
    for _ in range(10):
        a[:, 2] = HOLD
        obs, *_ = env.step(a)
    assert (env.controls.gear == np.minimum(gear0 + 1, 6)).all()
    assert (np.rint(env._raw[:, F["gear"]]) == env.controls.gear).all()  # engaged after the shift time
