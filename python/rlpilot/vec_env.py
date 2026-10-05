"""Gymnasium VectorEnv over rlpilot-worker processes (one solo race per worker).

Same-step autoreset: when an env ends, the returned obs is already the first obs of its next
episode; the last obs is in info["final_obs"] (and the last raw shm row in info["final_raw"]).
Per-episode statistics (reward components, behaviour, progress) are accumulated with a few vector
ops per step and only materialized when an episode ends, in info["episode"] and info["stats"].
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv

from . import features
from ._native import CMD_NONE, CMD_RESET, CMD_STEP, A, Client, F
from .config import EnvConfig
from .reward import REWARD_COMPONENTS, TERM_REASONS, RewardTracker

FULL_THROTTLE = 0.95
OFFTRACK_WHEELS = 2  # behaviour metric: at least half the car off the road


class SpeedDreamsVecEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self.num_envs = n = cfg.n_workers
        self.client = Client(n, n_cars=1, frame_skip=cfg.frame_skip, auto_gear=cfg.auto_gear, spin_us=cfg.spin_us,
                             race=cfg.race, run_dir=cfg.run_dir, deterministic=cfg.deterministic)
        self.act_dim = 3 if cfg.separate_pedals else 2
        od = features.obs_dim(self.act_dim)
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (od,), np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, n)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, n)

        self._raw = self.client.obs[:, 0]  # [n, OBS_DIM] view on shared memory
        self._act = self.client.act[:, 0]
        self._obs = np.zeros((n, od), np.float32)
        self._prev_action = np.zeros((n, self.act_dim), np.float32)
        self._cmds = np.full(n, CMD_STEP, np.uint32)
        self._rew = RewardTracker(n, cfg.reward, cfg.step_seconds, cfg.laps_per_episode)
        self._max_steps = int(round(cfg.max_episode_seconds / cfg.step_seconds))
        self.track_length = float(self._raw[0, F["track_length"]])
        # per-episode accumulators
        self._ret = np.zeros(n, np.float64)
        self._len = np.zeros(n, np.int64)
        self._t0 = np.zeros(n, np.float64)
        self._comp = np.zeros((len(REWARD_COMPONENTS), n), np.float64)
        self._dsteer = np.zeros(n, np.float64)
        self._off = np.zeros(n, np.int64)
        self._speed = np.zeros(n, np.float64)
        self._full = np.zeros(n, np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # the sim is deterministic; seed only feeds self.np_random
        self.client.exchange(CMD_RESET)
        self._prev_action[:] = 0
        self._rew.reset(self._raw)
        self._clear(slice(None), time.perf_counter())
        return features.build_obs(self._raw, self._prev_action, self._obs).copy(), {}

    def _clear(self, idx, now):
        self._ret[idx] = self._len[idx] = 0
        self._t0[idx] = now
        self._comp[:, idx] = 0
        self._dsteer[idx] = self._off[idx] = self._speed[idx] = self._full[idx] = 0

    def step(self, actions):
        actions = np.clip(np.asarray(actions, np.float32).reshape(self.num_envs, self.act_dim), -1, 1)
        features.write_actions(actions, self._act, self.cfg.separate_pedals)
        self.client.exchange(CMD_STEP)
        self._dsteer += np.abs(actions[:, 0] - self._prev_action[:, 0])
        self._full += self._act[:, A["accel"]] >= FULL_THROTTLE
        self._prev_action[:] = actions

        raw = self._raw
        reward, terminated, reason = self._rew.step(raw)
        self._comp += self._rew.components
        self._off += raw[:, F["wheels_off"]] >= OFFTRACK_WHEELS
        self._speed += raw[:, F["speed_x"]]
        self._ret += reward
        self._len += 1
        truncated = (self._len >= self._max_steps) & ~terminated
        obs = features.build_obs(raw, self._prev_action, self._obs).copy()
        done = terminated | truncated
        if not done.any():
            return obs, reward, terminated, truncated, {}

        idx = np.flatnonzero(done)
        now = time.perf_counter()
        n = self.num_envs
        final_raw = raw[idx].copy()
        steps = np.maximum(self._len, 1)

        def at_end(values):  # per-env array, zero where the episode did not end
            return np.where(done, values, 0)

        stats = {
            "dist_raced": at_end(_scatter(final_raw[:, F["dist_raced"]], idx, n)),
            "progress": at_end(_scatter(np.maximum(final_raw[:, F["dist_raced"]], 0) / self.track_length, idx, n)),
            "laps": at_end(_scatter(np.maximum(final_raw[:, F["laps"]] - 1, 0), idx, n)),
            "best_lap_time": at_end(_scatter(final_raw[:, F["best_lap_time"]], idx, n)),
            "damage": at_end(_scatter(final_raw[:, F["damage"]], idx, n)),
            "term_reason": reason.copy(),
            "steer_delta": at_end(self._dsteer / steps),
            "offtrack_frac": at_end(self._off / steps),
            "mean_speed": at_end(self._speed / steps),
            "full_throttle_frac": at_end(self._full / steps),
        } | {f"reward_{c}": at_end(self._comp[i]) for i, c in enumerate(REWARD_COMPONENTS)}
        final_obs = np.empty(n, dtype=object)
        final_obs[idx] = list(obs[idx])
        final_raw_obj = np.empty(n, dtype=object)
        final_raw_obj[idx] = list(final_raw)
        info = {
            "final_obs": final_obs, "_final_obs": done.copy(),
            "final_raw": final_raw_obj, "_final_raw": done.copy(),
            "episode": {"r": at_end(self._ret), "l": at_end(self._len), "t": at_end(now - self._t0)},
            "_episode": done.copy(),
            "stats": stats, "_stats": done.copy(),
        }

        self._cmds[:] = CMD_NONE
        self._cmds[idx] = CMD_RESET
        self.client.exchange(self._cmds)
        self._prev_action[idx] = 0
        self._rew.reset(self._raw, idx)
        self._clear(idx, now)
        obs[idx] = features.build_obs(self._raw[idx], self._prev_action[idx], self._obs[idx])
        return obs, reward, terminated, truncated, info

    def close_extras(self, **kwargs):
        self.client.close()

    @property
    def term_reasons(self) -> tuple[str, ...]:
        return TERM_REASONS


def _scatter(values: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, values.dtype)
    out[idx] = values
    return out
