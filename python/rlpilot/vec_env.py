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
from .track import LookaheadTable

FULL_THROTTLE = 0.95
OFFTRACK_WHEELS = 2  # behaviour metric: at least half the car off the road
CAR_FACTORS = ("car_grip", "car_mass", "car_power")  # EnvConfig ranges = shm act slots, in this order
CAR_OBS_SCALE = 10.0  # observation = (factor - 1) x this
GEARBOX_ACTIONS = {"auto": (), "agent": ("shift",), "points": ("shift_up", "shift_down")}  # extra actions


class SpeedDreamsVecEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self.num_envs = n = cfg.n_workers
        self.races = tuple(cfg.races) or (cfg.race,)
        self._rng = np.random.default_rng(0)  # track draws (the sim itself is deterministic)
        self.client = Client(n, n_cars=1, frame_skip=cfg.frame_skip, gearbox=cfg.gearbox, spin_us=cfg.spin_us,
                             race=[self.races[i % len(self.races)] for i in range(n)], run_dir=cfg.run_dir,
                             deterministic=cfg.deterministic)
        self._look = LookaheadTable(self.races) if cfg.lookahead else None
        if self._look is not None:
            self._look_off, self._look_size = self._look.rows(self.client.races)
        if cfg.gearbox not in GEARBOX_ACTIONS:
            raise ValueError(f"gearbox must be one of {list(GEARBOX_ACTIONS)}, not {cfg.gearbox!r}")
        self.action_names = (("steer", "accel", "brake") if cfg.separate_pedals else ("steer", "pedal")) \
            + GEARBOX_ACTIONS[cfg.gearbox]
        self.act_dim = len(self.action_names)
        self._car_ranges = np.array([getattr(cfg, f) for f in CAR_FACTORS], np.float32)  # [3, (lo, hi)]
        self._car = np.ones((n, len(CAR_FACTORS)), np.float32)  # factors of each env's current episode
        n_extra = (self._look.n_features if self._look is not None else 0) + (len(CAR_FACTORS) if cfg.car_obs else 0)
        od = features.obs_dim(self.act_dim, n_extra)
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (od,), np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, n)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, n)

        self._raw = self.client.obs[:, 0]  # [n, OBS_DIM] view on shared memory
        self._act = self.client.act[:, 0]
        self._obs = np.zeros((n, od), np.float32)
        self._prev_action = np.zeros((n, self.act_dim), np.float32)
        self._cmds = np.full(n, CMD_STEP, np.uint32)
        self._shift_wait = np.zeros(n, np.int64)  # agent steps until the next shift is allowed
        self._shift_steps = int(np.ceil(cfg.shift_cooldown / cfg.step_seconds - 1e-9))
        self._rew = RewardTracker(n, cfg.reward, cfg.step_seconds, cfg.laps_per_episode)
        self._max_steps = int(round(cfg.max_episode_seconds / cfg.step_seconds))
        self.track_length = float(self._raw[0, F["track_length"]])  # of worker 0; per env: raw track_length
        # per-episode accumulators
        self._ret = np.zeros(n, np.float64)
        self._steps_on_track = np.zeros(n, np.int64)  # since the worker's last (re)spawn
        self._len = np.zeros(n, np.int64)
        self._t0 = np.zeros(n, np.float64)
        self._comp = np.zeros((len(REWARD_COMPONENTS), n), np.float64)
        self._dsteer = np.zeros(n, np.float64)
        self._off = np.zeros(n, np.int64)
        self._speed = np.zeros(n, np.float64)
        self._full = np.zeros(n, np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # the sim is deterministic; seed only feeds self.np_random
        self._draw_car(np.arange(self.num_envs))
        self.client.exchange(CMD_RESET)
        self._prev_action[:] = 0
        self._rew.reset(self._raw)
        self._clear(slice(None), time.perf_counter())
        return features.build_obs(self._raw, self._prev_action, self._obs, self._extra()).copy(), {}

    def _draw_car(self, idx: np.ndarray) -> None:
        """New car factors for envs idx; the workers apply them when these envs' races restart."""
        lo, hi = self._car_ranges[:, 0], self._car_ranges[:, 1]
        self._car[idx] = self._rng.uniform(lo, hi, (len(idx), len(lo))) if (hi > lo).any() else lo
        a = A[CAR_FACTORS[0]]
        self._act[idx, a:a + len(CAR_FACTORS)] = self._car[idx]

    @property
    def car_factors(self) -> np.ndarray:
        """[n, 3] grip, mass, power factors of each env's current episode."""
        return self._car.copy()

    def _extra(self, idx: np.ndarray | None = None) -> np.ndarray | None:
        parts = [self._lookahead(idx)]
        if self.cfg.car_obs:
            parts.append(((self._car if idx is None else self._car[idx]) - 1.0) * CAR_OBS_SCALE)
        parts = [x for x in parts if x is not None]
        return np.concatenate(parts, axis=1).astype(np.float32) if parts else None

    def _lookahead(self, idx: np.ndarray | None = None) -> np.ndarray | None:
        if self._look is None:
            return None
        if idx is None:
            return self._look(self._raw[:, F["dist_from_start"]], self._look_off, self._look_size)
        return self._look(self._raw[idx, F["dist_from_start"]], self._look_off[idx], self._look_size[idx])

    def _write_gear(self, shift):
        gear = np.rint(self._raw[:, F["gear"]]).astype(np.int64)
        target = np.maximum(gear, 1)  # neutral/reverse at the start: engage first (the worker clamps the top gear)
        dz = self.cfg.shift_deadzone
        up = (shift > dz) & (self._shift_wait <= 0)
        down = (shift < -dz) & (self._shift_wait <= 0) & (target > 1)
        target = target + up - down
        self._shift_wait = np.where(target != gear, self._shift_steps, np.maximum(self._shift_wait - 1, 0))
        self._act[:, A["gear"]] = target

    def _write_shift_points(self, up, down):
        (ulo, uhi), (dlo, dhi) = self.cfg.shift_up_range, self.cfg.shift_down_range
        self._act[:, A["shift_up"]] = ulo + (up + 1) * 0.5 * (uhi - ulo)
        self._act[:, A["shift_down"]] = dlo + (down + 1) * 0.5 * (dhi - dlo)

    def _clear(self, idx, now):
        self._shift_wait[idx] = 0
        self._ret[idx] = self._len[idx] = 0
        self._t0[idx] = now
        self._comp[:, idx] = 0
        self._dsteer[idx] = self._off[idx] = self._speed[idx] = self._full[idx] = 0

    def step(self, actions):
        actions = np.clip(np.asarray(actions, np.float32).reshape(self.num_envs, self.act_dim), -1, 1)
        features.write_actions(actions, self._act, self.cfg.separate_pedals)
        if self.cfg.gearbox == "agent":
            self._write_gear(actions[:, -1])
        elif self.cfg.gearbox == "points":
            self._write_shift_points(actions[:, -2], actions[:, -1])
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
        self._steps_on_track += 1
        truncated = (self._len >= self._max_steps) & ~terminated
        obs = features.build_obs(raw, self._prev_action, self._obs, self._extra()).copy()
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
            "progress": at_end(_scatter(np.maximum(final_raw[:, F["dist_raced"]], 0) / final_raw[:, F["track_length"]], idx, n)),
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

        since = self.client.resets_since_spawn[idx]
        switch = self._steps_on_track[idx] >= self.cfg.track_switch_steps if len(self.races) > 1 \
            else np.zeros(len(idx), bool)
        recycle = idx[(since >= self.cfg.recycle_resets) | switch]
        self._cmds[:] = CMD_NONE
        self._cmds[idx] = CMD_RESET
        self._draw_car(idx)  # before the reset / respawn, which applies them
        self._cmds[recycle] = CMD_NONE
        if len(recycle) < len(idx):
            self.client.exchange(self._cmds)
        for i in recycle:
            race = str(self._rng.choice(self.races)) if len(self.races) > 1 else None
            self.client.respawn(int(i), race)
            self._steps_on_track[i] = 0
            if self._look is not None:
                self._look_off[i], self._look_size[i] = self._look.offset[race or self.client.races[i]], \
                    self._look.size[race or self.client.races[i]]
        self._prev_action[idx] = 0
        self._rew.reset(self._raw, idx)
        self._clear(idx, now)
        obs[idx] = features.build_obs(self._raw[idx], self._prev_action[idx], self._obs[idx], self._extra(idx))
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
