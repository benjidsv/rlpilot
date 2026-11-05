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
from .features import CAR_FACTORS, CAR_OBS_SCALE
from .track import EdgeTable, LineTable, LookaheadTable

FULL_THROTTLE = 0.95
OFFTRACK_WHEELS = 2  # behaviour metric: at least half the car off the road
BRAKE_ON = 0.1  # behaviour metric: the brake counts as applied from here
START_RETRIES = 4  # new launch draws for an env whose first observation isn't finite; the grid after that
GEARBOX_ACTIONS = {"auto": (), "agent": ("shift",), "points": ("shift_up", "shift_down")}  # extra actions
# Per-episode behaviour means in info["stats"], logged as behavior/* and eval/behavior/*.
BEHAVIOR_STATS = ("steer_delta", "steer_requested", "steer_reversal_frac", "steer_limit_frac", "steer_lock_frac",
                  "steer_abs", "pedal_delta", "throttle_delta", "brake_delta", "brake_frac", "brake_onsets_per_s", "offtrack_frac",
                  "mean_speed", "full_throttle_frac", "line_dev", "abs_offset")


class SpeedDreamsVecEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self.num_envs = n = cfg.n_workers
        self.races = tuple(cfg.races) or (cfg.race,)
        if cfg.track_switch_steps <= 0 and n % len(self.races):
            raise ValueError(f"{n} workers can't drive {len(self.races)} tracks equally: without track switching a "
                             "worker keeps its track, so n_workers must be a multiple of the number of races")
        if cfg.steer_limit is not None and not 0 < cfg.steer_limit <= 1:
            raise ValueError("steer_limit must be in (0, 1]")
        if cfg.steer_history and cfg.steer_limit is None:
            raise ValueError("steer_history needs steer_limit: its values are normalized by the limit")
        if not 0 <= cfg.launch_share <= 1:
            raise ValueError("launch_share must be in [0, 1]")
        self._rng = np.random.default_rng(0)  # track, car and launch draws (the sim itself is deterministic)
        self.client = Client(n, n_cars=1, frame_skip=cfg.frame_skip, gearbox=cfg.gearbox, spin_us=cfg.spin_us,
                             race=[self.races[i % len(self.races)] for i in range(n)], run_dir=cfg.run_dir,
                             deterministic=cfg.deterministic)
        # Track tables, with each worker's rows in them. _assign_track is the only thing that changes those rows.
        self._look = LookaheadTable(self.races, cfg.lookahead_stat) if cfg.lookahead else None
        self._line = LineTable(self.races)  # the line_dev metric; a policy input only with line_obs
        self._edges = EdgeTable(self.races, cfg.edge_points, cfg.edge_horizon, cfg.edge_floor) if cfg.edge_points else None
        if self._look is not None:
            self._look_off, self._look_size = self._look.rows(self.client.races)
        self._line_off, self._line_size = self._line.rows(self.client.races)
        if self._edges is not None:
            self._edge_rows = self._edges.rows(self.client.races)
        self._race_index = np.array([self.races.index(r) for r in self.client.races], np.int64)
        self.track_steps = np.zeros(len(self.races), np.int64)  # agent steps driven per race (self.races order)
        if cfg.gearbox not in GEARBOX_ACTIONS:
            raise ValueError(f"gearbox must be one of {list(GEARBOX_ACTIONS)}, not {cfg.gearbox!r}")
        self.action_names = (("steer", "accel", "brake") if cfg.separate_pedals else ("steer", "pedal")) \
            + GEARBOX_ACTIONS[cfg.gearbox]
        self.act_dim = len(self.action_names)
        self._car_ranges = np.array([getattr(cfg, f) for f in CAR_FACTORS], np.float32)  # [3, (lo, hi)]
        self._car = np.ones((n, len(CAR_FACTORS)), np.float32)  # factors of each env's current episode
        self._launched = np.zeros(n, bool)  # the current episode started from a random launch
        self.start_retries = 0  # launches restarted because their first observation wasn't finite
        self._launch_steps = np.zeros(n, np.int64)  # agent steps driven in launched episodes, per worker
        self._mode_steps = np.zeros(n, np.int64)    # and in all episodes, while launches are on
        self.obs_layout = features.make_layout(self.action_names, self._look.n_features if self._look is not None else 0,
                                               len(self._line.points) if cfg.line_obs else 0, cfg.car_obs,
                                               cfg.steer_history, cfg.edge_points)
        od = self.obs_layout.dim
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (od,), np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, n)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, n)

        self._raw = self.client.obs[:, 0]  # [n, OBS_DIM] view on shared memory
        self._act = self.client.act[:, 0]
        self._prev_action = np.zeros((n, self.act_dim), np.float32)  # applied controls, 0 at an episode's start
        self.applied = np.zeros((n, self.act_dim), np.float32)  # controls of the last step, kept across autoreset
        self._steer_hist = np.zeros((n, cfg.steer_history), np.float32)  # applied changes, most recent first
        self._cmds = np.full(n, CMD_STEP, np.uint32)
        self._shift_wait = np.zeros(n, np.int64)  # agent steps until the next shift is allowed
        # "points" upshift hold: gear and rpm fraction at the previous step, and the current cap on the down point
        self._gear_prev = np.zeros(n, np.float32)
        self._rpm_prev = np.zeros(n, np.float32)
        self._down_cap = np.full(n, np.inf, np.float32)
        self._shift_steps = int(np.ceil(cfg.shift_cooldown / cfg.step_seconds - 1e-9))
        self._rew = RewardTracker(n, cfg.reward, cfg.step_seconds, cfg.laps_per_episode)
        self.set_time_limits(None)  # self._max_steps [n]
        self.track_length = float(self._raw[0, F["track_length"]])  # of worker 0; per env: raw track_length
        # per-episode accumulators
        self._ret = np.zeros(n, np.float64)
        self._steps_on_track = np.zeros(n, np.int64)  # since the worker's last (re)spawn
        self._len = np.zeros(n, np.int64)
        self._t0 = np.zeros(n, np.float64)
        self._comp = np.zeros((len(REWARD_COMPONENTS), n), np.float64)
        self._dsteer = np.zeros(n, np.float64)
        self._dsteer_req = np.zeros(n, np.float64)
        self._reversals = np.zeros(n, np.int64)
        self._at_limit = np.zeros(n, np.int64)
        self._at_lock = np.zeros(n, np.int64)
        self._steer_sum = np.zeros(n, np.float64)
        self._dpedal = np.zeros(n, np.float64)
        self._prev_pedals = np.zeros((n, 2), np.float64)  # throttle, brake, in physical units [0, 1]
        self._dpedals = np.zeros((n, 2), np.float64)
        self._braking = np.zeros(n, bool)
        self._brake_steps = np.zeros(n, np.int64)
        self._brake_onsets = np.zeros(n, np.int64)
        self._linedev = np.zeros(n, np.float64)
        self._absoff = np.zeros(n, np.float64)
        self._off = np.zeros(n, np.int64)
        self._speed = np.zeros(n, np.float64)
        self._full = np.zeros(n, np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # the sim is deterministic; seed only feeds self.np_random
        self._draw_car(np.arange(self.num_envs))
        self._draw_start(np.arange(self.num_envs))
        self.client.exchange(CMD_RESET)
        self._retry_bad_starts(np.arange(self.num_envs))
        self._start_rewards(slice(None))
        self._clear(slice(None), time.perf_counter())
        return self._observe(), {}

    def _draw_car(self, idx: np.ndarray) -> None:
        """New car factors for envs idx; the workers apply them when these envs' races restart."""
        lo, hi = self._car_ranges[:, 0], self._car_ranges[:, 1]
        self._car[idx] = self._rng.uniform(lo, hi, (len(idx), len(lo))) if (hi > lo).any() else lo
        a = A[CAR_FACTORS[0]]
        self._act[idx, a:a + len(CAR_FACTORS)] = self._car[idx]

    def _draw_start(self, idx: np.ndarray) -> None:
        """Where envs idx start their next episode; the workers apply it when these envs' races restart. Each env picks
        the mode behind its target share of samples: per env, because envs ending on the same step would otherwise all
        pick the same one, and by samples, because an episode that survives holds far more of them than one that crashes."""
        cfg = self.cfg
        if cfg.launch_share <= 0:
            return
        unstarted = self._mode_steps[idx] == 0  # no history yet: drawn, so that the envs don't all begin the same way
        launch = np.where(unstarted, self._rng.random(len(idx)) < cfg.launch_share,
                          self._launch_steps[idx] <= cfg.launch_share * self._mode_steps[idx])
        self._launched[idx] = launch
        k = len(idx)
        half_range = 0.5 * cfg.launch_lateral
        slots = np.stack([self._rng.uniform(1e-4, 1 - 1e-4, k), self._rng.uniform(0.5 - half_range, 0.5 + half_range, k),
                          self._rng.uniform(-cfg.launch_yaw, cfg.launch_yaw, k), self._rng.uniform(*cfg.launch_speed, k)], 1)
        self._act[idx, A["launch_pos"]:A["launch_speed"] + 1] = np.where(launch[:, None], slots, 0).astype(np.float32)

    def _retry_bad_starts(self, idx: np.ndarray) -> None:
        """Rare launches come back from the engine with a NaN position and speed, before any reward reads them, and would
        reach the policy: seen near the right edge with some heading, probably a car corner starting past a barrier. Restart
        those envs with a new draw, then from the grid. A grid start that isn't finite is a different bug, so it raises."""
        for attempt in range(START_RETRIES + 2):
            bad = idx[~np.isfinite(self._raw[idx]).all(axis=1)]
            if not len(bad):
                return
            if attempt == START_RETRIES + 1 or self.cfg.launch_share <= 0:
                raise RuntimeError(f"non-finite first observation from a grid start: envs {bad.tolist()}")
            self.start_retries += len(bad)
            self._draw_start(bad)
            if attempt == START_RETRIES:
                self._launched[bad] = False
                self._act[bad, A["launch_pos"]:A["launch_speed"] + 1] = 0
            self._cmds[:] = CMD_NONE
            self._cmds[bad] = CMD_RESET
            self.client.exchange(self._cmds)

    def _start_rewards(self, idx) -> None:
        """Reward state of envs idx at their first observation. A launched car's distance raced counts from where it
        starts: the engine's dist_raced there is about minus a lap, as it hasn't crossed the start line."""
        self._rew.reset(self._raw, idx, self._line_dev(None if isinstance(idx, slice) else idx))
        self._rew.raced[idx] = np.where(self._launched[idx], 0.0, self._rew.raced[idx])

    def set_time_limits(self, seconds: np.ndarray | None) -> None:
        """Per-env episode time limits (sim seconds, [n]); None restores cfg.max_episode_seconds for all."""
        sec = self.cfg.max_episode_seconds if seconds is None else np.asarray(seconds, np.float64)
        self._max_steps = np.round(np.broadcast_to(sec, (self.num_envs,)) / self.cfg.step_seconds).astype(np.int64)

    @property
    def car_factors(self) -> np.ndarray:
        """[n, 3] grip, mass, power factors of each env's current episode."""
        return self._car.copy()

    def _observe(self, idx: np.ndarray | None = None) -> np.ndarray:
        """The policy's observations of all envs, or of envs idx: the one code path behind reset, step and autoreset,
        training and eval. Everything it reads is per-env state, indexed the same way."""
        sel = slice(None) if idx is None else idx
        raw = self._raw[sel]
        cfg = self.cfg
        parts = features.base_parts(raw, self._prev_action[sel])
        if self._look is not None:
            parts["lookahead"] = self._look(raw[:, F["dist_from_start"]], self._look_off[sel], self._look_size[sel])
        if cfg.line_obs:
            parts["line"] = self._line_obs(idx)
        if cfg.car_obs:
            parts["car"] = (self._car[sel] - 1.0) * CAR_OBS_SCALE
        if cfg.steer_history:
            parts["steer_history"] = self._steer_hist[sel] / cfg.steer_limit
        if self._edges is not None:
            world = self._edges.world(raw[:, F["dist_from_start"]], np.hypot(raw[:, F["speed_x"]], raw[:, F["speed_y"]]),
                                      *(r[sel] for r in self._edge_rows))
            parts |= features.edge_parts(self._edges.car_frame(raw[:, F["pos_x"]], raw[:, F["pos_y"]], raw[:, F["yaw"]],
                                                               world))
        return features.assemble(self.obs_layout, parts)

    def _assign_track(self, i: int, race: str) -> None:
        """Point worker i's rows in the track tables at `race` (after a respawn on it)."""
        self._race_index[i] = self.races.index(race)
        if self._look is not None:
            self._look_off[i], self._look_size[i] = self._look.offset[race], self._look.size[race]
        self._line_off[i], self._line_size[i] = self._line.offset[race], self._line.size[race]
        if self._edges is not None:
            off, cells, spacing = self._edge_rows
            off[i], cells[i], spacing[i] = self._edges.offset[race], self._edges.cells[race], self._edges.spacing[race]

    def _line_at(self, idx: np.ndarray | None, ahead) -> np.ndarray:
        """The racing line's offset (m) at `ahead` metres from each env's position: [n or len(idx), len(ahead)]."""
        sel = slice(None) if idx is None else idx
        return self._line(self._raw[sel, F["dist_from_start"]], self._line_off[sel], self._line_size[sel], ahead)

    def _line_dev(self, idx: np.ndarray | None = None) -> np.ndarray | None:
        """|offset from the racing line| (m), a metric (and shaping, when on)."""
        sel = slice(None) if idx is None else idx
        return np.abs(self._raw[sel, F["to_middle"]] - self._line_at(idx, (0,))[:, 0])

    def _line_obs(self, idx: np.ndarray | None = None) -> np.ndarray:
        sel = slice(None) if idx is None else idx
        half_width = np.maximum(self._raw[sel, F["track_width"]], 1.0)[:, None] * 0.5
        rel = (self._line_at(idx, self._line.points) - self._raw[sel, F["to_middle"]][:, None]) / half_width
        return np.clip(rel, -3, 3)

    def set_line_shaping(self, k: float) -> None:
        """Racing-line shaping coefficient from now on (training fades it)."""
        self._rew.line_k = float(k)

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
        down_point = dlo + (down + 1) * 0.5 * (dhi - dlo)
        gap = self.cfg.upshift_hold_gap
        if gap is not None:
            # an upshift seen now happened during the last step, at about the rpm of the previous step
            gear = self._raw[:, F["gear"]]
            rpm = self._raw[:, F["rpm"]] / np.maximum(self._raw[:, F["rpm_redline"]], 1.0)
            upshifted = (gear > self._gear_prev) & (self._gear_prev >= 1)
            self._down_cap = np.where(upshifted, self._rpm_prev - gap,
                                      np.where(gear != self._gear_prev, np.inf, self._down_cap))
            self._gear_prev[:], self._rpm_prev[:] = gear, rpm
            down_point = np.minimum(down_point, self._down_cap)
        self._act[:, A["shift_up"]] = ulo + (up + 1) * 0.5 * (uhi - ulo)
        self._act[:, A["shift_down"]] = down_point

    def _clear(self, idx, now):
        """Start envs idx's next episode in all per-episode state (controls are 0 on the first observation)."""
        self._prev_action[idx] = 0
        self._prev_pedals[idx] = self._dpedals[idx] = 0
        self._braking[idx] = False
        self._brake_steps[idx] = self._brake_onsets[idx] = 0
        self._steer_hist[idx] = 0
        self._shift_wait[idx] = 0
        self._gear_prev[idx] = self._rpm_prev[idx] = 0
        self._down_cap[idx] = np.inf
        self._ret[idx] = self._len[idx] = 0
        self._t0[idx] = now
        self._comp[:, idx] = 0
        self._dsteer[idx] = self._dpedal[idx] = self._off[idx] = self._speed[idx] = self._full[idx] = 0
        self._dsteer_req[idx] = self._reversals[idx] = self._at_limit[idx] = self._at_lock[idx] = self._steer_sum[idx] = 0
        self._linedev[idx] = self._absoff[idx] = 0

    def step(self, actions, active: np.ndarray | None = None):
        """active [n] bool (default all): envs left out are not simulated this step and cannot end; their
        returned obs / reward are meaningless (evaluation pauses envs whose episode is over)."""
        actions = np.clip(np.asarray(actions, np.float32).reshape(self.num_envs, self.act_dim), -1, 1)
        prev_steer = self._prev_action[:, 0].copy()
        self.applied[:] = actions
        limit = self.cfg.steer_limit
        if limit is None:
            requested = actions[:, 0] - prev_steer
        else:
            requested = actions[:, 0] * limit
            steer = np.clip(prev_steer + requested, -1.0, 1.0)
            self.applied[:, 0] = steer if active is None else np.where(active, steer, prev_steer)  # paused: no change
        features.write_actions(self.applied, self._act, self.cfg.separate_pedals)
        if self.cfg.gearbox == "agent":
            self._write_gear(actions[:, -1])
        elif self.cfg.gearbox == "points":
            self._write_shift_points(actions[:, -2], actions[:, -1])
        self.client.exchange(CMD_STEP if active is None else np.where(active, CMD_STEP, CMD_NONE).astype(np.uint32))
        d_steer = self.applied[:, 0] - prev_steer
        if self.cfg.steer_history:
            hist = self._steer_hist
            shifted = np.concatenate([d_steer[:, None], hist[:, :-1]], axis=1)
            hist[:] = shifted if active is None else np.where(active[:, None], shifted, hist)
        self.track_steps += np.bincount(self._race_index if active is None else self._race_index[active],
                                        minlength=len(self.races))
        self._dsteer += np.abs(d_steer)
        self._dsteer_req += np.abs(requested)
        self._at_limit += np.abs(actions[:, 0]) >= 1.0
        self._at_lock += np.abs(self.applied[:, 0]) >= 1.0
        self._steer_sum += np.abs(self.applied[:, 0])
        self._dpedal += np.abs(self.applied[:, 1] - self._prev_action[:, 1])
        self._full += self._act[:, A["accel"]] >= FULL_THROTTLE
        pedals = self._act[:, [A["accel"], A["brake"]]].astype(np.float64)
        self._dpedals += np.abs(pedals - self._prev_pedals)
        self._prev_pedals[:] = pedals
        braking = pedals[:, 1] >= BRAKE_ON
        self._brake_steps += braking
        self._brake_onsets += braking & ~self._braking
        self._braking[:] = braking
        self._prev_action[:] = self.applied

        raw = self._raw
        line_dev = self._line_dev()
        reward, terminated, reason = self._rew.step(raw, d_steer, line_dev)
        self._reversals += self._rew.reversed
        self._linedev += line_dev
        self._absoff += np.abs(raw[:, F["to_middle"]]) / (np.maximum(raw[:, F["track_width"]], 1.0) * 0.5)
        self._comp += self._rew.components
        self._off += raw[:, F["wheels_off"]] >= OFFTRACK_WHEELS
        self._speed += raw[:, F["speed_x"]]
        self._ret += reward
        self._len += 1
        self._steps_on_track += 1
        if self.cfg.launch_share > 0:
            moving = 1 if active is None else active
            self._mode_steps += moving
            self._launch_steps += self._launched * moving
        truncated = (self._len >= self._max_steps) & ~terminated
        if active is not None:
            terminated &= active
            truncated &= active
        obs = self._observe()
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
            "dist_raced": at_end(self._rew.raced),  # not the raw field, which a backward crossing of the line inflates
            "progress": at_end(_scatter(np.maximum(self._rew.raced[idx], 0) / final_raw[:, F["track_length"]], idx, n)),
            "laps": at_end(_scatter(np.maximum(final_raw[:, F["laps"]] - 1, 0), idx, n)),
            "best_lap_time": at_end(_scatter(final_raw[:, F["best_lap_time"]], idx, n)),
            "damage": at_end(_scatter(final_raw[:, F["damage"]], idx, n)),
            "term_reason": reason.copy(),
            "launched": at_end(self._launched.astype(np.float64)),
            "track": at_end(self._race_index),                     # index in self.races
            "end_dist": at_end(_scatter(final_raw[:, F["dist_from_start"]], idx, n)),  # m along the lap
            "seconds": at_end(self._len * self.cfg.step_seconds),
            "steer_delta": at_end(self._dsteer / steps),          # applied change of the steering command, per step
            "steer_requested": at_end(self._dsteer_req / steps),  # what the policy asked for; differs at full lock
            "steer_reversal_frac": at_end(self._reversals / steps),
            "steer_limit_frac": at_end(self._at_limit / steps),   # steer action saturated at +-1
            "steer_lock_frac": at_end(self._at_lock / steps),     # steering command at +-1
            "steer_abs": at_end(self._steer_sum / steps),
            "pedal_delta": at_end(self._dpedal / steps),
            "throttle_delta": at_end(self._dpedals[:, 0] / steps),  # physical throttle and brake, in any pedal mode
            "brake_delta": at_end(self._dpedals[:, 1] / steps),
            "brake_frac": at_end(self._brake_steps / steps),        # brake at BRAKE_ON or more
            "brake_onsets_per_s": at_end(self._brake_onsets / (steps * self.cfg.step_seconds)),
            "line_dev": at_end(self._linedev / steps),
            "abs_offset": at_end(self._absoff / steps),
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
        switch = np.zeros(self.num_envs, bool)
        if self.cfg.track_switch_steps > 0 and len(self.races) > 1:
            switch[idx] = self._steps_on_track[idx] >= self.cfg.track_switch_steps
        recycle = idx[(since >= self.cfg.recycle_resets) | switch[idx]]
        self._cmds[:] = CMD_NONE
        self._cmds[idx] = CMD_RESET
        self._draw_car(idx)  # before the reset / respawn, which apply them
        self._draw_start(idx)
        self._cmds[recycle] = CMD_NONE
        if len(recycle) < len(idx):
            self.client.exchange(self._cmds)
        for i in recycle:
            # a recycled worker keeps its track; only a track switch draws a new one
            race = str(self._rng.choice(self.races)) if switch[i] else self.client.races[i]
            self.client.respawn(int(i), race)
            self._assign_track(int(i), race)
            if switch[i]:
                self._steps_on_track[i] = 0
        self._retry_bad_starts(idx)
        self._start_rewards(idx)
        self._clear(idx, now)
        obs[idx] = self._observe(idx)
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
