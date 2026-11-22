"""Gymnasium VectorEnv over rlpilot-worker processes: one solo race per worker, each on a fixed track.

- Each env is one worker on one track (see tracks.allocate for spreading workers over tracks by road length).
- Every episode starts from a start drawn by `start_fn` (default: random training starts, see starts.py). Disturbed
  starts run their forced steps inside the reset, so an episode begins at the handover.
- Same-step autoreset: when an env's episode ends, the observation returned for it is already the first of its next
  episode; the last one is in info["final_obs"] (dense [n, DIM], valid where info["_final_obs"]). Per-episode
  statistics are in info["episode_end"] (arrays [n], valid where info["_episode_end"]).
- Ends: a failure (reward.FAILURES) terminates; the time limit, or `max_distance` along the track (eval scenarios),
  truncates.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv

from . import obs as observation
from .client import CMD_NONE, CMD_RESET, CMD_STEP, Client, F
from .controls import HOLD, Controls
from .reward import COMPONENTS, FAILURES, Reward
from .starts import K, Starts, draw, train_kinds, write_launch
from .tracks import Track, TrackTable, race_for

ENDS = FAILURES + ("time_limit", "distance")  # end codes of info["episode_end"]["end"]
START_RETRIES = 4  # new draws for a start whose first observation isn't finite (rare launches); then the grid
FAR_OFF = 3.0      # m beyond the road's edge that counts as having lost control
NEAR_START = 50.0  # m: an episode that fails without ever getting this far down the road led nowhere (start too hard?)


class RacingEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, tracks: list[str], episode_seconds: float = 30.0, max_distance: float | None = None,
                 start_fn: Callable[[np.ndarray], Starts] | None = None,
                 retry_fn: Callable[[np.ndarray, int], Starts] | None = None, seed: int = 0, frame_skip: int = 2,
                 spin_us: int = 50, recycle_resets: int = 2000, run_dir=None):
        """tracks: one per worker. start_fn(env indices) -> their next Starts (default: random training starts).
        retry_fn(env indices, attempt) -> replacements for starts whose first observation wasn't finite (default: new
        random starts, then the grid)."""
        self.track_names = sorted(set(tracks))
        self.table = TrackTable([Track.load(t) for t in self.track_names])
        self.track = np.array([self.track_names.index(t) for t in tracks], np.int64)  # per env, index in track_names
        self.num_envs = n = len(tracks)
        self.dt = 0.02 * frame_skip
        self.max_steps = int(round(episode_seconds / self.dt))
        self.max_distance = max_distance
        self.recycle_resets = recycle_resets
        self.start_fn = start_fn or self._random_starts
        self.retry_fn = retry_fn or self._random_retry
        self._rng = np.random.default_rng(seed)
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (observation.DIM,), np.float32)
        # steer change and pedal in [-1, 1]; shift choice 0 / 1 / 2 = down / hold / up
        self.single_action_space = gym.spaces.Box(np.array([-1, -1, 0], np.float32), np.array([1, 1, 2], np.float32))
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, n)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, n)

        self.client = Client([race_for(t) for t in tracks], frame_skip, spin_us, run_dir)
        self._raw = self.client.obs[:, 0]  # [n, OBS_DIM] view on shared memory
        self._act = self.client.act[:, 0]
        self._last_raw = np.zeros_like(self._raw)  # each env's last finite state
        self._cmds = np.full(n, CMD_STEP, np.uint32)
        self._sim_failed = np.zeros(n, bool)
        self.controls = Controls(n)
        self.reward = Reward(n, self.dt)
        self.starts = draw(np.zeros(n, np.int64), self.track, np.zeros(n), self.table, self._rng)  # current, per env
        self.start_retries = 0
        # per-episode accumulators
        self._ret = np.zeros(n)
        self._len = np.zeros(n, np.int64)
        self._comp = np.zeros((len(COMPONENTS), n))
        self._shifts = np.zeros(n, np.int64)
        self._refused = np.zeros(n, np.int64)
        self._off_steps = np.zeros(n, np.int64)
        self._speed = np.zeros(n)
        self._ever_off = np.zeros(n, bool)     # all four wheels off the road at some point
        self._lost = np.zeros(n, bool)         # lost control at some point: hit something, faced backwards, far off

    # ---- starts ----

    def _random_starts(self, idx: np.ndarray) -> Starts:
        s = self._rng.uniform(0, self.table.length[self.track[idx]])
        return draw(train_kinds(len(idx), self._rng), self.track[idx], s, self.table, self._rng)

    def _random_retry(self, idx: np.ndarray, attempt: int) -> Starts:
        """Rare launches come back non-finite (probably a car corner placed past a barrier): redraw, then the grid."""
        starts = self._random_starts(idx)
        if attempt == START_RETRIES - 1:
            starts.kind[:], starts.kick_steps[:] = K["grid"], 0
        return starts

    def _start(self, idx: np.ndarray) -> None:
        """Start envs idx's next episodes: launch (reset, or respawn a worker due for recycling), kick, check."""
        starts = self.start_fn(idx)
        todo = np.arange(len(idx))  # rows of idx / starts still to (re)launch
        for attempt in range(START_RETRIES + 1):
            envs = idx[todo]
            launch = self._act[envs].copy()
            write_launch(starts[todo], self.track[envs], self.table, launch)
            self._act[envs] = launch
            due = envs[self.client.resets_since_spawn[envs] >= self.recycle_resets]
            self._cmds[:] = CMD_NONE
            self._cmds[envs] = CMD_RESET
            self._cmds[due] = CMD_NONE
            for i in due:
                self.client.respawn(int(i))
            if len(due) < len(envs):
                self.client.exchange(self._cmds)
            self.controls.reset(envs, self._raw)
            self._kick(envs, starts[todo])
            bad = ~np.isfinite(self._raw[envs]).all(axis=1)
            if not bad.any():
                break
            if attempt == START_RETRIES:
                raise RuntimeError(f"non-finite first observations after {START_RETRIES} retries: envs {envs[bad].tolist()}")
            self.start_retries += int(bad.sum())
            todo = todo[bad]
            redraw = self.retry_fn(idx[todo], attempt)
            for f, v in redraw.to_columns().items():
                getattr(starts, f)[todo] = v
        for f, v in starts.to_columns().items():
            getattr(self.starts, f)[idx] = v
        self.reward.reset(idx, self._raw)
        self._last_raw[idx] = self._raw[idx]
        self._ret[idx] = self._len[idx] = self._comp[:, idx] = 0
        self._shifts[idx] = self._refused[idx] = self._off_steps[idx] = self._speed[idx] = 0
        self._ever_off[idx] = self._lost[idx] = False

    def _kick(self, envs: np.ndarray, starts: Starts) -> None:
        """Forced actions of disturbed starts, stepped inside the reset (no reward, no observation)."""
        steps = starts.kick_steps
        action = np.zeros((self.num_envs, 3), np.float32)
        action[:, 2] = HOLD
        action[envs, 0], action[envs, 1] = starts.kick_steer, starts.kick_pedal
        for k in range(int(steps.max(initial=0))):
            on = np.zeros(self.num_envs, bool)
            on[envs[steps > k]] = True
            self.controls.apply(action, self._raw, self._act, active=on)
            self.client.exchange(np.where(on, CMD_STEP, CMD_NONE).astype(np.uint32))

    # ---- gymnasium API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._start(np.arange(self.num_envs))
        return self._observe(), {}

    def _observe(self) -> np.ndarray:
        return observation.observe(self._raw, self.track, self.controls, self.table)

    def step(self, actions, active: np.ndarray | None = None):
        """actions [n, 3]. active [n] bool (default all): envs left out aren't simulated and can't end; their returned
        observation and reward are meaningless (evaluation pauses envs that have run out of scenarios)."""
        n = self.num_envs
        on = np.ones(n, bool) if active is None else np.asarray(active, bool)
        self.controls.apply(np.asarray(actions, np.float32).reshape(n, 3), self._raw, self._act, on)
        self.client.exchange(np.where(on, CMD_STEP, CMD_NONE).astype(np.uint32))
        bad = ~np.isfinite(self._raw).all(axis=1) & on
        if bad.any():
            self._sim_error(np.flatnonzero(bad))
        raw = self._raw

        reward, terminated, reason = self.reward.step(raw, on)
        if self._sim_failed.any():  # no reward, no penalty: the episode just ends, on its last finite state
            f = self._sim_failed
            reward[f], self.reward.components[:, f] = 0.0, 0.0
            terminated |= f
            reason[f] = FAILURES.index("sim_error")
            self._sim_failed[:] = False
        hw = np.maximum(raw[:, F["track_width"]], 1.0) * 0.5
        off = raw[:, F["wheels_off"]] >= 4
        lost = self.reward.damage_taken | (np.cos(raw[:, F["angle"]]) < 0) | (np.abs(raw[:, F["to_middle"]]) > hw + FAR_OFF)
        self._ret += reward * on
        self._len += on
        self._comp += self.reward.components * on
        self._shifts += self.controls.shifted
        self._refused += self.controls.refused
        self._off_steps += off & on
        self._speed += raw[:, F["speed_x"]] * on
        self._ever_off |= off & on
        self._lost |= lost & on

        truncated = self._len >= self.max_steps
        if self.max_distance is not None:
            truncated |= self.reward.raced >= self.max_distance
        truncated &= on & ~terminated
        obs = self._observe()
        done = terminated | truncated
        self._last_raw[on] = raw[on]
        if not done.any():
            return obs, reward, terminated, truncated, {}

        idx = np.flatnonzero(done)
        end = np.where(terminated, reason, np.where(self._len >= self.max_steps, ENDS.index("time_limit"), ENDS.index("distance")))
        steps = np.maximum(self._len, 1)
        at_end = lambda v: np.where(done, v, 0)
        stats = {
            "return": at_end(self._ret), "steps": at_end(self._len), "seconds": at_end(self._len * self.dt),
            "distance": at_end(self.reward.raced), "end": np.where(done, end, -1), "kind": at_end(self.starts.kind),
            "track": at_end(self.track), "failed_near_start": at_end(terminated & (self.reward.best < NEAR_START)),
            "shifts_per_min": at_end(self._shifts / (steps * self.dt / 60)), "refused_shifts": at_end(self._refused),
            "offroad_frac": at_end(self._off_steps / steps), "mean_speed": at_end(self._speed / steps),
            "ever_off": at_end(self._ever_off), "lost_control": at_end(self._lost),
        } | {f"reward_{c}": at_end(self._comp[i]) for i, c in enumerate(COMPONENTS)}
        info = {"final_obs": obs.copy(), "_final_obs": done.copy(), "episode_end": stats, "_episode_end": done.copy()}
        self._start(idx)
        obs[idx] = self._observe()[idx]
        return obs, reward, terminated, truncated, info

    def _sim_error(self, idx: np.ndarray) -> None:
        """Envs idx came back from a step with a non-finite state (v1 saw a NaN position mid-episode, about once per
        100M steps). The episode ends on its last finite state with no reward, and the worker is replaced."""
        for i in idx:
            print(f"rlpilot: non-finite engine state, env {i} on {self.client.races[i]} after {self._len[i]} steps: "
                  "episode ended, worker replaced", flush=True)
        self._raw[idx] = self._last_raw[idx]
        self._sim_failed[idx] = True
        self.client.resets_since_spawn[idx] = self.recycle_resets  # respawn at the coming start

    def close_extras(self, **kwargs):
        self.client.close()
