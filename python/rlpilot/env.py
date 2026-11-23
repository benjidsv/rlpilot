"""Gymnasium VectorEnv over rlpilot-worker processes: one solo race per worker, each on a fixed track.

- Each env is one worker on one track (see tracks.allocate for spreading workers over tracks by road length).
- Every episode starts from a start drawn by `start_fn` (default: random training starts, see starts.py). Disturbed
  starts run their forced steps inside the reset, so an episode begins at the handover.
- Same-step autoreset: when an env's episode ends, the observation returned for it is already the first of its next
  episode; the last one is in info["final_obs"] (dense [n, DIM], valid where info["_final_obs"]). Per-episode
  statistics are in info["episode_end"] (arrays [n], valid where info["_episode_end"]).
- Ends: a failure (reward.FAILURES) terminates; the time limit, or `max_distance` along the track (eval scenarios),
  truncates.
- `overlap_starts` (evaluation): an env whose episode ended launches its next one while the others keep driving. Its
  reset is sent without waiting for the answer (a race restart takes ~8 ms, several steps of the others), and its
  kick steps run in the others' exchanges. Otherwise every reset holds up every worker. Not gymnasium's autoreset:
  until `driving` is true again, the env's returned observation, reward and action are meaningless.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv

from . import obs as observation
from .client import CMD_NONE, CMD_RESET, CMD_STEP, Client, F
from .controls import HOLD, Controls
from .reward import COMPONENTS, FAILURES, LOST_CAUSES, LOST_COST, TERMINAL, Reward
from .starts import K, Starts, draw, train_kinds, write_launch
from .tracks import Track, TrackTable, race_for

ENDS = FAILURES + ("time_limit", "distance")  # end codes of info["episode_end"]["end"]
START_RETRIES = 4  # new draws for a start whose first observation isn't finite (rare launches); then the grid
REF_SPEED = 40.0   # m/s: progress_units="bot_seconds" pays a metre at this bot speed like a metre (the bot averages
                   # 39.3 m/s over the training road), so the costs keep their size against progress
NEAR_START = 50.0  # m: an episode that fails without ever getting this far down the road led nowhere (start too hard?)


class RacingEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, tracks: list[str], episode_seconds: float = 30.0, max_distance: float | None = None,
                 start_fn: Callable[[np.ndarray], Starts] | None = None,
                 retry_fn: Callable[[np.ndarray, int], Starts] | None = None, seed: int = 0, frame_skip: int = 2,
                 spin_us: int = 50, recycle_resets: int = 2000, run_dir=None, overlap_starts: bool = False,
                 lost_cost: float = LOST_COST, best_only_progress: bool = True, terminal: float = TERMINAL,
                 lost_causes: tuple[str, ...] = LOST_CAUSES, progress_units: str = "metres"):
        """tracks: one per worker. start_fn(env indices) -> their next Starts (default: random training starts).
        retry_fn(env indices, attempt) -> replacements for starts whose first observation wasn't finite (default: new
        random starts, then the grid). lost_cost, best_only_progress, terminal,
        lost_causes: see reward.py. progress_units: "metres", or "bot_seconds": each metre pays REF_SPEED / the bot's speed
        there, so a second at the bot's pace pays the same everywhere (and a metre at REF_SPEED as much as in metres)."""
        self.track_names = sorted(set(tracks))
        self.table = TrackTable([Track.load(t) for t in self.track_names])
        self.track = np.array([self.track_names.index(t) for t in tracks], np.int64)  # per env, index in track_names
        self.num_envs = n = len(tracks)
        self.dt = 0.02 * frame_skip
        self.max_steps = int(round(episode_seconds / self.dt))
        self.max_distance = max_distance
        self.recycle_resets = recycle_resets
        self.overlap_starts = overlap_starts
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
        if progress_units not in ("metres", "bot_seconds"):
            raise ValueError(f"progress_units: metres or bot_seconds, not {progress_units!r}")
        weight = self._bot_second_weight if progress_units == "bot_seconds" else None
        self.reward = Reward(n, self.dt, lost_cost, best_only_progress, terminal, lost_causes, weight)
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
        self._lost_first = np.full(n, -1)      # the first of LOST_CAUSES seen (-1 = none)
        self._lost_count = np.zeros((len(LOST_CAUSES), n), np.int64)  # times each was entered (not at the launch)
        self._damage = np.zeros(n)
        # overlap_starts: envs waiting for their launch, launched but not answered yet, forced steps left of their kick,
        # retries of their launch
        self._pending = np.zeros(n, bool)
        self._in_flight = np.zeros(n, bool)
        self._kick_left = np.zeros(n, np.int64)
        self._attempt = np.zeros(n, np.int64)
        self._next = self.starts[np.arange(n)]  # the start each pending env launches next (valid where _queued)
        self._queued = np.zeros(n, bool)

    @property
    def driving(self) -> np.ndarray:
        """[n] bool: envs in an episode, driven by the actions (overlap_starts: false while an env is launching)."""
        return ~self._pending & ~self._in_flight & (self._kick_left == 0)

    def _settle(self) -> None:
        """Wait for the launches still in flight (before anything synchronous: a reset, closing)."""
        if self._in_flight.any():
            self.client.wait(np.where(self._in_flight, CMD_RESET, CMD_NONE).astype(np.uint32))
            self._overlap_landed(np.flatnonzero(self._in_flight), np.zeros(0, np.int64))

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
        self._begin(idx)

    def _begin(self, idx: np.ndarray) -> None:
        """Envs idx are launched (and kicked): their episode starts at the current state."""
        self.reward.reset(idx, self._raw)
        self._last_raw[idx] = self._raw[idx]
        self._ret[idx] = self._len[idx] = self._comp[:, idx] = 0
        self._shifts[idx] = self._refused[idx] = self._off_steps[idx] = self._speed[idx] = 0
        self._ever_off[idx] = self._lost[idx] = False
        self._lost_first[idx], self._lost_count[:, idx], self._damage[idx] = -1, 0, 0

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

    def _overlap_launch(self, on: np.ndarray, cmds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """overlap_starts, before the exchange: pending envs get their launch and a RESET, sent right away (answered
        later), kicking envs their forced action and a STEP in `cmds`. Returns (landed, kicked) env indices: the
        launches answered since the last step, and the envs kicked in this one."""
        n = self.num_envs
        kicked = np.flatnonzero(on & (self._kick_left > 0))
        if len(kicked):
            action = np.zeros((n, 3), np.float32)
            action[:, 0], action[:, 1], action[:, 2] = self.starts.kick_steer, self.starts.kick_pedal, HOLD
            k = np.zeros(n, bool)
            k[kicked] = True
            self.controls.apply(action, self._raw, self._act, active=k)
            cmds[kicked] = CMD_STEP
        launched = np.flatnonzero(on & self._pending)
        if len(launched):
            fresh = launched[~self._queued[launched]]  # retries already have their start
            if len(fresh):
                for f, v in self.start_fn(fresh).to_columns().items():
                    getattr(self._next, f)[fresh] = v
            self._queued[launched] = False
            starts = self._next[launched]
            for f, v in starts.to_columns().items():
                getattr(self.starts, f)[launched] = v
            launch = self._act[launched].copy()
            write_launch(starts, self.track[launched], self.table, launch)
            self._act[launched] = launch
            self._pending[launched] = False
            due = launched[self.client.resets_since_spawn[launched] >= self.recycle_resets]
            for i in due:
                self.client.respawn(int(i))  # rare: blocks, and comes back launched
            send = np.setdiff1d(launched, due)
            reset = np.full(self.num_envs, CMD_NONE, np.uint32)
            reset[send] = CMD_RESET
            self.client.send(reset)
            self._in_flight[send] = True
            landed = due
        else:
            landed = np.zeros(0, np.int64)
        return landed, kicked

    def _overlap_landed(self, launched: np.ndarray, kicked: np.ndarray) -> None:
        """overlap_starts, after the exchange: launches answered by now land and start their kick, and envs done
        launching begin their episode, or retry if their state isn't finite (as in _start)."""
        fl = np.flatnonzero(self._in_flight)
        if len(fl):
            done = fl[np.array([self.client.answered(int(i)) for i in fl], bool)]
            self._in_flight[done] = False
            launched = np.union1d(launched, done)
        if len(launched):
            self.controls.reset(launched, self._raw)
            self._kick_left[launched] = self.starts.kick_steps[launched]
        self._kick_left[kicked] -= 1
        ready = np.concatenate([launched, kicked])
        ready = ready[self._kick_left[ready] == 0]
        if not len(ready):
            return
        bad = ~np.isfinite(self._raw[ready]).all(axis=1)
        if bad.any():
            b = ready[bad]
            if (self._attempt[b] >= START_RETRIES).any():
                raise RuntimeError(f"non-finite first observations after {START_RETRIES} retries: envs {b.tolist()}")
            self.start_retries += len(b)
            for a in np.unique(self._attempt[b]):  # retry_fn takes one attempt number per call
                envs = b[self._attempt[b] == a]
                for f, v in self.retry_fn(envs, int(a)).to_columns().items():
                    getattr(self._next, f)[envs] = v
            self._attempt[b] += 1
            self._pending[b] = self._queued[b] = True  # relaunched at the next step
        good = ready[~bad]
        self._attempt[good] = 0
        self._begin(good)

    # ---- gymnasium API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._settle()
        self._pending[:], self._kick_left[:] = False, 0
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
        asked = np.ones(n, bool) if active is None else np.asarray(active, bool)
        on = asked & self.driving  # simulated and driven by the actions
        cmds = np.where(on, CMD_STEP, CMD_NONE).astype(np.uint32)
        if self.overlap_starts:  # first: the kick's controls.apply would overwrite this step's shift flags
            landed, kicked = self._overlap_launch(asked & ~on, cmds)
        self.controls.apply(np.asarray(actions, np.float32).reshape(n, 3), self._raw, self._act, on)
        self.client.exchange(cmds)
        if self.overlap_starts:
            self._overlap_landed(landed, kicked)
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
        off = raw[:, F["wheels_off"]] >= 4
        lost = self.reward.lost.any(axis=0)  # the reward's tests (reward.lost_states), so both agree
        self._ret += reward * on
        self._len += on
        self._comp += self.reward.components * on
        self._shifts += self.controls.shifted
        self._refused += self.controls.refused
        self._off_steps += off & on
        self._speed += raw[:, F["speed_x"]] * on
        self._ever_off |= off & on
        first = lost & on & (self._lost_first < 0)
        self._lost_first[first] = self.reward.lost[:, first].argmax(axis=0)
        self._lost |= lost & on
        self._lost_count += self.reward.onset
        self._damage += self.reward.d_damage

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
            "lost_first": np.where(done, self._lost_first, -1), "damage": at_end(self._damage),
        } | {f"lost_{c}": at_end(self._lost_count[i]) for i, c in enumerate(LOST_CAUSES)} \
          | {f"reward_{c}": at_end(self._comp[i]) for i, c in enumerate(COMPONENTS)}
        info = {"final_obs": obs.copy(), "_final_obs": done.copy(), "episode_end": stats, "_episode_end": done.copy()}
        if self.overlap_starts:
            self._pending[idx] = True  # launched during the next steps
        else:
            self._start(idx)
            obs[idx] = self._observe()[idx]
        return obs, reward, terminated, truncated, info

    def _bot_second_weight(self, raw: np.ndarray) -> np.ndarray:
        v = self.table.at(self.track, raw[:, F["dist_from_start"]].astype(np.float64)[:, None], "bot_speed")[:, 0]
        return REF_SPEED / np.maximum(v, 5.0)

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
        self._settle()
        self.client.close()
