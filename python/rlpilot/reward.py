"""Reward and termination.

Reward per step, split into components so each can be logged:
- progress: PROGRESS per metre along the track. With `best_only` (the default), only metres past the furthest point
  reached this episode pay: reversing costs nothing, and driving back over metres already covered pays nothing. Without
  it, progress is signed (driving backwards costs). Either way nothing is gained while all four wheels are off the road
  (no shortcuts across the grass).
- damage:   -DAMAGE per point of damage taken.
- control:  -lost_cost each time the car loses control, by any of LOST_CAUSES: a new contact (damage taken this step and
  not the last), starting to face backwards, or going more than FAR_OFF beyond the road's edge. Charged once, when the
  car enters the state, not for as long as it lasts: a long recovery mustn't cost more than giving up. A cause already
  true at the launch (a crashed start facing backwards) isn't charged. These are the evaluation's tests: a scenario that
  reached its distance but lost control is "recovered" (evaluate.py).
- terminal: -TERMINAL when an episode fails.

Termination (fails the episode): no new best distance along the track for NO_PROGRESS_SECONDS (a spin, a reverse out
of the grass or a three-point turn are all fine, as long as the car gets going again), damage over MAX_DAMAGE, or the
engine ending the race. The time limit is a truncation, not a failure: the agent bootstraps through it.
"""

from __future__ import annotations

import numpy as np

from .client import F

PROGRESS = 0.1
DAMAGE = 0.001
LOST_COST = 2.0         # per loss of control: 20 m of progress
TERMINAL = 10.0
FAR_OFF = 3.0           # m beyond the road's edge that counts as having lost control
NO_PROGRESS_SECONDS = 10.0
PROGRESS_STEP = 1.0     # m: a new best distance must beat the last one by this (jitter against a wall isn't progress)
MAX_DAMAGE = 5000.0
COMPONENTS = ("progress", "damage", "control", "terminal")
LOST_CAUSES = ("contact", "backwards", "far_off")
FAILURES = ("no_progress", "damage", "race_over", "sim_error")  # reason codes 0.. of terminated episodes


def lost_states(raw: np.ndarray, damage_taken: np.ndarray) -> np.ndarray:
    """[len(LOST_CAUSES), n] bool: which loss-of-control states each car is in."""
    hw = np.maximum(raw[:, F["track_width"]], 1.0) * 0.5
    return np.stack([damage_taken, np.cos(raw[:, F["angle"]]) < 0, np.abs(raw[:, F["to_middle"]]) > hw + FAR_OFF])


class Reward:
    def __init__(self, n: int, step_seconds: float, lost_cost: float = LOST_COST, best_only: bool = True):
        self.dt = step_seconds
        self.lost_cost, self.best_only = lost_cost, best_only
        self.raced = np.zeros(n, np.float64)      # m along the track this episode, signed
        self.peak = np.zeros(n, np.float64)       # furthest `raced` so far (best_only progress pays past it)
        self.best = np.zeros(n, np.float64)       # best `raced` so far, in PROGRESS_STEP increments (the clock's)
        self.since_best = np.zeros(n, np.float64)  # s since `best` last improved
        self.prev_dist = np.zeros(n, np.float64)
        self.prev_damage = np.zeros(n, np.float64)
        self.d_damage = np.zeros(n, np.float64)    # damage taken this step
        self.damage_taken = np.zeros(n, bool)      # this step
        self.lost = np.zeros((len(LOST_CAUSES), n), bool)   # loss-of-control states, this step
        self.onset = np.zeros((len(LOST_CAUSES), n), bool)  # entered this step
        self.components = np.zeros((len(COMPONENTS), n), np.float32)  # this step

    def reset(self, idx: np.ndarray, raw: np.ndarray) -> None:
        self.raced[idx] = self.peak[idx] = self.best[idx] = self.since_best[idx] = 0
        self.prev_dist[idx] = raw[idx, F["dist_raced"]]
        self.prev_damage[idx] = raw[idx, F["damage"]]
        self.lost[:, idx] = lost_states(raw[idx], np.zeros(len(idx), bool))  # the launch's states aren't onsets
        self.onset[:, idx] = False

    def step(self, raw: np.ndarray, active: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """raw [n, OBS_DIM] after the step -> reward [n], terminated [n], reason [n] (index in FAILURES, -1 = none).
        Envs not `active` (paused, not stepped) keep their state, get no reward and can't end."""
        on = np.ones(len(raw), bool) if active is None else active
        dist, damage = raw[:, F["dist_raced"]].astype(np.float64), raw[:, F["damage"]].astype(np.float64)
        # The engine's dist_raced jumps by a lap when a car crosses the start line backwards (and back when it recrosses):
        # remove anything lap-sized. A car can't cover half a lap in one step, so real driving is unchanged.
        length = np.maximum(raw[:, F["track_length"]], 1.0)
        progress = dist - self.prev_dist
        progress -= length * np.rint(progress / length)
        d_damage = np.maximum(damage - self.prev_damage, 0)
        progress, d_damage = progress * on, d_damage * on
        self.prev_dist = np.where(on, dist, self.prev_dist)
        self.prev_damage = np.where(on, damage, self.prev_damage)
        self.d_damage, self.damage_taken = d_damage, d_damage > 0
        self.raced += progress
        gained = np.maximum(self.raced - self.peak, 0)
        self.peak = np.maximum(self.peak, self.raced)

        lost = np.where(on, lost_states(raw, self.damage_taken), self.lost)
        self.onset = lost & ~self.lost & on
        self.lost = lost

        improved = self.raced >= self.best + PROGRESS_STEP
        self.best = np.where(improved, self.raced, self.best)
        self.since_best = np.where(improved, 0.0, self.since_best + self.dt * on)
        conds = np.stack([self.since_best >= NO_PROGRESS_SECONDS, damage >= MAX_DAMAGE, raw[:, F["race_over"]] > 0])
        terminated = conds.any(axis=0) & on
        reason = np.where(terminated, conds.argmax(axis=0), -1)

        off = raw[:, F["wheels_off"]] >= 4
        paid = gained if self.best_only else progress
        self.components[0] = PROGRESS * np.where(off & (paid > 0), 0.0, paid)
        self.components[1] = -DAMAGE * d_damage
        self.components[2] = -self.lost_cost * self.onset.sum(axis=0)
        self.components[3] = -TERMINAL * terminated
        return self.components.sum(axis=0), terminated, reason
