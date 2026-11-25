"""Reward and termination.

Reward per step, split into components so each can be logged:
- progress: PROGRESS per metre along the track (`metre_weight` reweights each metre: env.py's "bot_seconds" units pay
  each metre by the time the bot took over it). By default progress is signed (driving backwards costs): the
  2026-11-23 reward sweep kept the phase-1 reward. With `best_only`, only metres past the furthest point reached this
  episode pay: reversing costs nothing, and driving back over metres already covered pays nothing. Either way nothing
  is gained while all four wheels are off the road (no shortcuts across the grass).
- damage:   -DAMAGE per point of damage taken.
- control:  -lost_cost (default 0: off) each time the car loses control, by any of LOST_CAUSES: a new contact (damage
  from hitting a wall this step and not the last), starting to face backwards, or going more than FAR_OFF beyond the
  road's edge. Damage from the ground (bottoming out in a dip, a kerb, a landing: the engine's collision counter
  doesn't move) isn't a loss of control but a *scrape*: the damage component still charges it, and the evaluation
  counts it against a clean scenario (evaluate.py).
  Charged once, when the car enters the state, not for as long as it lasts: a long recovery mustn't cost more than
  giving up. A cause already true at the launch (a crashed start facing backwards) isn't charged. These are the
  evaluation's tests: a scenario that reached its distance but lost control is "recovered" (evaluate.py).
  `lost_causes` picks the causes charged; the states are tracked (and reported) whatever it is.
- terminal: -terminal (default TERMINAL) when an episode fails.
- wall:     -wall_cost x (speed / WALL_SPEED)^2 at each step with wall damage (default 0: off). Fuchs et al. 2021
  (Gran Turismo, RA-L) charge -c |v|^2 in wall contact: without it their car grinded along the walls in sharp curves,
  and a fixed cost made it either ignore the cost or brake to a standstill (here: lost_cost 2 made the car give up).
  The damage component barely charges a wall: a median contact costs ~0.5 m of progress (2026-11-24).
- offroad:  with all four wheels off, -offroad_cost x how far beyond the road's edge the car is (its centre's
  distance past the edge, less OFF_MARGIN, its half width; at most OFF_CAP) x the metres it drove this step (default
  0: off). Progress already pays nothing off the road, but the same whether 2 m or 15 m off: this makes a far
  excursion (Spa's runoffs) cost more than a brush, twice as far twice as much. Per metre driven, not per second: a
  cost per second made the early, near-random car (often far off) worse off than standing still on the road (a smoke
  run's returns fell from ~0 to -30..-50, 2026-11-25), a giving-up optimum. Per metre, a car stopped off the road
  pays nothing more than the no-progress terminal it gets anywhere, and driving back is always worth it. 0.01: a
  metre driven at OFF_CAP costs as much again as the progress it doesn't pay (0.1); 2 m out, +20%. The smoke run's
  early returns: ~0 without, ~-5 at 0.01, ~-10 at 0.02 (= standing still, rejected). Beyond 2 m from the edge
  the engine reports all four wheels off 100% of the time (2026-11-25 probe): this covers the eval's "far off".
- width:    +width_bonus x |to_middle| / half width (at most 1) x the metres of forward progress this step, when progress
  pays (not with all four wheels off; default 0: off). A bonus for using the road's width, either side: every policy
  so far enters corners from the middle (-0.05..+0.14 of the half width 50 m before the apex, the bot -0.38), and the
  2026-11-25 line probe found that steering this policy onto the bot's line costs ~0.2 s a corner (it brakes for a
  line it has never driven), so a policy improving locally never finds that wide pays. Paid on progress, not on time
  or distance: standing at the edge, or circling there, earns nothing. Sized under the cost of a bad line: a badly
  wide line costs ~0.3 s a corner, ~1.2 of reward at 40 m/s, and the bonus over a 150 m corner is width_bonus x 150,
  so it stays well below 0.007. 0.004: progress at the edge pays 4% more. Meant to fade (train.py --width-fade): it
  gets the car to wide states and lets it learn to drive from there, then the true reward decides.
  The 2026-11-25 30M run (0.004, faded by 15M): the car hugs the edges but still comes back to the middle on corner
  exits; lap clean 0.82 -> 0.67, pace unchanged (0.991 -> 0.994). It pays any edge, not the line.
- line:     +line_bonus x closeness to the bot's line x the metres of forward progress this step, when progress pays
  (default 0: off). closeness = 1 - |to_middle - the bot's offset there| / half width (at least 0): 1 on the bot's
  line, 0 a half width or more away. Unlike width, it pays the line's whole shape: wide entry, apex, and running out
  to the edge on the exit (where every policy so far comes back to the middle). Training only: the bot's line is never
  an input, so the policy still drives any track from its geometry. Meant to fade (train.py --line-fade) like width.
  Not potential-based on purpose: a potential (gamma Phi(s') - Phi(s)) leaves the advantages unchanged once the critic
  has learnt it, so it can't move a policy out of a local optimum; this is a curriculum, then the true reward decides.

Termination (fails the episode): no new best distance along the track for NO_PROGRESS_SECONDS (a spin, a reverse out
of the grass or a three-point turn are all fine, as long as the car gets going again), damage over MAX_DAMAGE, or the
engine ending the race. The time limit is a truncation, not a failure: the agent bootstraps through it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .client import F

PROGRESS = 0.1
DAMAGE = 0.001
LOST_COST = 0.0         # per loss of control (off: 2 made the car give up, 1 was within the seed noise)
TERMINAL = 10.0
FAR_OFF = 3.0           # m beyond the road's edge that counts as having lost control
NO_PROGRESS_SECONDS = 10.0
PROGRESS_STEP = 1.0     # m: a new best distance must beat the last one by this (jitter against a wall isn't progress)
MAX_DAMAGE = 5000.0
WALL_SPEED = 40.0       # m/s: wall_cost is the cost of a step of wall damage at this speed
OFF_MARGIN = 1.0        # m: the car's half width: with all four wheels off its centre is ~1 m past the edge
OFF_CAP = 10.0          # m: offroad_cost stops growing here (a car stranded 30 m out isn't 3x worse off)
COMPONENTS = ("progress", "damage", "control", "terminal", "wall", "offroad", "width", "line")
LOST_CAUSES = ("contact", "backwards", "far_off")
FAILURES = ("no_progress", "damage", "race_over", "sim_error")  # reason codes 0.. of terminated episodes


def lost_states(raw: np.ndarray, wall_hit: np.ndarray) -> np.ndarray:
    """[len(LOST_CAUSES), n] bool: which loss-of-control states each car is in (wall_hit: damage from a wall this
    step)."""
    hw = np.maximum(raw[:, F["track_width"]], 1.0) * 0.5
    return np.stack([wall_hit, np.cos(raw[:, F["angle"]]) < 0, np.abs(raw[:, F["to_middle"]]) > hw + FAR_OFF])


class Reward:
    def __init__(self, n: int, step_seconds: float, lost_cost: float = LOST_COST, best_only: bool = False,
                 terminal: float = TERMINAL, lost_causes: tuple[str, ...] = LOST_CAUSES,
                 metre_weight: Callable[[np.ndarray], np.ndarray] | None = None, wall_cost: float = 0.0,
                 offroad_cost: float = 0.0, width_bonus: float = 0.0, line_bonus: float = 0.0,
                 line_target: Callable[[np.ndarray], np.ndarray] | None = None):
        """metre_weight(raw) -> [n]: what each metre of progress is worth, relative to the others (None: 1).
        wall_cost: reward lost per step of wall damage at WALL_SPEED, x (speed / WALL_SPEED)^2. offroad_cost: reward
        lost per metre driven and per metre beyond the edge, with all four wheels off. width_bonus: reward per metre of
        progress at the road's edge, x |offset| / half width (the trainer may change it between steps to fade it; see
        the module docstring). line_bonus: reward per metre of progress on the bot's line, x closeness to it;
        line_target(raw) -> [n]: the bot's offset from the middle where each car is (needed when line_bonus > 0; the
        trainer may fade line_bonus too)."""
        if line_bonus > 0 and line_target is None:
            raise ValueError("line_bonus needs line_target")
        self.dt = step_seconds
        self.line_bonus, self.line_target = line_bonus, line_target
        self.wall_cost, self.offroad_cost, self.width_bonus = wall_cost, offroad_cost, width_bonus
        self.lost_cost, self.best_only, self.terminal, self.metre_weight = lost_cost, best_only, terminal, metre_weight
        unknown = set(lost_causes) - set(LOST_CAUSES)
        if unknown:
            raise ValueError(f"unknown causes of losing control: {sorted(unknown)} (known: {LOST_CAUSES})")
        self.charged = np.array([c in lost_causes for c in LOST_CAUSES])[:, None]
        self.raced = np.zeros(n, np.float64)      # m along the track this episode, signed
        self.peak = np.zeros(n, np.float64)       # furthest `raced` so far (best_only progress pays past it)
        self.best = np.zeros(n, np.float64)       # best `raced` so far, in PROGRESS_STEP increments (the clock's)
        self.since_best = np.zeros(n, np.float64)  # s since `best` last improved
        self.prev_dist = np.zeros(n, np.float64)
        self.prev_damage = np.zeros(n, np.float64)
        self.prev_coll = np.zeros(n, np.float64)
        self.d_damage = np.zeros(n, np.float64)    # damage taken this step
        self.wall_hit = np.zeros(n, bool)          # this step: damage, and the engine counted a wall collision
        self.ground_hit = np.zeros(n, bool)        # this step: damage without a wall collision (a scrape)
        self.lost = np.zeros((len(LOST_CAUSES), n), bool)   # loss-of-control states, this step
        self.onset = np.zeros((len(LOST_CAUSES), n), bool)  # entered this step
        self.components = np.zeros((len(COMPONENTS), n), np.float32)  # this step

    def reset(self, idx: np.ndarray, raw: np.ndarray) -> None:
        self.raced[idx] = self.peak[idx] = self.best[idx] = self.since_best[idx] = 0
        self.prev_dist[idx] = raw[idx, F["dist_raced"]]
        self.prev_damage[idx] = raw[idx, F["damage"]]
        self.prev_coll[idx] = raw[idx, F["coll_count"]]
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
        coll = raw[:, F["coll_count"]].astype(np.float64)
        walled = (coll > self.prev_coll) & on
        self.prev_coll = np.where(on, coll, self.prev_coll)
        self.d_damage = d_damage
        self.wall_hit, self.ground_hit = (d_damage > 0) & walled, (d_damage > 0) & ~walled
        self.raced += progress
        gained = np.maximum(self.raced - self.peak, 0)
        self.peak = np.maximum(self.peak, self.raced)

        lost = np.where(on, lost_states(raw, self.wall_hit), self.lost)
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
        if self.metre_weight is not None:
            paid = paid * self.metre_weight(raw)
        self.components[0] = PROGRESS * np.where(off & (paid > 0), 0.0, paid)
        self.components[1] = -DAMAGE * d_damage
        self.components[2] = -self.lost_cost * (self.onset & self.charged).sum(axis=0)
        self.components[3] = -self.terminal * terminated
        if self.wall_cost > 0:
            v2 = raw[:, F["speed_x"]].astype(np.float64) ** 2 + raw[:, F["speed_y"]].astype(np.float64) ** 2
            self.components[4] = -self.wall_cost * v2 / WALL_SPEED**2 * self.wall_hit
        if self.offroad_cost > 0:
            beyond = np.abs(raw[:, F["to_middle"]]) - np.maximum(raw[:, F["track_width"]], 1.0) * 0.5 - OFF_MARGIN
            v = np.hypot(raw[:, F["speed_x"]].astype(np.float64), raw[:, F["speed_y"]].astype(np.float64))
            self.components[5] = -self.offroad_cost * np.clip(beyond, 0.0, OFF_CAP) * v * self.dt * (off & on)
        if self.width_bonus > 0:
            side = np.minimum(np.abs(raw[:, F["to_middle"]]) / (np.maximum(raw[:, F["track_width"]], 1.0) * 0.5), 1.0)
            self.components[6] = self.width_bonus * side * np.where(off, 0.0, np.maximum(paid, 0.0))
        else:
            self.components[6] = 0.0  # it may have faded to 0 after paying
        if self.line_bonus > 0:
            hw = np.maximum(raw[:, F["track_width"]], 1.0) * 0.5
            close = np.maximum(1.0 - np.abs(raw[:, F["to_middle"]] - self.line_target(raw)) / hw, 0.0)
            self.components[7] = self.line_bonus * close * np.where(off, 0.0, np.maximum(paid, 0.0))
        else:
            self.components[7] = 0.0
        return self.components.sum(axis=0), terminated, reason
