"""Vectorized reward and termination from raw shared-memory fields.

Kept in Python on purpose: shaping changes need no rebuild, and over a batch of envs this is a few
numpy ops per step. The reward is returned split into components so they can be logged separately
(total return hides reward hacking)."""

from __future__ import annotations

import numpy as np

from ._native import F
from .config import RewardConfig

# Termination reasons (index = code in info["stats"]["term_reason"]; -1 = time limit / truncation).
TERM_REASONS = ("offtrack", "stuck", "backwards", "damage", "race_over", "lap_done")
LAP_DONE = TERM_REASONS.index("lap_done")
# Coarse groups for the dashboard.
END_GROUPS = {"offtrack": "off_track", "stuck": "stuck_crashed", "backwards": "stuck_crashed",
              "damage": "stuck_crashed", "race_over": "stuck_crashed", "lap_done": "lap_done",
              "time_limit": "timeout"}
REWARD_COMPONENTS = ("progress", "offtrack", "damage", "terminal", "steer", "line", "steer_change", "steer_reversal")


class RewardTracker:
    def __init__(self, n: int, cfg: RewardConfig, step_seconds: float, laps_per_episode: int = 0):
        self.cfg, self.dt, self.laps_per_episode = cfg, step_seconds, laps_per_episode
        self.prev_dist = np.zeros(n, np.float32)
        self.raced = np.zeros(n, np.float64)  # m along the track this episode, without lap-sized jumps (see step)
        self.prev_damage = np.zeros(n, np.float32)
        self.off_t = np.zeros(n, np.float32)
        self.stuck_t = np.zeros(n, np.float32)
        self.back_t = np.zeros(n, np.float32)
        self.components = np.zeros((len(REWARD_COMPONENTS), n), np.float32)
        self.line_k = cfg.line_shaping  # mutable: training can fade it
        self.prev_dev = np.zeros(n, np.float32)  # |offset from the racing line| (m) at the previous state
        self.prev_change = np.zeros(n, np.float32)  # steering change of the previous step
        self.reversed = np.zeros(n, bool)  # this step's reversal flags: consecutive changes of opposite signs, both
                                           # above the dead-band

    def reset(self, raw: np.ndarray, idx: np.ndarray | slice = slice(None), line_dev: np.ndarray | None = None) -> None:
        """line_dev: |offset from the racing line| (m) of the envs idx (all n if idx is a slice)."""
        if line_dev is not None:
            self.prev_dev[idx] = line_dev
        self.prev_dist[idx] = self.raced[idx] = raw[idx, F["dist_raced"]]
        self.prev_damage[idx] = raw[idx, F["damage"]]
        self.prev_change[idx] = 0  # the first change of an episode can't be a reversal
        self.off_t[idx] = self.stuck_t[idx] = self.back_t[idx] = 0

    def step(self, raw: np.ndarray, d_steer: np.ndarray | float = 0.0, line_dev: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """raw [n, OBS_DIM], d_steer [n] change of the steer action since the previous step, line_dev [n]
        |offset from the racing line| (m) of the new state (needed when line_k > 0).
        Returns reward [n], terminated [n] bool, reason [n] int (-1 = none, else TERM_REASONS index).
        The per-component rewards of this step are left in self.components [n_components, n]."""
        c, dt = self.cfg, self.dt
        dist, damage = raw[:, F["dist_raced"]], raw[:, F["damage"]]
        wheels_off = raw[:, F["wheels_off"]]
        # The engine's dist_raced = (laps - 1) x length + dist_from_start jumps by a lap when a car crosses the start line
        # backwards (no lap is counted, but dist_from_start wraps), and back when it recrosses. A policy found this: creep
        # off the grid, spin at full lock across the line, collect a lap of reward. Remove anything lap-sized: a car can't
        # cover half a lap in one step, so real driving is unchanged.
        length = raw[:, F["track_length"]]
        progress = dist - self.prev_dist
        progress = progress - length * np.rint(progress / np.maximum(length, 1.0))
        self.raced += progress
        d_damage = damage - self.prev_damage
        self.prev_dist[:], self.prev_damage[:] = dist, damage

        self.off_t = np.where(wheels_off >= 4, self.off_t + dt, 0)
        self.stuck_t = np.where(raw[:, F["speed_x"]] < c.stuck_speed, self.stuck_t + dt, 0)
        self.back_t = np.where(np.cos(raw[:, F["angle"]]) < 0, self.back_t + dt, 0)

        # laps counts start-line crossings; the first one (from the grid) is not a completed lap
        laps_done = (raw[:, F["laps"]] - 1 >= self.laps_per_episode) if self.laps_per_episode > 0 \
            else np.zeros(len(raw), bool)
        conds = np.stack([self.off_t >= c.offtrack_patience,
                          self.stuck_t >= c.stuck_patience,
                          self.back_t >= c.backwards_patience,
                          damage >= c.max_damage,
                          raw[:, F["race_over"]] > 0,
                          laps_done])
        terminated = conds.any(axis=0)
        reason = np.where(terminated, conds.argmax(axis=0), -1)
        failed = terminated & (reason != LAP_DONE)

        comp = self.components
        comp[0] = c.progress * progress
        comp[1] = -c.offtrack * wheels_off / 4
        comp[2] = -c.damage * d_damage
        comp[3] = -c.terminal * failed
        comp[4] = -c.steer_rate * np.square(d_steer)
        comp[6] = -c.steer_change * np.abs(d_steer)
        self.reversed = (d_steer * self.prev_change < 0) & (np.abs(d_steer) > c.steer_deadband) \
            & (np.abs(self.prev_change) > c.steer_deadband)
        comp[7] = -c.steer_reversal * self.reversed
        self.prev_change[:] = d_steer
        if line_dev is not None and self.line_k > 0:
            # both potentials with the current k, so fading it never creates a one-off jump
            phi = -self.line_k * np.minimum(line_dev, c.line_cap) * ~terminated
            prev = -self.line_k * np.minimum(self.prev_dev, c.line_cap)
            comp[5] = c.gamma * phi - prev
        else:
            comp[5] = 0
        if line_dev is not None:
            self.prev_dev[:] = line_dev
        return comp.sum(axis=0), terminated, reason
