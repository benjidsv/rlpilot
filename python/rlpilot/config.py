"""Experiment-facing configuration. Everything that changes env behaviour lives here so that one
`asdict(cfg)` fully describes an env for experiment tracking."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, fields


REMOVED_SETTINGS_OFF = {"steer_filter": "1.0", "pedal_filter": "1.0"}  # logged by older runs, with their "off" values


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 0.1          # per metre of distance raced
    offtrack_progress: float = 1.0  # share of forward progress paid with all four wheels off the road. 0: an excursion
                                    # costs what it drives, and cutting across the grass earns nothing
    offtrack: float = 0.05         # per step, scaled by the fraction of wheels off the road
    damage: float = 0.001          # per damage point
    terminal: float = 10.0         # penalty when an episode fails (terminated other than by lap_done)
    # Potential-based shaping towards the bot's racing line: Phi = -line_shaping x min(|offset from the line|,
    # line_cap) (m), reward += gamma x Phi(next) - Phi(current), Phi = 0 at a terminal state. Over an episode it
    # sums to ~-Phi(start), so it leaves the optimal policy unchanged (Ng et al. 1999) and only moves credit earlier.
    # The cap keeps it from pushing back from far off the line. 0: off. Training can fade it (set_line_shaping).
    line_shaping: float = 0.0      # per metre
    line_cap: float = 3.0          # m
    gamma: float = 0.995           # shaping discount; must equal the algorithm's gamma
    steer_rate: float = 0.02       # per step, x (change of the steer action)^2: proto-4's deterministic steer
                                   # flipped sign every step (steer_delta 0.5 -> 0.005/step); smooth driving (~0.05)
                                   # costs ~nothing. It also taxes exploration noise (std 0.5: ~0.01/step), so 0.1
                                   # outweighed the terminal penalty in early (noisy) episodes.
    # Penalties on the steering change the policy actually made (steer_limit mode: its own output). Weights are chosen
    # against reward/progress (~0.14 per step at mean speed); see docs/research-notes-proto7.md for the smoke-run numbers.
    steer_change: float = 0.0      # per step, x |change of steering| (command units, 1 = full lock)
    steer_reversal: float = 0.0    # per step where two consecutive changes reverse sign, both above the dead-band
    steer_deadband: float = 0.015  # command units per step; the 0.1 std floor is ~0.006 per step at limit 0.0625
    # termination rules (seconds of sim time)
    offtrack_patience: float = 1.0   # all four wheels off the road for this long
    stuck_patience: float = 4.0      # slower than stuck_speed for this long
    stuck_speed: float = 1.0         # m/s
    backwards_patience: float = 1.0  # facing more than 90 degrees from the track direction
    max_damage: float = 5000.0


@dataclass(frozen=True)
class EnvConfig:
    n_workers: int = 4
    frame_skip: int = 2              # robot ticks (20 ms) per agent step -> 25 Hz control
    race: str = "rlpilot"            # data/raceman/<race>.xml
    # Several races: worker i drives races[i % len(races)]. With n_workers a multiple of len(races), every update
    # holds every track equally (proto-7), and recycling a worker keeps its track. Empty: every worker drives `race`.
    races: tuple[str, ...] = ()
    # > 0: a worker respawns on a random race at its first reset after this many agent steps on a track (proto-3..6
    # used 200k, ~2 h 15 min of driving; a respawn costs ~380 ms of the whole batch). 0: workers never change track.
    track_switch_steps: int = 0
    lookahead: bool = False          # add the centreline curvature over windows ahead (track.LOOKAHEAD_EDGES)
    line_obs: bool = False           # add the bot's racing line at track.LINE_POINTS ahead, relative to the car
                                     # ((line - to_middle) / half width): the line through the next 200 m
    # Road edges ahead, in the car's frame, from the track map only (features.make_layout has frame, signs, units).
    edge_points: int = 0             # points per edge (0: off); at equal spacing of centreline distance, the first
                                     # one abreast of the car and the last one at max(horizon x speed, floor)
    edge_horizon: float = 6.0        # s of travel at the current speed
    edge_floor: float = 60.0         # m, the span near standstill (= 6 s at 36 km/h)
    # Road height ahead, relative to the road at the car, from the track map only. Over the same span as the edges (and at
    # the same places when this is edge_points - 1); the point abreast of the car is left out, since it always reads 0.
    height_points: int = 0
    # The car's acceleration in its own frame (x forward, y left, z up; m/s^2), as the engine computes it from the net force:
    # what a driver feels as weight transfer. The policy has no memory, so it can't get this from two consecutive speeds.
    accel_obs: bool = False
    # The clocks of the termination rules: time off the road (all 4 wheels), facing backwards and below stuck_speed, each over its
    # rule's patience. A driver knows how long they've been on the grass; without this, the agent can't tell 0.5 s from 4.9 s.
    timer_obs: bool = False
    lookahead_stat: str = "peak"     # per window: "peak" (signed max |curvature|) or "mean" (proto-3/4; dilutes
                                     # short tight corners, e.g. Spa's Bus Stop read -0.82 for a -1.34 peak)
    # Car randomization: factors drawn uniformly per episode (applied when the race restarts, ~ms).
    car_grip: tuple[float, float] = (1.0, 1.0)   # x tyre friction
    car_mass: tuple[float, float] = (1.0, 1.0)   # x car mass
    car_power: tuple[float, float] = (1.0, 1.0)  # x engine torque
    car_obs: bool = False            # observe the factors, as (factor - 1) x CAR_OBS_SCALE (+-10% -> +-1)
    # "auto": the worker shifts at fixed points (90% of redline). "points": two extra actions set the worker's
    # upshift / downshift points (fractions of redline, mapped from [-1, 1] onto the ranges below).
    # "agent": one extra action shifts directly (> +shift_deadzone up, < -shift_deadzone down).
    gearbox: str = "auto"
    shift_up_range: tuple[float, float] = (0.7, 0.97)   # "points": upshift when the engine reaches this
    shift_down_range: tuple[float, float] = (0.4, 0.9)  # "points": downshift when the lower gear would be <= this
    shift_cooldown: float = 0.25     # s, "agent": minimum time between two agent shifts
    # "points": after an upshift at engine rpm r (fraction of redline), the downshift point is capped at r - this
    # until the next gear change, so an upshift is only undone once the car has slowed down. proto-5 flicked its
    # upshift point to the floor and back, upshifting and re-downshifting ~20 times a minute under throttle.
    # None: off (proto-1..5).
    upshift_hold_gap: float | None = 0.05
    shift_deadzone: float = 0.33     # "agent"
    separate_pedals: bool = False    # action = [steer, pedal] (pedal < 0 brakes) or [steer, accel, brake]
    # None: the steer action is the steering command (up to proto-6). Else it is a change:
    # steer_t = clip(steer_t-1 + action x steer_limit, -1, 1), starting from 0 in each episode. 0.0625 per 40 ms step
    # = 70 deg/s, which the bot exceeds on 0.77% of its steps; the sim's own wheel slew cap is 0.107 per step.
    steer_limit: float | None = None
    steer_history: int = 0           # observe the last N steering changes (needs steer_limit), most recent first
    # Random launches: this share of a worker's training samples comes from episodes that start anywhere on the lap, the
    # rest from the grid. Each worker picks at every reset whichever of the two is behind, so every track gets both. A
    # launched car faces down the road, in a gear for its speed. 0: grid only, and no random draws are made for it.
    launch_share: float = 0.0
    launch_speed: tuple[float, float] = (0.0, 29.0)  # m/s, uniform (29 m/s = 105 km/h, the range of Sony's 2024 vision paper)
    # Uniform over this share of the road's width, around its middle. proto-8 used 0.8 until 130.9M: 0.10 from the right edge
    # with some heading can come back from placement with a NaN position, about 1 launch in 10,000 (see vec_env._retry_bad_starts).
    launch_lateral: float = 0.7
    launch_yaw: float = 0.1          # rad, uniform within +- this of the track direction
    max_episode_seconds: float = 300.0  # sim time; keep below the engine's slow-lap elimination (~425 s on Jarama)
    laps_per_episode: int = 0        # > 0: completing this many laps ends the episode (terminated, "lap_done")
    deterministic: bool = True       # -nr: no srand(time) in the engine
    spin_us: int = 200               # busy-wait budget before sleeping (workers wait while Python runs)
    recycle_resets: int = 2000       # replace a worker process after this many resets (engine grows ~0.15 MB/reset)
    run_dir: str | None = None       # worker local dirs; temp dir if None
    reference_race: str = "rlref"    # same track and car driven by a built-in bot (evaluation baseline)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def step_seconds(self) -> float:
        return 0.02 * self.frame_skip

    def flat(self) -> dict[str, object]:
        """Flat dict for experiment tracking (nested dataclasses become dotted keys)."""
        out = {}

        def walk(prefix, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(f"{prefix}{k}.", v)
                else:
                    out[prefix + k] = v

        walk("", asdict(self))
        return out

    @classmethod
    def from_flat(cls, params: dict[str, str], prefix: str = "") -> "EnvConfig":
        """Rebuild a config from flat() as logged by experiment tracking, where every value became a string. A logged
        setting this code no longer has would silently give a different controller, so it is refused unless it was off."""
        def read(dc, pre):
            out = {}
            for f in fields(dc):
                if f.name != "reward" and pre + f.name in params:
                    try:
                        out[f.name] = ast.literal_eval(params[pre + f.name])
                    except (ValueError, SyntaxError):  # a plain string
                        out[f.name] = params[pre + f.name]
            return out

        known = {prefix + f.name for f in fields(cls)} | {f"{prefix}reward.{f.name}" for f in fields(RewardConfig)}
        for key, value in params.items():
            if key.startswith(prefix) and key not in known and REMOVED_SETTINGS_OFF.get(key[len(prefix):]) != value:
                raise ValueError(f"this code can't reproduce {key}={value}")
        return cls(**read(cls, prefix), reward=RewardConfig(**read(RewardConfig, f"{prefix}reward.")))
