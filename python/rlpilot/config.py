"""Experiment-facing configuration. Everything that changes env behaviour lives here so that one
`asdict(cfg)` fully describes an env for experiment tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 0.1          # per metre of distance raced
    offtrack: float = 0.05         # per step, scaled by the fraction of wheels off the road
    damage: float = 0.001          # per damage point
    terminal: float = 10.0         # penalty when an episode fails (terminated other than by lap_done)
    steer_rate: float = 0.02       # per step, x (change of the steer action)^2: proto-4's deterministic steer
                                   # flipped sign every step (steer_delta 0.5 -> 0.005/step); smooth driving (~0.05)
                                   # costs ~nothing. It also taxes exploration noise (std 0.5: ~0.01/step), so 0.1
                                   # outweighed the terminal penalty in early (noisy) episodes.
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
    # Several races: each worker drives one, drawn at random, and switches to another random one every
    # track_switch_steps agent steps on a track, at its next reset (by respawning, ~0.3 s, which blocks the whole
    # batch: counting resets instead made early training, with its short episodes, respawn constantly).
    # Empty: every worker drives `race`.
    races: tuple[str, ...] = ()
    track_switch_steps: int = 200_000  # ~2 h 15 min of driving at 25 Hz; a respawn costs ~380 ms of the whole batch
    lookahead: bool = False          # add the centreline curvature over windows ahead (track.LOOKAHEAD_EDGES)
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
