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
    # "auto": the worker shifts at fixed points (90% of redline). "points": two extra actions set the worker's
    # upshift / downshift points (fractions of redline, mapped from [-1, 1] onto the ranges below).
    # "agent": one extra action shifts directly (> +shift_deadzone up, < -shift_deadzone down).
    gearbox: str = "auto"
    shift_up_range: tuple[float, float] = (0.7, 0.97)   # "points": upshift when the engine reaches this
    shift_down_range: tuple[float, float] = (0.4, 0.9)  # "points": downshift when the lower gear would be <= this
    shift_cooldown: float = 0.25     # s, "agent": minimum time between two agent shifts
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
