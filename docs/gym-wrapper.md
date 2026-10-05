# Gym wrapper

A Gymnasium `VectorEnv` over headless Speed Dreams, built to cost as little as possible on top of the simulator. SB3 and MLflow sit on top.

## Design

```
Python (SB3 / trainer)
  SpeedDreamsVecEnv ── numpy views on one shared-memory block ── librlpilot_client (ctypes, no GIL)
        │  write actions → bump cmd seq → wait for done seq (spin, then os_sync_wait_on_address)
        ▼
N × rlpilot-worker (pure C++, one solo race each)
  own host instead of main.cpp + textonly: pumps updateState(), restartRace() on reset
  robot module "rlpilot": forwards rbDrive to the host, which writes obs and waits for the action
```

- **One simulator per process.** The engine is full of globals, so parallelism comes from processes, never threads.
- **No Python in the workers, one native call per vector step.** Gymnasium's `AsyncVectorEnv` costs 50–200 µs per env per step, while one 20 ms sim step costs about 12 µs.
- **Reward and termination are computed in Python** (`reward.py`), vectorized over the batch. Changing them needs no rebuild.
- **The C++ side exports raw signals only** (`native/protocol.h`, `RLP_F_*`), plus 19 track-edge rays. The rays use sphere tracing over `RtTrackGlobal2Local` and are accurate to < 0.3 m.
- **Resets happen in-process** with `restartRace()`: ~7.5 ms, plus the engine's 2 s countdown (~1 ms of sim). Relaunching the process would cost ~300 ms.
- **Same-step autoreset.** Episode stats are only built when an episode ends.

## Files

| | |
|---|---|
| `native/protocol.h` | Shared-memory layout and obs/action fields. Python parses it, so add fields **only here** and bump `RLP_VERSION`. |
| `native/sync.h` | Cross-process wait/wake (spin, then futex). |
| `native/worker/main.cpp` | Worker host: engine startup, decision ticks, obs, rays, auto gear and clutch, resets. |
| `native/robot/rlpilot.cpp` | Robot shim. Without a host (the normal game) it drives a centreline follower. |
| `native/client/client.c`, `native/bench/bench.c` | Client library and a C-only benchmark. |
| `data/raceman/rlpilot.xml`, `rlref.xml`, `data/drivers/rlpilot/` | Master race configs (agent / bot reference) and driver config, copied into each worker's `-lc` dir. |
| `python/rlpilot/` | `config.py`, `vec_env.py`, `features.py`, `reward.py`, `sb3.py`, `workdirs.py`; tracking: `tracking.py`, `evaluate.py`, `reference.py`, `plots.py` |
| `scripts/` | `bench_env.py`, `train_sb3.py` |

## Build and run

```sh
./build-macos.sh        # engine, then native/ (cmake build + install into sd-install/), then the uv venv
.venv/bin/python -m pytest tests                  # determinism, rays, autoreset, lap_done, bot reference
.venv/bin/python scripts/bench_env.py --workers 1 5
.venv/bin/python scripts/train_sb3.py --workers 5 --timesteps 2000000 --eval-every 100000
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`cmake --install` puts the worker in `sd-install/games/` on purpose. The engine derives its lib and data dirs from the executable's location. Native binaries are linked with their install rpaths and only run from there.

## Experiment tracking (MLflow)

Each run gets the full `EnvConfig` and the algorithm hyperparameters as params, plus tags for the rlpilot and engine git SHAs, a dirty flag, and the protocol version. Metric prefixes are how the MLflow UI groups charts. Nothing is logged per env step: episode stats are only materialized when an episode ends.

| Tier | Metrics |
|---|---|
| 1. Dashboard | `eval/lap_time_best`, `eval/lap_time_median`, with `eval/bot_lap_time_best`/`_median` as the reference line (same car and track, built-in simplix bot). `eval/completion_rate`, `eval/progress`, `episode/progress` (distance / track length). `end/{off_track,stuck_crashed,timeout,lap_done}`. `reward/{progress,offtrack,damage,terminal,total}`. `ppo/{approx_kl,clip_fraction,entropy,explained_variance}` |
| 2. Behaviour | `behavior/{steer_delta,offtrack_frac,mean_speed,full_throttle_frac}` (training) and `eval/behavior/*`. `perf/{env_steps_per_s,total_steps_per_s,reset_share}` |
| 3. Artifacts, per eval | `eval/<timesteps>/track_map.png` (line coloured by speed), `speed_vs_distance.png` (agent vs bot on the same lap index, plus the cumulative time gap), `traces.npz` (every eval episode, every step) |

Everything else SB3 logs is under `sb3/`.

- **Evaluation** (`evaluate.py`, `EvalCallback`) uses the deterministic policy, `--eval-episodes` envs, and episodes ending after `--eval-laps` laps. The sim is deterministic, so identical episodes would add nothing. Episode 0 runs clean; the others get small action noise during the first second only, so completion rate and median lap time measure robustness. Lap times include the standing-start lap 1, and so do the bot's.
- **Bot reference** (`reference.py`): `rlpilot-worker -o <prefix>` (trace mode, no client) runs `data/raceman/rlref.xml`, which is the rlpilot race with simplix driving. It records 3 laps every 20 ms plus the track borders. The result is cached in `reference/`, keyed on the race file content. On Jarama the bot runs 111.6 s, then 107.4 s and 107.4 s. Simplix uses its own car setup; the agent uses the default one.
- **Termination groups:** `off_track` = all wheels off for 1 s. `stuck_crashed` = stuck, backwards, damage, or race ended by the engine. `timeout` = `max_episode_seconds`. `lap_done` = `laps_per_episode` reached; this counts as terminated and gets no penalty.
- **Behaviour definitions:** `steer_delta` = mean |Δsteer| per step, in action units. `offtrack_frac` = fraction of steps with ≥ 2 wheels off the road. `full_throttle_frac` = accel ≥ 0.95.

## Performance (M1 Pro, 6 P + 2 E cores)

| Path | Env steps/s |
|---|---|
| C bench, 1 worker, 50 Hz control | 49k (~980× real time) |
| C bench, 5 workers, 50 Hz | **178k** (~3,560× real time) |
| Python VectorEnv, 5 workers, 25 Hz | ~46k |
| SB3 PPO, 5 workers (whole training loop) | ~6k. The learner is the bottleneck. |

- **Use at most (performance cores − 1) workers.** Batches are lockstep, so a worker on an efficiency core, or more workers than cores, slows every env. Here 6 workers gave 80k steps/s against 178k for 5.
- **Spin-waiting matters.** Without spinning, the C bench dropped to 69k. `spin_us` (default 200) is how long a waiter busy-waits before sleeping.
- **Python overhead is ~35 µs per vector step** (obs 9, reward 19, actions 4), independent of N.

## Gotchas found on the way

- **The engine holds the clutch fully down during the pre-start and never releases it.** A robot that doesn't drive `clutchCmd` never moves, so the worker slips the clutch out when pulling away.
- **Speed Dreams generates random bot drivers for every installed robot module, including `rlpilot`.** Robot indices are therefore not 0..K-1. The host maps them to slots in order of appearance.
- **Robot ticks fire when `currentTime - lastRobTime >= 0.02`, but float accumulation makes some ticks 22 ms.** An agent step is "about" `frame_skip × 20 ms`. Use `RLP_F_TIME` if exact time matters.
- **The engine eliminates non-human robots whose lap exceeds `84.5 + trackLen/10` s** (≈425 s on Jarama). Keep `max_episode_seconds` below that.
- **The engine rewrites race configs when it loads them**, so `workdirs.py` re-copies them from `data/` on every launch.

## Next steps

1. **Resets.** About 1% of worker time in a PPO run, but they block the lockstep batch. If early training (short episodes) shows it, a small fork patch could skip results XML and robot/physics reloads.
2. **Opponents.** Add bots to `rlpilot.xml` and opponent sensors to the protocol. Then add K>1 agent cars per race: the shm is already `[worker][car]`, but `vec_env.py` assumes K=1.
3. **Rendered evaluation.** Run the same robot in a normal-display race. That needs the policy reachable from the robot without the worker host, e.g. a worker flag that opens a window.
