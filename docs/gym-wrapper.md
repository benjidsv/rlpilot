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
| 1. Dashboard | `eval/lap_time_best`, `eval/lap_time_median`, with `eval/bot_lap_time_best`/`_median` as the reference line (same car and track, built-in simplix bot); `eval/flying_lap_time_*` exclude lap 1 (standing start). `eval/fail_dist_from_start`, `eval/fail_lap`: where failed eval episodes ended. `eval/completion_rate`, `eval/progress`, `episode/progress` (distance / track length). `end/{off_track,stuck_crashed,timeout,lap_done}`. `reward/{progress,offtrack,damage,terminal,total}`. `ppo/{approx_kl,clip_fraction,entropy,explained_variance}` |
| 2. Behaviour | `behavior/{steer_delta,offtrack_frac,mean_speed,full_throttle_frac}` (training) and `eval/behavior/*`. `perf/{env_steps_per_s,total_steps_per_s,reset_share}` |
| 3. Artifacts, per eval | `eval/<timesteps>/track_map.png` (line coloured by speed), `speed_vs_distance.png` (agent vs bot on the same lap index, plus the cumulative time gap), `traces.npz` (every eval episode, every step) |

Everything else SB3 logs is under `sb3/`.

- **Evaluation** (`evaluate.py`, `EvalCallback`) uses the deterministic policy, `--eval-episodes` envs, and episodes ending after `--eval-laps` laps. The sim is deterministic, so identical episodes would add nothing. Episode 0 runs clean; the others get small action noise during the first second only, so completion rate and median lap time measure robustness. Lap times include the standing-start lap 1, and so do the bot's.
- **Bot reference** (`reference.py`): `rlpilot-worker -o <prefix>` (trace mode, no client) runs `data/raceman/rlref.xml`, which is the rlpilot race with simplix driving. It records 3 laps every 20 ms plus the track borders. The result is cached in `reference/`, keyed on the race file content. On Jarama the bot runs 111.6 s, then 107.4 s and 107.4 s. Simplix uses its own car setup; the agent uses the default one.
- **Termination groups:** `off_track` = all wheels off for 1 s. `stuck_crashed` = stuck, backwards, damage, or race ended by the engine. `timeout` = `max_episode_seconds`. `lap_done` = `laps_per_episode` reached; this counts as terminated and gets no penalty.
- **Behaviour definitions:** `steer_delta` = mean |Δsteer| per step, in action units. `offtrack_frac` = fraction of steps with ≥ 2 wheels off the road. `full_throttle_frac` = accel ≥ 0.95.

## Training notes

**pilot-1** (old defaults: `ent_coef=0`, 512 steps x 5 envs, minibatch 1024, constant lr 3e-4) peaked at 1-4M steps (3 laps completed in most evals, best lap 112.6 s vs the bot's 107.4 s) and then degraded: mean eval completion by 10M-step block was 0.62, 0.42, 0.35, 0.22. PPO health explains it: action std 0.19 -> 0.016, approx KL 0.018 -> 0.044, clip fraction 0.19 -> 0.30. The policy became near-deterministic and brittle.

Current defaults (`scripts/train_sb3.py`): `ent_coef=0.003`, `target_kl=0.02`, linear lr decay to 10% (`--lr-schedule`), 2048 steps x workers per update with minibatch 2048, 8 eval episodes. The best eval so far is saved to `best/` (score = -(median lap time, 150 s if none + 10 s x (1 - completion rate) + 10 s x missing laps of progress), i.e. pace first with a moderate reliability penalty); the tag `best_step` says when.

Where the gap to the bot is: at 3M steps the agent matches the bot's corner speeds and braking points, and loses ~6.7 s per lap evenly in acceleration zones (203 vs 231 km/h on the main straight). Hypotheses, untested: the automatic gearbox (shift points and shift time), wheel spin without traction control, 25 Hz vs the bot's 50 Hz control.

## Multiple tracks and the curvature lookahead

- **Tracks.** Track `t` is raced with `data/raceman/rlpilot-<t>.xml` (Jarama: `rlpilot.xml`) and has its bot reference in `rlref-<t>.xml`. Both are the Jarama files with only the track name and category changed. The downloaded tracks live in `sd-local/tracks/`.
- **Training on several tracks** (`--tracks a b c`, `EnvConfig.races`): each worker starts on a track and, at its first reset after `track_switch_steps` steps on it (default 200k, ~2 h 15 min of driving; a respawn costs ~380 ms of the whole batch, ~2% of the run at this default), respawns on a random one (~0.3 s, blocking the batch, hence not counted in resets). Workers can be on different tracks within the same batch.
- **Evaluation** (`--heldout-tracks x y`): every eval runs all training tracks plus the held-out ones (never trained on). The first training track keeps the `eval/*` metrics and artifacts. Others log `eval_track/<t>/*`, and the averages go to `eval_train/*` and `eval_heldout/*`. `flying_gap_pct` is the median flying lap vs the bot's best, in %. With several training tracks the best-checkpoint score is averaged over training tracks, with lap times in % of the bot.
- **Lookahead** (`--lookahead`, `EnvConfig.lookahead`): 9 extra observations, the mean centreline curvature over windows 0-10, 10-25, ..., 250-300 m ahead (x25, + = left, clipped to ±3). It's computed in Python from the bot reference's border samples (`track.py`), with each track precomputed on a 1 m grid (~2 µs per vector step). Its distance-from-start matches the engine's to ~1 m (Imola: up to 5.5 m). It's 2D only, with no elevation.
- **Why** (2026-10-07): Jarama-only checkpoints crashed at the first heavy braking zone on every other track, braking 40-60 m late. A probe showed the braking onset doesn't depend on speed at all: telling the policy it was 40% faster before a braking point made it brake in only 3% of cases. Shrinking the forward rays did trigger it. On one track the speed at each point is nearly constant, so the policy learned "brake when the road looks like this".

## Track library and car randomization

- **Tracks:** `scripts/download_tracks.py` fetches tracks from the official asset index (the same index and sha256 check as the in-game download manager) into `sd-local/tracks/`. `scripts/setup_tracks.py` writes the `rlpilot-<t>.xml` / `rlref-<t>.xml` race configs and runs the bot reference on each track, writing `reference/tracks.csv`. Status on 2026-10-07: 41 usable circuit/road/gprix tracks. Melbourne was dropped (the bot can't complete a lap). Karwada is a figure-eight (net turn 0, which is correct).
- **The bot is handicapped by simplix "skilling"** (`drivers/simplix/default.xml`, driver level 0.6). With it off, the bot is ~2% faster (Jarama 105.3 s, Braga 83.8 s). The references keep it on.
- **Car randomization** (`--car-random 0.1`, `EnvConfig.car_grip/car_mass/car_power`): per episode, the tyre mu (all compounds, dry and wet), the car mass and the engine torque curve are each scaled by a factor drawn in [1-x, 1+x]. The factors travel in shm act slots `RLP_A_CAR_*` (protocol v4). The worker scales the car's merged params in `rbNewTrack` (host API v2). A RESET restarts the race whenever the factors changed, including right after a spawn. `--car-obs` adds the factors to the observation as (f-1)x10. Evaluation always uses the stock car.
- **The downshift action** (proto-3) is used as a two-position switch: its output is beyond ±1 78% of the time, which is why its std drifted to the 0.5 cap without affecting behaviour. Forcing any fixed downshift point in the gearbox breaks the trained policy, even with the observation kept unchanged: its driving is co-adapted to its own gear timing. 34-63% of downshifts are decided by the `down <= up - 0.05` clamp, i.e. the request was "as early as possible".

## Performance (M1 Pro, 6 P + 2 E cores)

| Path | Env steps/s |
|---|---|
| C bench, 1 worker, 50 Hz control | 49k (~980× real time) |
| C bench, 5 workers, 50 Hz | **178k** (~3,560× real time) |
| Python VectorEnv, 5 workers, 25 Hz | ~46k |
| SB3 PPO, 5 workers, 200 µs spin (whole training loop) | ~6-8.6k |
| SB3 PPO, 20 workers, 50 µs spin, one-thread rollouts (current defaults) | **~10.5-12.3k** (rollout ≈ update time, ~0.45 s each per 10k samples) |

- **Env alone (C bench): use at most (performance cores − 1) workers.** Batches are lockstep, so a worker on an efficiency core, or more workers than cores, slows every env. Here 6 workers gave 80k steps/s against 178k for 5.
- **Whole PPO loop: the opposite.** Each vector step spends ~300 µs in SB3/torch and the env's Python, and the sim workers are idle ~75% of the time. So 20 workers with a short spin (50 µs, then sleep) amortize the per-step policy cost over 4x more envs. A long spin (1 ms) with more than 5 workers oversubscribes the cores and is much slower (4.2k at 8 workers). Measured 2026-10-07: 5→20 workers +40-50%, one-thread rollouts (`CpuRolloutPPO`) +13-33% at 20 workers. MPS for the updates was slower (SB3 syncs every minibatch). Keep n_steps x workers ≈ 10k samples per update.
- **Spin-waiting matters.** Without spinning, the C bench dropped to 69k. `spin_us` (default 200) is how long a waiter busy-waits before sleeping.
- **Python overhead is ~35 µs per vector step** (obs 9, reward 19, actions 4), independent of N.

## Gotchas found on the way

- **The engine holds the clutch fully down during the pre-start and never releases it.** A robot that doesn't drive `clutchCmd` never moves, so the worker slips the clutch out when pulling away.
- **Speed Dreams generates random bot drivers for every installed robot module, including `rlpilot`.** Robot indices are therefore not 0..K-1. The host maps them to slots in order of appearance.
- **Robot ticks fire when `currentTime - lastRobTime >= 0.02`, but float accumulation makes some ticks 22 ms.** An agent step is "about" `frame_skip × 20 ms`. Use `RLP_F_TIME` if exact time matters.
- **The engine eliminates non-human robots whose lap exceeds `84.5 + trackLen/10` s** (≈425 s on Jarama). Keep `max_episode_seconds` below that.
- **The engine rewrites race configs when it loads them**, so `workdirs.py` re-copies them from `data/` on every launch.

- **Memory grows ~0.14 MB per in-process restart** (measured with `footprint`; flat reset time, no file growth). `leaks` finds almost nothing unreachable, so the engine keeps something reachable across restarts. Workaround: `EnvConfig.recycle_resets` (default 2000) replaces a worker process after that many resets, which caps each worker near 300 MB. A respawned worker gives exactly the same episodes as a reset (tested).

## Next steps

1. **Find the per-restart memory growth** in the engine, then drop worker recycling.
2. **Resets.** About 1% of worker time in a PPO run, but they block the lockstep batch. If early training (short episodes) shows it, a small fork patch could skip results XML and robot/physics reloads.
3. **Opponents.** Add bots to `rlpilot.xml` and opponent sensors to the protocol. Then add K>1 agent cars per race: the shm is already `[worker][car]`, but `vec_env.py` assumes K=1.
4. **Rendered evaluation.** Run the same robot in a normal-display race. That needs the policy reachable from the robot without the worker host, e.g. a worker flag that opens a window.
