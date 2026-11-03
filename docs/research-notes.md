# Research notes

Experiment log and plans. How the code works belongs in [gym-wrapper.md](gym-wrapper.md); what we tried, why, and what we decided belongs here.

## Goal

**True generalization.** Held-out completion comes first and pace second. Beating the bot is not the goal in itself, so per-track fine-tuning is out. Opponents wait until the car is fast and doesn't crash alone.

## Status (2026-11-03)

proto-6.5 is finished. The next step is a full run, proto-7, with a new steering design. **proto-7's contents below are agreed and not yet implemented.** The filter from proto-6.5 is not carried forward.

## proto-6 probes and proto-6.5

**Goal: true generalization.** Held-out completion comes first and pace second. Beating the bot is not the goal in itself, so per-track fine-tuning is out: it would defeat the point. Opponents wait until the car is fast and doesn't crash alone.

**What proto-6 showed when probed in closed loop** (`best/`, 10 eval tracks, 8 episodes, official launch noise, 3-5 seeds, no retraining; the checkpoint was selected on these tracks, so its numbers are optimistic):

| Change at eval time | Completion, train / held-out | Pace vs bot, train / held-out | Steer change per step | Brake applications/s |
|---|---|---|---|---|
| None | 0.80 / 0.42 | -0.6% / +1.0% | 0.115 | 1.40 |
| Low-pass on steer (alpha 0.25) and pedal (alpha 0.5) | 0.70 / 0.43 | -1.75% / -0.7% (3 seeds) | 0.014 | 0.54 |
| Racing-line inputs zeroed | 0.33 / 0.00 | | | |
| Racing-line inputs replaced by the centreline | 0.09 / 0.00 | | | |

- **Jitter is an output problem.** Steer lock is 45 deg and the physics caps the wheels at 120 deg/s (0.107 of command per 40 ms step). 44% of the agent's steering changes exceed that cap, and above 60 m/s its median steering is 0.16 against the bot's 0.0005. The steer-rate penalty is ~0.3% of the progress reward at that level, so it can't fix it.
- **The filter has a cost the completion column hides.** Held-out progress falls from 1.15 to 0.87 laps (lap-2 crashes, e.g. Imola), and Braga goes from 0.2 to 0 completion. Lighter filters are not safer: steer alpha 0.5 does worse than 0.2-0.35.
- **Crashes are a separate problem.** The policy dies at the same corners with or without the filter. On Brno, Lombaro and Corkscrew it enters 15-40 km/h faster than the bot, often back on throttle, then runs wide. +36 km/h moves its pedal by only 0.06, so it is still largely blind to speed. Braga looks different (entry speeds are fine).
- **Ruled out:** exploration noise as the crash cause (steer sigma 0.1 barely lowers completion on the trained policy), wheel lock (0.8% of full-brake steps, so ABS is low priority), and the bot's setup (same gears, brake pressure 18500 vs 18000 kPa). The bot plans with `scale mu` 0.75, so it has margin.
- **The racing line is a leak we depend on.** On held-out tracks it comes from the bot driving that exact track, and proto-6 finishes no held-out laps without it. That is co-adaptation, not proof that a policy needs a line: proto-4 and proto-5 never had one (held-out 0.39 and 0.30 over their last 10 evals, against 0.20 for proto-6 on the same measure).

**Plan:**

1. **proto-6.5: fine-tune proto-6 `best/` with the filter** (`--resume`, 50M steps, ~70 min, LR 7e-5, the schedule's value at 256M). This tests whether the policy can recover the completion the filter costs, without a 7 h run. It keeps the racing line, because the checkpoint can't drive without it. Read it on the last 5 evals, on completion and progress together, against step 0 of the same run. `best/` is a selected peak, so some regression is expected. If the result is unclear, run the same fine-tune without the filter as a control.
2. **Restore a full car state** (engine fork patch, list of values: body, wheels, suspension, engine, gear, clutch, steering, tyres), so that a car can start from a saved agent state without the prestart. Pass/fail test: save a state, replay the same controls, compare the paths.
3. **proto-6.6: mistake learning on the fine-tuned policy, ~70 min.** Spawn from agent states 1-5 s before crashes, perturbed in lateral position, speed and heading, with difficulty adapted to keep survival around 30-70% per zone. Crash points come from training tracks only: using held-out evals would train on those tracks.
4. **Retrain without the racing line, ~7 h:** filter, spawns and the new score together, plus map-based spawns (any position and width, speed from curvature), because an agent from scratch has no snapshots past its first crash. To compare against proto-4 and proto-5 as well as proto-6. Whether this run also moves to a train/validation/test split is still open.

**proto-6.5 result** (MLflow 17831884..., 50M steps, one seed, last 5 evals against proto-6's last 10):

| | proto-6 | proto-6.5 |
|---|---|---|
| Train completion | 0.63 | 0.53 |
| Train progress, laps | 1.31 | 1.16 |
| Train pace vs bot | -0.85% | -2.85% |
| Held-out completion | 0.20 | 0.19 |
| Held-out progress, laps | 0.78 | 0.73 |
| Held-out pace vs bot, survivors only | +0.30% | -0.21% |
| Jarama `steer_delta` (proto-6.5: final eval) | 0.09 | 0.029 |

The filter stays: it is about 3x smoother and faster, with no measurable held-out loss. Fine-tuning did not lift completion, though. Train completion reached 0.97 at 4M (`best/`) and then fell, São Paulo stayed near 0 from 24M on, and `steer_delta` crept up from 0.013 to 0.029. Crashes keep happening at the same corners, and Spa and Brno never finished a lap. No no-filter control was run, so the slide can't be split between the filter and fine-tuning drift.

**Revised verdict on the filter (same day).** The result above read the filter as a win. Looking at what the policy requests shows otherwise (Jarama, clean episode, filter on):

| Checkpoint | Requested steer change | Applied steer change | Requested steer change, lag-1 correlation |
|---|---:|---:|---:|
| Step 0 | 0.059 | 0.014 | -0.43 |
| 10M | 0.076 | 0.015 | -0.56 |
| 30M | 0.130 | 0.022 | -0.65 |
| 50M | 0.169 | 0.029 | -0.65 |

(20M is left out: that episode crashed after 966 steps.)

- **The policy got more erratic behind the filter.** Requested steering chatter tripled and alternates more. The steer penalty only sees the applied value, so raw chatter is free, and the filter hides it. That was a design flaw in the filter.
- **The fine-tune wasn't gentle.** KL stayed at ~0.023 as in proto-6: a lower LR just runs more of the KL budget, because `target_kl` is unchanged.
- The filter's pace and smoothness at step 0 are real, but as a retrofit it isn't stable. Hence a design where the policy's own output is limited and penalized.

## What the Sony papers actually say

- **Nature 2022 (GT Sophy, QR-SAC):** absolute steering in [-1, 1] at 10 Hz, no substantial gain above 10 Hz. Course points on the left edge, centre line and right edge (60 equally spaced 3D points), spaced to span ~6 s of travel at the current speed, egocentric. Hand-tuned scenario ratios were not enough: random sampling "often led to skills being unlearned", and multi-table stratified sampling "to explicitly enforce proportions of each scenario in each training mini-batch" stabilized skill retention. Mistake learning launches training from the state shortly before a lost-control event. Time-trial training: 10 PS4s, ~8 days.
- **arXiv 2504.09021 (GT7, vision-based):** steering is a delta in [-3, +3] degrees per 10 Hz update, linearly interpolated over the 60 Hz ticks, with zero-order hold on the pedal. Penalties: steering change `-|theta_t - theta_t-1|` and a history term that fires when consecutive deltas both exceed a dead-band and reverse sign. Observes steering angle and changes over the last 3 steps. No pedal smoothness term and no stratified replay in this paper. Its 6 s track points are critic-only, because the actor drives from vision.

## Measurements for the design

- **Steering limit from the bot** (cached reference laps, some tracks twice, flying laps; 1.0 command = 45 deg):

| Limit | Per 40 ms step | Bot exceeds it |
|---|---:|---:|
| 30 deg/s (the GT7 rate) | 0.027 | 5.97% (8.88% at 20-40 m/s) |
| 56 deg/s | 0.050 | 1.39% |
| 70 deg/s | 0.0625 | 0.77% |
| 120 deg/s (the sim's own cap) | 0.107 | 0.18% |

  Post-hoc limits on proto-6: 0.05 per step still drives, 0.02 crashes. It's a co-adapted policy, so this is only a hint.
- **Track coverage in PPO.** No replay buffer, so a batch holds what the workers are driving now. 20 workers draw from 35 tracks with replacement and stay ~200k steps: on average 15.4 of 35 tracks are in the batch, a given track is absent 56% of the time, and the set lasts ~390 updates (4M steps, equal to proto-6.5's eval interval). Plausible cause of skills being forgotten between evals; **not verified**, because track assignments aren't logged.

## proto-7 (full run, agreed 2026-11-03)

1. **Steering increments.** Action = change of steering, bounded by a hard limit (start at ~70 deg/s, 0.0625 per step, sized from the bot). 25 Hz is kept and no worker change is needed: the sim's 500 Hz physics already slews moves of this size.
2. **Penalties on the policy's own output.** Steering change, plus a reversal penalty with a dead-band so exploration noise doesn't dominate it. Weights set relative to progress (~0.14 per step at mean speed); the old penalty was ~0.3% of it.
3. **Steering history in observations:** angle and changes over the last 3 steps.
4. **Remove the filter.**
5. **One worker per training track** (workers = tracks, n_steps so samples per update stay ~10k), so every update contains every track, with no respawns. The PPO version of stratified sampling; scenario ratios can later be set per worker at reset.
6. **Speed-scaled road geometry:** left and right edges, egocentric, ~6 s ahead, from the track map. It should expose speed to the policy (a corner appears earlier when faster) and replaces the bot's racing line, removing that leak.
7. **Metrics:** log requested and applied changes separately, and track assignments.

Compared against proto-4 and proto-5 (no racing line) as well as proto-6. If it beats them substantially, run ablations to find out which changes did it.

**After proto-7:** snapshot restore and mistake learning, as a fine-tune from proto-7 with a lower `target_kl`, then a new full run if it helps. Real stratified replay needs an off-policy algorithm (QR-SAC or TQC); deferred.

### Implementation notes

**Split (29 train / 6 validation / 6 test).** Validation = the current held-out tracks (Imula, ardennen-spa, brno, corkscrew, lombaro, street-1), which select checkpoints. Test = alicante, atlanta, auriac, brondehach, nordschleife, spring: drawn before any proto-7 result, with `random.Random(20261103).sample(sorted(proto-6 train tracks minus jarama, braga, sao-paulo, petit), 6)`. Don't redraw, even though it picked the two longest tracks. Test tracks never select anything: evaluate them only on the final `best/`, with a separate script, not in the training evals. They were trained on in proto-4/5/6, so no test comparison with those runs exists.

**Steering.** `steer_t = clip(steer_t-1 + action x limit, -1, 1)`, limit 0.0625 per 40 ms step, 0 at episode start. Pedal stays absolute, gearbox `points` unchanged. Remove the action filter and its flags. Keep `--resume`.

**Penalties.** On the applied change (now the policy's own output): `-w1 x |change|`, and `-w2` when consecutive changes reverse sign and both exceed a dead-band. With the 0.1 std floor, noise is ~0.006 per step, so start the dead-band around 0.015. Set the weights from a short smoke run against `reward/progress`, logged as separate components, and check they neither dwarf the terminal penalty nor stay negligible.

**Observations.** Steering angle and the changes over the last 3 steps. Edges: `--edge-obs`, left and right borders in the car's frame, ~30 points per edge spanning 6 s at current speed with a floor distance, from the borders already cached in `reference/` (Python only, 2D, as `track.py`). Remove `--line-obs` and `--line-shaping` from the run, not necessarily the code (`line_dev` stays as a metric). Keep rays and curvature lookahead. Check 30 points resolves Spa's Bus Stop, and measure throughput before raising the 256x256 net.

**Workers.** One per training track: 29 workers, `n_steps` 353 (~10.2k samples per update). Worker i keeps its track; today a respawn draws a random one in `vec_env.step`, so recycling must keep the same track. Check memory and throughput (eval adds 80 workers on a 16 GB machine).

**Metrics.** Log requested and applied changes separately. Check the score still reads sensibly, with validation tracks counted in it.

**Order.** Tests, smoke run, `scripts/bench_env.py`, then launch 300M under `caffeinate` with the same flags as proto-6 otherwise (seed 1, `--car-random 0.1 --car-obs --min-std 0.1 --max-std 0.5`, `--eval-every 8000000`).

Launch long runs with `caffeinate -i -w <pid>`, on AC with the lid open: the Mac slept through ~90 min of proto-6.5.
