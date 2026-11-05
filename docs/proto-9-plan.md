# proto-9 plan

Draft, written 2026-11-04 while proto-8 runs. Nothing here is built yet, and proto-8's result can change the order. The evidence behind section 2 is in [research-notes-proto7.md](research-notes-proto7.md), "Before proto-8: pedals and the critic".

**Wait for proto-8 to finish before changing code.** Workers respawn from the installed binary during the run, and `report.py` imports `python/rlpilot` when it ends, so a rebuild or an edit there could affect the run. Docs are safe.

## 1. Road height in the observations

**Idea.** The policy can't see hills. Laguna Seca's Corkscrew drops hard and needs heavy braking, and Corkscrew is a track the policy keeps failing. Feed it the height of the road ahead, spaced by speed like the road edges.

**What's missing today.** No input carries height or slope: the only vertical signal is `speed_z`, the car's vertical speed right now. The road edges are 2D, and so is the track map.

**Why it could matter.** A descent lengthens braking distances, and a crest takes load off the tyres just as the car turns. Neither can be anticipated from a flat map.

**Evidence so far, not all of it in favour:**

- proto-7's check of elevation against deaths found no signal, but it was crude: mesh heights, with a median match error of 6 m.
- proto-5 completed Corkscrew at 0.75-0.88 without height, more slowly (see `gym-wrapper.md`). So height isn't strictly required there, though that doesn't mean it wouldn't help.
- Corkscrew is a validation track, so it is never trained on. The policy can only learn what hills mean from the training tracks, so they need steep descents and crests too. That hasn't been measured.

**Design:**

- **Input:** road height relative to the car's position on the road, at the same points as the edges (30 points, 6 s of travel at current speed, 60 m floor). This reuses the edge table's indexing, and the observation grows from 170 to 200.
- **Source:** the engine's `RtTrackHeightL`, in the worker's track map (`writeTrack`, `native/worker/main.cpp`). Python then looks heights up, like the edges, so no protocol change is needed at each step.
- **Scale:** measure it first over the 35 tracks and on the bot's laps, as for the edges. Don't guess it.
- **Cache:** reference files are keyed by the race file only, so a changed map format won't regenerate them. Write heights to a separate file, so that 41 tracks of bot laps don't need rerunning.
- **Flag:** off by default, e.g. `--height-points 30`. A checkpoint can't resume with a different observation size, so this needs a full run.

**Steps:**

1. Dump exact heights for all 41 tracks and plot Corkscrew's drop to confirm the data is right.
2. Compare the steepest descents and crests on train and validation tracks. If training has nothing like the Corkscrew, say so before spending a run.
3. Redo the elevation check with exact heights: controlling for curvature, do deaths and overspeed cluster on descents and crests? A weak result doesn't cancel the run, but it lowers expectations.
4. Tests: known answers on a synthetic slope, signs, lap wrap, first observation and `final_obs`, a figure-eight crossing, and byte-identical behaviour with the flag off.
5. Render it with `scripts/plot_obs.py` at Corkscrew at low and high speed, and smoke-run it.

**How to read the run.** Fix the criterion before seeing results, once proto-8's report gives a baseline. Draft: compare survival at the steepest descent and crest corners against all other corners, proto-9 vs proto-8. Height should help the first group more. A general rise could just be run-to-run noise, since one track flip moves validation by 0.167.

**Option:** left and right heights separately would also give banking, for 30 more inputs. Start with the centre height unless step 3 says banking matters.

## 2. GAE lambda 0.98

**Idea.** `--gae-lambda` exists now, and every run so far used 0.95. With gamma 0.995 at 25 Hz, advantages look about 0.73 s ahead at 0.95 and 1.6 s at 0.98, while braking decisions come 2-4 s before a corner.

**What the probe showed:**

- The critic isn't blind: on training tracks it tells crash-bound passes from surviving ones about 2 s out, and is borderline at 3 s. Validation is too thin to call.
- 0.98 carries about 1.5-2.3x the crash signal at 3-4 s before a crash, even after its extra variance.

So 0.98 is worth trying, but 0.95 isn't shown to be the limit. Those numbers come from episodes selected for crashing, with critics trained at 0.95.

**Watch:** `dashboard/ppo_approx_kl` (proto-7 sat near the 0.03 cutoff for much of the run) and whether corners still flip between checkpoints.

## 3. Why results swing between evals

**Observation.** At proto-8's 80M, 88M and 96M, validation went 0.48 -> 0.19 -> 0.48. Swings like these are larger than the gap between runs, so they limit what any comparison can show. Two suspects, plus a third that fits the evidence and would change the fix.

**A. Step size too high (learning rate, `target_kl`).** Each update moves the policy enough to forget tracks.

- For: KL sat near the 0.03 cutoff from 100M until the learning rate fell below ~5e-5 at ~260M, and the fine-tune's best numbers came at 3e-5 and `target_kl` 0.01.
- Against: proto-7's last 20M and the control fine-tune (`target_kl` 0.01, read at 21M of 50M) both flipped ~6 of 35 tracks per 4M. So that evidence doesn't show step size causes flipping.
- Test, no new run: proto-8's learning rate decays over the run, so count track flips per 4M against learning rate and KL from its saved checkpoints. Then fine-tune from the same checkpoint at two step sizes and compare flip rates, keeping the run alive past 21.9M this time.

**B. Too few eval samples.** More episodes per track probably won't help: results are almost all-or-nothing (93% of per-track completions were 0 or 1 in proto-7's smoke run), so 8 episodes act like one. Re-running `best/` gives identical results, and raising launch noise didn't change which tracks passed, though that was on a young checkpoint. The likelier gap is **tracks and checkpoints**: 10 tracks, one eval every 8M, and `best/` chosen on those.

- Test, no new run: on a few checkpoints, split the variance into same checkpoint with different perturbations (launch noise, grip +-1%) and adjacent checkpoints.
- If the same checkpoint is stable and adjacent ones differ, the policy really is changing, and more eval only helps read that and pick `best/`.
- Likely change anyway: evaluate all 35 tracks live with 2 episodes each (70 workers, against 80 today), with a score that doesn't let the 29 training tracks dominate `best/`.

**C. No margin.** The policy may drive so close to the limit that tiny changes flip whole corners, whatever the step size. proto-7's train completion falls from 0.69 to 0.33 at grip 0.9, and in October's eval rework, action noise of 3e-7 flipped crash vs lap. If so, a lower step size hides the flips without making the driving safer. The test in B, with its small grip changes, also measures this. If this is the cause, the fix is in training (risk in the reward, or more car and launch variation), not the optimizer.

## 4. Order

**Do section 3's tests first:** they need no full run and they set how the later runs get read.

Then, recommended: **one change per run, height first.**

| Run | Change | Compared with |
|---|---|---|
| proto-9 | proto-8 + height | proto-8 |
| proto-9b | proto-9 + `--gae-lambda 0.98`, or proto-8 + it if height doesn't help | The run it starts from |

Each run takes about 10 h. Changing both at once saves a run, but wouldn't show which one worked. Height goes first because it adds information the policy can't get any other way, while lambda only changes how it learns from what it already sees.

If proto-8 already clears the lethal corners, rethink this plan before building anything.

## Parked

Raised earlier, and not part of proto-9 unless proto-8 calls for them:

- **Launch more often upstream of crash spots,** using `tracks/failures.csv`. It reuses the launch code, and it's the next step if lethal corners stay.
- **Fix `--seed`** so it reaches the car and launch draws. Needed before any multi-seed comparison.
- **Test split.** Still unspent. Run it once, on a final candidate.
