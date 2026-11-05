# proto-9 plan

Written 2026-11-04 while proto-8 ran, updated 2026-11-05 with its result. **proto-9 is built and smoke-tested, and not yet run.** Sections 1 and 2 and the off-track rules are in the code; section 3's tests are not done. The evidence behind section 2 is in [research-notes-proto7.md](research-notes-proto7.md), "Before proto-8: pedals and the critic".

## proto-8 result

`proto-8` crashed at 130.9M and `proto-8-cont` resumed it, stopped by hand at 372M. Report: `reports/proto-8-cont-b60db21f/`. Completion over all 35 tracks, 2 episodes per checkpoint, one seed per run:

| Steps | proto-7, 29 train | proto-8, 29 train | proto-7, 6 validation | proto-8, 6 validation |
|---|---|---|---|---|
| 250-300M | 0.67 | 0.68 | 0.33 | 0.35 |
| 300-340M | 0.76 (300M only) | 0.70 | 0.17 (300M only) | 0.43 |
| 340-372M | no data | 0.75 | no data | 0.28 |

`best/`, 8 episodes per track: 0.65 -> 0.73 on train, 0.48 -> 0.48 on validation.

- **Launches didn't clearly change the result.** The fine-tune's validation 0.53 didn't carry over to a full run. They did no visible harm, so they stay in the baseline.
- **The live dashboard misled.** Its 4 train tracks read 1.00 while all 29 averaged about 0.72.
- **Flipping didn't fall with the learning rate:** 7-9 of 35 tracks per checkpoint (about 6M apart) throughout, including at 3e-5.
- **Lethal corners remain, on trained tracks too.** At `best/`, Kurpfalzring, Ole-road-1, Salty, Hidden-valley, Tuna, Torino, Allondaz, Imula and Corkscrew finish 0 of 8, nearly always at one spot per track.
- **Most of those crashes look like late braking.** On 10 of the 12 crash corners checked, the car was 20%+ above the bot's speed 100 m out (Corkscrew 186 vs 103 km/h, Kurpfalzring 167 vs 96, Imula 211 vs 154 with no braking). Not measured: how often it passes corners at similar overspeed.
- **Same checkpoint, different outcome.** Re-recording `best/` alone flipped the result on Lombaro, Street-1 and Summer-Olsen against the report's first episode. The cause isn't traced.

### Correction: many of those corners aren't lethal

`best/` on all 35 tracks, 4 episodes each, with the off-track and backwards limits raised from 1 s to 20 s and no retraining. The policy has never trained in these situations and can't reverse, so these are lower bounds. Of 140 episodes, 41 would have failed under the 1 s rules:

| What happened next | Episodes | Tracks |
|---|---|---|
| Came back and finished 2 laps | 18 | Tuna, Kurpfalzring, Torino (4 each), Allondaz (3), Summer-Olsen (2), Street-1 (1) |
| Came back, then failed later | 6 | Imula (4), Hidden-valley, Ole-road-1 (1 each) |
| Didn't come back, heavy damage (468-1056) | 9 | Corkscrew, Salty (4 each), Lombaro (1) |
| Didn't come back, little damage | 8 | Hidden-valley, Ole-road-1 (3 each), Street-1 (2) |

Completion, strict -> 20 s limits: train 0.75 -> 0.90, validation 0.50 -> 0.54.

In one clean episode per track, the excursions on Tuna, Kurpfalzring, Torino and Summer-Olsen lasted 1.4-5.4 s, 3-5 m from the road, at 60-95 km/h on entry. Path length matched track progress within about 7%, so they aren't shortcuts, and those laps finished 6-8% quicker than twice the bot's best lap. Imula's first excursion gained 12% more progress than its path, the only hint of cutting.

So the 1 s rule itself accounts for a large share of the "lethal" train corners. The late-braking finding still fits Corkscrew, Salty and Imula, but not these excursions.

### Imula's first corner

The chicane is in the observations before the bot's braking point (s ~ 440 m): at s = 424 m both the edge points and the lookahead windows show it. The mean pedal output falls from +3.1 at 250 m out to +1.1 at 125 m, but it stays positive, so the car stays at full throttle. The critic's value barely moves until the car is leaving the road. So this isn't a visibility problem: the policy reacts and doesn't react enough. It never trains on Imula, so this can't separate lambda from generalisation.

## 1. Road height in the observations

**Idea.** The policy can't see hills. Laguna Seca's Corkscrew drops hard and needs heavy braking, and Corkscrew is a track the policy keeps failing. Feed it the height of the road ahead, spaced by speed like the road edges.

**What's missing today.** No input carries height or slope: the only vertical signal is `speed_z`, the car's vertical speed right now. The road edges are 2D, and so is the track map.

**Why it could matter.** A descent lengthens braking distances, and a crest takes load off the tyres just as the car turns. Neither can be anticipated from a flat map.

**What the exact heights show (2026-11-05):** best/ leaves Corkscrew at 2459 m, just past the top of the hill (49 m at 2410 m), as the road starts dropping at up to 12.6%. Training has steeper descents and sharper crests (Inuksuk -20.6%, Petit -16.2%, Allondaz -16.0%, Torino -15.8%), so it can learn what hills mean. Step 3 below wasn't done, since height is going into the run regardless.

**Evidence so far, not all of it in favour:**

- proto-7's check of elevation against deaths found no signal, but it was crude: mesh heights, with a median match error of 6 m.
- proto-5 completed Corkscrew at 0.75-0.88 without height, more slowly (see `gym-wrapper.md`). So height isn't strictly required there, though that doesn't mean it wouldn't help.
- Corkscrew is a validation track, so it is never trained on. The policy can only learn what hills mean from the training tracks, so they need steep descents and crests too. That hasn't been measured.

**Design:**

- **Input:** centreline road height relative to the road at the car, at the same points as the edges except the one abreast of the car, which always reads 0 (`--height-points 29`, 6 s of travel at current speed, 60 m floor). The observation grows from 170 to 199.
- **Source:** the engine's `RtTrackHeightL`, from `rlpilot-worker -z <file>` (`writeHeights`, `native/worker/main.cpp`). Python then looks heights up, like the edges, so no protocol change is needed at each step.
- **Scale:** metres / 10. At 60 m/s the worst value on the 35 tracks is 4.97, like the edges.
- **Cache:** `reference/<race>-<key>_height.csv`, beside the track map, so no bot laps had to run again.
- **Flag:** off by default. A checkpoint can't resume with a different observation size, so this needs a full run.

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

Before the next run, also **evaluate all 35 tracks live**, since the 10-track dashboard misled on this run. Done: `--split proto7` now evaluates all 35, 2 episodes each, and `best/` is selected on the strict score, averaged per group so the 29 training tracks don't outvote the 6 validation tracks.

Then, agreed on 2026-11-05: **one full run with three changes.** No fine-tunes, since the launch fine-tune's gain didn't carry over.

| Change | What |
|---|---|
| Road height | Section 1 |
| `--gae-lambda 0.98` | Section 2 |
| Off-track rules | No progress reward with all 4 wheels off, off-track limit up from 1 s to about 5 s, terminal penalty raised to about 50 |

Reverse stays out of this run. It needs the gearbox action and changes to the stuck, backwards and progress rules, and it's most relevant to the 9 heavy-impact episodes.

Each run takes about 10 h, and 300M is enough: 300-372M added little. Changing three things loses attribution, which is accepted here to save runs. A later run can remove one if the result needs explaining.

**How to read proto-9:** on all 35 tracks, in the report, under both rules: the old strict 1 s rule, which keeps comparison with proto-7 and proto-8 fair, and the new limits. Also the speed against the bot 100 m before Corkscrew, Salty and Imula's first corner, and whether those corners clear. Not only mean completion, which swings too much.

## Parked

Raised earlier, and not part of proto-9 unless results call for them:

- **Launch at racing speed upstream of crash spots,** using `tracks/failures.csv`. Launches already cover these corners and they stay lethal, but launches top out at 105 km/h, while the crashes come from arriving at 160-210. So match the speed too, rather than only launching more often.
- **Fix `--seed`** so it reaches the car and launch draws. Needed before any multi-seed comparison.
- **Test split.** Still unspent. Run it once, on a final candidate.
