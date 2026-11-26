After fixing the env spawns and eval categorization, the new runs are not comparable anymore.

I have therefore decided to use the replay utility to observe them. For now i have only watched training tracks, not test.

I will note everything i notice here:
all tracks: 
- it seemingly doesn't know how to flat out in long straights - never shifts up, redlines at 3rd gear and half throttle. doesn't seem to know it can go faster (telemetry: 92-98% of the time in 3rd, top speed 166 km/h = 3rd's limiter, throttle ~0.44 up there. almost never brakes hard either: the limiter does the braking)
- it scrapes walls a loooot
- bad trajectories all around: sticks to the center, rarely takes the apex

two broader possibilities: 
1. not enough reward to make it go fast instead of just progressing through the track (current just pays per metre of progress. note: that does pay for speed indirectly, more metres per second = more reward per second, and gamma makes reward sooner worth more. so the incentive is there, the agent just doesn't find it -> exploration problem, not reward problem)
2. not enough penalty on damage and/or offroad (a median wall contact cost ~0.4 m of progress. basically free)

the first problem in more detail:
the agent will not accelerate faster than a set speed. It handles correctly at high speeds when launched from the bot laps (survives 86-89% of launches at 60+ m/s), it just won't accelerate past 3rd gear (160kmh) or 4th gear in automatic gear mode (190, by lifting: throttle ~0.33 at the top). I first thought it was a problem with the gear but actually it's the agent that refuses to accelerate, looks like a local optimum.

Tried:
- is the critic valuing faster states less ? : no, the critic does value faster states to be higher. which makes sense, it nets more reward (93% of steps, corr 0.85 with the speed gain. and forcing full throttle + upshifts on straights took pace 0.933 -> 0.968: going faster is right there, the actor just never tries it)
- is GAE lambda a problem ? : current lambda (0.95) may be too low and preventing ppo from finding that accelerating makes it faster. Shouldn't be the case since the critic works fine. Tried 0.98- no meaningful difference (same pace, faster straights but slower corners and start, 2x the walls), behavior wise it uses 4th gear all the time and doesn't use the gearbox
- is normalizing gear obs a problem ? : every observation feature is currently normalized (default cleanrl behavior so i left it since our obs vary in units). but that means the gear numbers also are, and more importantly that rare gears will be weighted much higher (5th gear read as +7 sd when engaged). PPO may avoid that so I did a run where the gear obs is fixed (no norm)- same behavior, worse on all metrics (survival -0.12, dropped)
- is the low pedal std the problem ? : each run's std collapses as the policies become more confident. however they may become confident too fast and miss how to accelerate higher. I have tried to put a floor on it so that it keeps exploring
    - 0.1: no change in metrics, barely useful (std was 0.1 at the end, testing 0.2 next)
    - 0.2: safest arm on seed 1 (crashed starts recovered +0.15) but over 3 seeds: pace a bit lower (lap pace 0.91 vs baseline 0.96), still lifts on straights like baseline. so more noise per step alone doesn't make it go fast
- colored noise is correlated over time and apparently works well for ppo according to a study i read (Hollenstein et al. 2024) so i'm going to test beta=0.5- no change in metrics - however it drives the most like a driver on replays (not like a bot - weird to explain, it seems more deliberate)
    - the "deliberate" thing is real and measurable: every colored run (9/9) is flat out on 85-98% of straights and brakes hard like the bot (decel p99 ~14-15 m/s², bot 15). every non-colored run: 18-85%, gentle braking
    - and floor 0.2 has the SAME per-step std as colored (0.21 vs 0.19) but drives like baseline. so it's the correlation over time that matters, not how much noise. white noise at 25 Hz averages out within a second, it never really "tries" a full straight flat out
    - why no change in metrics then: what it gains on straights it loses in corners (for now)
    - beta 1 (pink) = too much, clearly worse. colored on steering + pedal = fastest corners at seed 1 but weaves on straights


==== update
so this was caused by a "bug". the eval uses the mode to evaluate (whatever the policy says most likely, the top 1 choice). However, the policy predicts something like [0.6, 0.3] which would upshift after a few steps. However in eval we only ever pick the 0.6, so we never upshift. (actual numbers at the limiter in 3rd: hold 0.67, up 0.29 per step -> sampled it upshifts within ~3 steps, argmax never. time on the limiter 0.63 -> ~0.25 with a "hazard" shift mode in eval)

however even when evaluating using sampling etc it avoids getting into 5th gear. the agent avoids higher speeds, like the original hypothesis, confirmed. still an open issue (and it's per seed: P(upshift) in 4th at the limiter = 0.00 on one seed, 0.91 on another. colored drops back from 5th to 4th almost every time -> the juggling i saw on spa)
- forcing upshifts at the right rpm made it faster but halved clean laps: it arrives at corners faster than its braking is calibrated for. so the gear habits are a speed governor, not the cause. speed and braking have to be learned together
for now i put the gearbox in auto mode
---

for the other 2 issues:
- after reading the gran turismo papers (Fuchs et al. 2021, the one before sophy) they had the same issue and fixed it by putting a penalty on wall contacts scaled by the speed squared (without it their car grinded along walls, a fixed cost made it freeze). We do already stop paying rewards when offroad, and penalize damage, however from the agent's behavior it's clear it needs more pressure in this area:
- more pressure on offroad as well: with 4 wheels off, -0.01 per meter driven x how far past the edge (capped at 10 m). per meter driven and not per second, because per second the early random car was better off standing still (returns went to -30/-50). 0.02 tied with standing still, 0.01 is fine
    - wallcost of v^2 + pressure on offroad: halved the wall contacts (1.5 per 10 km vs 2.1-4.0); boosted clean laps (+0.1, best of any run: 0.82) and pace remains at the same level. free win (caveat: one seed, and it came with auto + colored + floor in the same run)
state: the agent still takes the middle of the road instead of a proper trajectory. it matches the bot in pace but doesn't go faster than that. the self speed limit had been fixed mostly by the noise
- where the time goes, per corner: it enters ~4% faster than the bot, then loses it from the apex out (0.97-0.98 of the bot's speed after) and carries that down the whole straight. enters from the middle (bot from the outside). = beginner driver: hot entry, bad exit
- forcing the bot's line on it made it SLOWER (+0.2 s per corner): it brakes for a line it has never driven. so line + braking have to be learned together too, can't bolt the line on
- width bonus (pay a bit more for progress near the edge, faded out by 15M): hugs the edges but still comes back to the middle on every exit, clean laps 0.82 -> 0.67, pace unchanged. pays any edge, not the line. dropped
- also ~3% top speed lost to the auto box shifting at 92% of redline

==== the v1 yardstick (25 nov)
ready = beat v1. same metric for both: lap 2 time vs the bot's best lap, laps that finished
- v1 (proto-8-cont): 6.7% faster than the bot on train tracks. but only ~half the tracks finished
- v2 pressure 30M: 0.7% slower on train, 3.1% slower on validation. but 30/32 train tracks finish
- biggest difference found: exploration. v1 had a fixed std clamped to [0.1, 0.5] on every action. v2's learned std sits at 0.03 for steering at every speed (the minimum allowed) -> v1 explored ~3x more steering at speed. that's the line problem: you can't find an outside entry without trying it
- (and v1 ran 300M+ steps, v2 runs stop at 30M)

next trials:
    - base: auto + pressure + std floor on pedal and colored noise
    - base + std floor on steering (0.1, like v1): at spa it just stops ? tf
        - why: completion 0.96 -> 0.81, 12/38 laps end stopped. at driving speeds the std sits right on the floors (it wants less). at a standstill it shoots to the max (1.6): random actions beat a mean that was never trained there, so the std grows, so the mean matters even less, loop. the eval drives the mean, which just brakes forever
        - v1 didn't have this because its 0.1 was there from step 0 at every speed, standstill included
    - base + colored noise both + yaw-rate cost on long straights (anti-weave): 0.5 killed learning (0 at 2M). 0.25 learns (completion 0.95, pace 0.97) and deletes the weaving, but the agent drives weird: always 2 wheels off, hugging the track limit. 2.9% slower than the bot on train, 6.3% on validation
    - base + std floor both + colored noise both + noise x0.25 on long straights: slow start, ends at pace 0.945, 7.7% slower than the bot on train. no visible improvements and behavior is similar
    - base + std floor both + colored noise both + gate the noise on straights to x0.05 (same noise in corners and their approaches, almost none on straights, which should eliminate the swerving): better, still improving at 10mil step. i think this is the best bet i have currently. the colored noise seems to help a lot; still a bit of weaving but much less than before.
        - numbers: pace 1.014 (best v2 run), lap 2 vs bot +0.16% train, +0.73% validation (best validation so far, pressure 30M was +3.1%). completion still climbing at 10M (0.42 -> 0.85), pressure was flat by 6M
        - weak spot: 8/38 laps end stopped, and they're crash -> freeze (same standstill std thing as above)
        - drives in-in-out instead of out-in-out. sensible beginner
        - "long straight" = no bend tighter than 800 m radius from 30 m behind to 150 m ahead, ~16% of a track. so the approach to a corner keeps all its noise
    - previous + noise ceiling at 0.5: the car would sometimes completely stop after a crash. at std 1.6, which wouldnt let it learn anything at a standstill (+-1.6 on the steering is bang bang, actions clip at ±1): 
        - this is what i ended up doing in v1 ([0.1, 0.5]) just because std 1 felt too high. now there's an actual reason
        - result: clear win, best v2 run. at 10M vs gate05: completion 0.85 -> 0.93, clean laps 0.49 -> 0.68, stopped laps 8 -> 5, pace 1.016
        - lap 2 vs bot: 1.1% FASTER on train (first v2 run under the bot), level on validation (+0.03%)
        - learns way faster too: completion 0.85 at 2M, gate05 needed the full 10M to get there
        - still a few crash -> freeze left (salty: hits at 112 km/h then sits on the brake, std at the 0.5 cap now). better, not solved
        - side note: grad norm 5-20 vs the 0.5 clip -> clipped 100% of the time. adam rescales the step anyway so it's not throttling learning, it just makes every minibatch count the same
    - is the cap the whole win? control = pressure + cap 0.5, no steering noise (10M): safe (completion 0.93, clean 0.78) but slow (pace 0.99, lap 2 +1.6% train / +4.2% validation). so the two halves do different jobs: the cap = safety (no freeze loop), colored steering noise + gate = pace
    - gate05 + cap 0.5 at 30M: keeps improving past 10M, slowly (pace 1.016 -> 1.036, completion 0.93 -> 0.95, clean 0.68 -> 0.79)
        - lap 2 vs bot: 3.6% FASTER on train (31/32 tracks finish, only salty doesn't), 1.6% faster on validation (4/6)
        - v1 was 6.7% faster but on 22 tracks. not quite there on pure pace yet, but way more consistent. still climbing at 30M, v1 needed 300M+
        - generalization: no gap. scenario suite validation >= train on every tier (nominal ok 0.89 train / 0.94 val), lap pace gap ~2% (laps), validation laps flip track to track though (karwada fails now, street-1 finishes now, torino never)
        - careful: the first 8M are bit-identical to the 10M run (same seed = deterministic), so this is still one seed
    - crashed tier (the weak one, ok ~0.4): it's all backwards starts. facing forward it recovers 99%, facing backwards (150-180 deg) 3%, and 94% of those never move at all. learnt it at 14M (45%) then lost it by 30M. backwards starts = ~4% of training
        - watch.py --scenario to see it (14M turns around, 30M sits there)
        - tried: stuck cost (-0.5/s after 2 s without progress, 10M): backwards starts 0% -> 19-39%, safer everywhere (clean laps 0.68 -> 0.82, all 6 validation laps finish for the first time) but ~7% slower (lap 2 +6% vs bot on train). makes sense in hindsight: almost every crash ends in some stuck time, so it's a crash cost too -> more careful driver. same lesson as v1's terminal 50
        - better idea: charge being idle, not not-progressing (-0.5/s under 0.5 m/s after 1 s). reversing to turn around doesn't make progress, but it isn't idle, so it doesn't get charged
            - 10M: completion 1.000, 0 stopped laps, all 32 train + 6 validation tracks finish (torino for the first time). pace 1.017, lap 2 about the same as cap05 (-0.6% vs -1.1% on matched tracks)
            - backwards starts still 0%. so idle kills the crash -> freeze in laps, but turning around needs practice (more backwards starts), not just a reason to move
    - overnight: gate05 + cap 0.5 + idle 0.5, 100M (anneal over 100M)
        - still climbing at the end: pace 1.016 (10M) -> 1.032 (90M) -> 1.044 (100M)
        - lap 2 vs bot at 100M: 4.3% FASTER on train (28/32 finish), 2.7% faster on validation (5/6, torino still the odd one). best v2 on both, closing on v1's 6.7% (which finished ~22 tracks)
        - generalization holds: validation lap pace 1.033 vs train 1.045, scenario suite validation ~= train on every tier
        - unseen tracks, 2 laps from the grid: alicante 5.9% faster than the bot, karwada 4.8%, dijon 3.2%
        - trade: a bit less robust than cap05 30M on the scenario suite (nominal ok 0.85 vs 0.89 train), crashed tier flat (~0.39), backwards starts still 0

==== is ppo using the data well? (could we train faster with more epochs / lr?)
- the formal check: approx_kl and clipfrac. both low (kl < 0.01, clipfrac < 0.1) = updates barely move the policy, more epochs/lr would help. both high = already pushing too hard
- ours: colored runs kl 0.05-0.18 (gate05 spiked to 0.83), clipfrac 0.25-0.35. baseline (white noise) 0.01-0.02 / 0.14-0.19. critic fine (explained variance 0.98)
- so no: it's already over-stepping, more epochs would make it unstable. slow learning = exploration + not enough samples, not wasted data. the lever is throughput (desktop ~2.4x the mac)
- grad norm wasn't logged, now it is (2_ppo/grad_norm, grad_clipped)
