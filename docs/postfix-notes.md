After fixing the env spawns and eval categorization, the new runs are not comparable anymore.

I have therefore decided to use the replay utility to observe them. For now i have only watched training tracks, not test.

I will note everything i notice here:
all tracks: 
- it seemingly doesn't know how to flat out in long straights - never shifts up, redlines at 3rd gear and half throttle. doesn't seem to know it can go faster
- it scrapes walls a loooot
- bad trajectories all around: sticks to the center, rarely takes the apex

two broader possibilities: 
1. not enough reward to make it go fast instead of just progressing through the track (current just pays per metre of progress so there is no direct incentive to go fast and explore faster trajs)
2. not enough penalty on damage and/or offroad

the first problem in more detail:
the agent will not accelerate faster than a set speed. It handles correctly at high speeds when launched from the bot laps, it just won't accelerate past 3rd gear (160kmh) or 4th gear in automatic gear mode (190). I first thought it was a problem with the gear but actually it's the bot the refuses to accelerate, looks like a local optimum.

Tried:
- is the critic valuing faster states less ? : no, the critic does value faster states to be higher. which makes sense, it nets more reward
- is GAE lambda a problem ? : current lambda (0.95) may be too low and preventing ppo from finding that accelerating makes it faster. Shouldn't be the case since the critic worrks fine. Tried 0.98- no meaningful difference, behavior wise it uses 4th gear all the time and doesn't use the gearbox
- is normalizing gear obs a problem ? : every observation feature is currently normalized (default cleanrl behavior so i left it since our obs vary in units). but that means the gear numbers also are, and more importantly that rare gears will be weighted much higher. PPO may avoid that so I did a run where the gear obs is fixed (no norm)- same behavior, worse on all metrics
- is the low pedal std the problem ? : each run's std collapses as the policies become more confident. however they may become confident too fast and miss how to accelerate higher. I have tried to put a floor on it so that it keeps exploring
    - 0.1: no change in metrics, barely useful (std was 0.1 at the end, testing 0.2 next)
    - 0.2: 
- colored noise is correlated over time and apparently works well for ppo according to a study i read so i'm going to test beta=0.5- no change in metrics - however it drives the most like a driver on replays (not like a bot - weird to explain, it seems more deliberate) 


==== update
so this was caused by a "bug". the eval uses the mode to evaluate (whatever the policy says most likely, the top 1 choice). However, the policy predicts something like [0.6, 0.3] which would upshift after a fewsteps. However in eval we only ever pick the 0.6, so we never upshift.  

however even when evaluating using sampling etc it avoids getting into 5th gear. the agent avoids higher speeds, like the original hypothesis, confirmed. still an open issue
for now i put the gearbox in auto mode
---

for the other 2 issues:
- colored noise did improve the driving a lot, especially when activated both on steering and pedals. The run has better pace than any so far, with similar safety metrics. The only problem is that it swerves on straights. So i'm trying things:
    - noise scale by min(1, 30/v) so as speed increases colored noise decreases: 
- after reading gt sophy's paper they had the same issue and fixed it by putting a penalty on wall contacts scaled by the speed squared. We do already stop paying rewards when offroad, and penalize damage, however from the bot's behavior it's clear it needs more pressure in this area:
- more pressure on offroad as well: -0.01 per meter off track (-5):
    - wallcost of v^2 + pressure on offroad: halved the wall contacts; boosted clean laps (+0.1) and pace remains at the same level. free win

---

Noise floor for 3 seeds against 3 (all 38 tracks): lap pace needs about +0.05, scenario pace about +0.03, nominal clean about +0.06, nominal survival about +0.08. Lap clean is descriptive only. The telemetry command will be added here once it's built, next.

Variants (each is baseline plus one change):

┌───────────────────┬───────────────────────────────────────────┬────────────────────────────────────────┐
│      VARIANT      │                  change                   │                 status                 │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ baseline          │ train.py defaults                         │ seeds 1-3 done                         │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ lambda98          │ GAE λ 0.98                                │ s1 done (no pace gain, survival −0.06) │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ colored           │ colored pedal noise, β 0.5                │ —                                      │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ pedalfloor        │ pedal std floor 0.1                       │ —                                      │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ gearfixed         │ gear inputs skip the normalization        │ —                                      │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ wallcost          │ wall cost 1 per step at 40 m/s, × (v/40)² │ —                                      │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ colored-pink      │ colored pedal noise, β 1 (pink)           │ follow-up                              │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ colored-both      │ colored noise β 0.5 on steering and pedal │ follow-up                              │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ pedalfloor02      │ pedal std floor 0.2                       │ follow-up                              │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ wallcost05        │ wall cost 0.5                             │ follow-up                              │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ wallcost2         │ wall cost 2                               │ follow-up                              │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ fix-height-centre │ elevation input (centre line)             │ s1-2 done, s3 missing                  │
├───────────────────┼───────────────────────────────────────────┼────────────────────────────────────────┤
│ fix-autoshift     │ automatic gearbox (diagnostic only)       │ s1 done                                │
└───────────────────┴───────────────────────────────────────────┴────────────────────────────────────────┘

Ready-made commands:
# the four seed-1 runs, most relevant first (~1h40)
caffeinate -i uv run python scripts/sweep.py --only colored pedalfloor gearfixed --seeds 1 --steps 10000000

# elevation, the missing seed
caffeinate -i uv run python scripts/sweep.py --only fix-height-centre --seeds 3 --steps 10000000

# overnight template: seeds 2-3 of whichever arms you keep, plus elevation s3
caffeinate -i uv run python scripts/sweep.py --only fix-height-centre <ARM> <ARM> ... --seeds 2 3 --steps 10000000

Reading results:
# per variant over seeds, z against baseline, 38 tracks pooled; --curve adds the 2M..10M evals
uv run python scripts/analysis/sweep_report.py --steps 10000000 --split all --only baseline <VARIANTS...> --curve
# same, validation or train only
uv run python scripts/analysis/sweep_report.py --steps 10000000 --split validation --only baseline <VARIANTS...>

# replay a run in the game with live telemetry against the bot
uv run python scripts/watch.py sw10m-<VARIANT> <track> [--laps 2] [--no-game] [--seed