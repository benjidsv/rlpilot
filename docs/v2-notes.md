# v2 notes

Started 2026-11-06. v2 drops the proto-N naming: the changes are large enough that its metrics won't be comparable with proto-1..9.

## Goal

Generalize to unseen tracks. To do that, the car has to be **safe before it is fast**.

## Observations

Rule of thumb: the agent gets what a human driver gets. Track geometry, the previous action and the steering history all pass that test.

- **Add acceleration in the car's frame:** longitudinal and lateral (`ACCEL_X`, `ACCEL_Y`, already sent to Python but not observed) and vertical (`_accel_z`, which the engine computes but doesn't export yet). A driver feels these as weight transfer, so they're legitimate. The policy has no memory, so it can't work them out by comparing consecutive speeds.
- **Don't add** the load or skid of each wheel (`_reaction[i]`, `_skid[i]`): a driver doesn't feel those directly.

## Reward

- The current terms are sound, except for the terminal penalty.
- **A terminal penalty of 50 is far too high.** With gamma 0.995 at 25 Hz (about an 8 s horizon), driving at 40 m/s is worth about 32 in progress, so a crash costs more than the whole discounted future. proto-9 got slower without completing more laps.

## Diagnosis

Raising the crash penalty made the car slower but didn't make it finish more laps. So it isn't failing because it takes too little care. It fails from **compounding errors**: a small deviation takes it off the one trajectory it knows, and from there it doesn't know how to drive.

## Directions (probably both are needed)

1. **Training dynamics:** mistake learning. Restart training from states just before failures, with variations, so the agent practises getting back onto a good line.
2. **Evaluation revamp:** results flip between checkpoints, and each eval only measures the one trajectory the model knows, with almost no noise. Design to come in the next session.
