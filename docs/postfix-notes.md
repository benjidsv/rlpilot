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