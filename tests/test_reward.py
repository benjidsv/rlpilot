"""The reward on hand-made states (no simulator): progress bookkeeping and the cost of losing control."""
import numpy as np

from rlpilot.client import F, OBS_DIM
from rlpilot.reward import COMPONENTS, DAMAGE, LOST_CAUSES, PROGRESS, Reward

P, D, C = (COMPONENTS.index(c) for c in ("progress", "damage", "control"))


def state(dist=0.0, damage=0.0, angle=0.0, to_middle=0.0, wheels_off=0, width=10.0):
    raw = np.zeros((1, OBS_DIM), np.float32)
    raw[0, F["dist_raced"]], raw[0, F["damage"]], raw[0, F["angle"]] = dist, damage, angle
    raw[0, F["to_middle"]], raw[0, F["wheels_off"]] = to_middle, wheels_off
    raw[0, F["track_width"]], raw[0, F["track_length"]] = width, 5000.0
    return raw


def drive(r, states):
    """Step through states; returns the components [steps, len(COMPONENTS)] and onsets [steps, len(LOST_CAUSES)]."""
    comps, onsets = [], []
    for s in states:
        r.step(s)
        comps.append(r.components[:, 0].copy())
        onsets.append(r.onset[:, 0].copy())
    return np.array(comps), np.array(onsets)


def test_best_only_progress():
    r = Reward(1, 0.04)
    r.reset(np.array([0]), state(100))
    # forward 10 m, back 5 m, forward 8 m (3 new), then 4 m fully off the road, then 2 m back on it
    path = [110, 105, 113, 117, 119]
    off = [0, 0, 0, 4, 0]
    comps, _ = drive(r, [state(d, wheels_off=w) for d, w in zip(path, off)])
    assert np.allclose(comps[:, P] / PROGRESS, [10, 0, 3, 0, 2], atol=1e-4)  # reversing free, off-road metres unpaid


def test_signed_progress_is_the_old_reward():
    r = Reward(1, 0.04, lost_cost=0.0, best_only=False)
    r.reset(np.array([0]), state(100))
    path, off = [110, 105, 113, 109, 117], [0, 0, 0, 4, 4]
    comps, _ = drive(r, [state(d, wheels_off=w) for d, w in zip(path, off)])
    assert np.allclose(comps[:, P] / PROGRESS, [10, -5, 8, -4, 0], atol=1e-4)  # reversing costs even off the road
    assert (comps[:, C] == 0).all()


def test_losing_control_is_charged_once_per_entry():
    r = Reward(1, 0.04, lost_cost=2.0)
    r.reset(np.array([0]), state(0))
    hits = [0, 50, 50, 80, 80, 80]                           # one contact over two steps, then nothing more
    angle = [0, 0, 0, 3.0, 3.0, 0]                          # backwards for two steps
    mid = [0, 0, 0, 0, 9.0, 9.0]                            # beyond the edge (5 m half width + 3) from step 4
    comps, onsets = drive(r, [state(10 * (i + 1), h, a, m) for i, (h, a, m) in enumerate(zip(hits, angle, mid))])
    expect = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]], bool)
    assert (onsets == expect).all()
    assert np.allclose(comps[:, C], -2.0 * expect.sum(axis=1))
    assert np.allclose(comps[:, D], -DAMAGE * np.diff([0] + hits))


def test_a_launch_already_out_of_control_isnt_charged():
    r = Reward(1, 0.04)
    r.reset(np.array([0]), state(0, angle=3.0, to_middle=9.0))  # a crashed start: backwards and far off
    comps, onsets = drive(r, [state(1, angle=3.0, to_middle=9.0), state(2, angle=0.0, to_middle=9.0),
                              state(3, angle=3.0, to_middle=0.0)])
    assert (onsets[:, LOST_CAUSES.index("far_off")] == 0).all()
    assert list(onsets[:, LOST_CAUSES.index("backwards")]) == [False, False, True]  # turned round, then back again
    assert np.allclose(comps[:, C], [0, 0, -2.0])


def test_charged_causes_terminal_and_metre_weight():
    r = Reward(1, 0.04, lost_cost=2.0, lost_causes=("contact",), terminal=20.0, metre_weight=lambda raw: np.full(1, 0.5))
    r.reset(np.array([0]), state(0))
    comps, onsets = drive(r, [state(10, angle=3.0), state(20, damage=5, angle=3.0)])
    assert onsets[0, LOST_CAUSES.index("backwards")] and comps[0, C] == 0  # tracked, not charged
    assert comps[1, C] == -2.0
    assert np.allclose(comps[:, P], [0.5, 0.5])  # 10 m at weight 0.5
    r.since_best[:] = 100.0  # the no-progress rule
    comps, _ = drive(r, [state(20, damage=5)])
    assert comps[0, COMPONENTS.index("terminal")] == -20.0
