import numpy as np

from rlpilot.report import cluster_positions, crash_index, pick_checkpoints, run_tracks, summarize_episode
from rlpilot.splits import PROTO7_TEST, PROTO7_TRAIN, PROTO7_VALIDATION
from rlpilot.track import race_for


def test_report_never_drives_the_test_split():
    params = {"env.race": race_for(PROTO7_TRAIN[0]), "env.races": repr(tuple(race_for(t) for t in PROTO7_TRAIN)),
              "eval.heldout_tracks": ",".join(PROTO7_VALIDATION)}
    tracks = run_tracks(params)
    assert len(tracks) == len(PROTO7_TRAIN) + len(PROTO7_VALIDATION)
    assert not set(tracks) & {race_for(t) for t in PROTO7_TEST}
    assert sum(s == "validation" for s in tracks.values()) == len(PROTO7_VALIDATION)


def test_single_track_run():
    assert run_tracks({"env.race": "rlpilot", "env.races": "()"}) == {"rlpilot": "train"}


def test_pick_checkpoints_keeps_the_last():
    steps = list(range(1, 301))
    picked = pick_checkpoints(steps, 38)
    assert len(picked) == 38 and picked[-1] == 300 and picked == sorted(picked)
    assert pick_checkpoints([1, 2], 38) == [1, 2]


def test_crash_is_where_the_car_left_for_good():
    off = np.array([0, 2, 0, 0, 0, 4, 4, 4], float)  # a kerb at 1 doesn't count
    assert crash_index({"wheels_off": off, "dist_raced": np.arange(8.0)}) == 5


def test_crash_on_the_road_is_the_furthest_point():
    trace = {"wheels_off": np.zeros(5), "dist_raced": np.array([0, 10, 20, 15, 12.0])}
    assert crash_index(trace) == 2


def test_summary():
    trace = {"wheels_off": np.array([0, 0, 4.0]), "dist_raced": np.array([0, 1, 2.0]), "dist_from_start": np.array([9, 10, 11.0]),
             "laps": np.array([1, 1, 1.0])}
    done = summarize_episode(trace, {"end": "lap_done", "progress": 2.0}, 2)
    assert done == {"end": "lap_done", "driven": 1.0}
    crash = summarize_episode(trace, {"end": "offtrack", "progress": 0.5}, 2)
    assert crash == {"end": "offtrack", "driven": 0.25, "crash_i": 2, "crash_m": 11.0, "lap": 1}


def test_cluster_positions():
    assert cluster_positions([], 1000) == []
    spots = cluster_positions([100, 120, 500, 110], 1000)
    assert [(round(at), sorted(m)) for at, m in spots] == [(110, [0, 1, 3]), (500, [2])]


def test_cluster_across_the_start_line_stays_whole():
    ((at, members), other) = cluster_positions([990, 10, 500], 1000)
    assert sorted(members) == [0, 1] and min(at, 1000 - at) < 1e-6
    assert other[1] == [2]
