"""proto-7 observations, steering and penalties, proved with known answers and the real engine.

Measured tolerances (one lap each on several tracks, 2026-11-03), used below as bounds:
- nearest edge point vs the engine's to_left / to_right: 3 cm at p99, 0.8 m at most (Imola's seam, where its map doesn't close)
- edge polylines vs the engine's rays: 0.2 m at p99. Rays that graze an edge at long range are coarser: the ray tracer's
  step grows to 4.5 m at 200 m.
"""

import random
import re
import subprocess
import sys

import numpy as np
import pytest

from rlpilot import EnvConfig, RewardConfig, SpeedDreamsVecEnv, reference, workdirs
from rlpilot._native import A, F, N_FIELDS, N_RAYS
from rlpilot.features import EDGE_GROUPS, EDGE_X_SCALE, EDGE_Y_SCALE, ObsGroup, ObsLayout, assemble, decode_edges, make_layout
from rlpilot.reward import REWARD_COMPONENTS, RewardTracker
from rlpilot.splits import PROTO7_EVAL_TRAIN, PROTO7_TEST, PROTO7_TRAIN, PROTO7_VALIDATION
from rlpilot.track import EdgeTable, LookaheadTable, TrackEdges, TrackProfile, race_for, reference_for

LIMIT = 0.0625  # dyadic: steering sums of multiples of it are exact in float32
POINTS = 30
RAY_DEG = np.array([-90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 75, 90], np.float64)
LAP_TRACKS = ("jarama", "karwada", "Imula", "ardennen-spa", "pinabashi", "corkscrew")


def proto7(**kw):
    defaults = dict(gearbox="points", lookahead=True, car_obs=True, edge_points=POINTS, steer_limit=LIMIT, steer_history=3,
                    spin_us=50)
    return EnvConfig(**(defaults | kw))


def controller(env, step=0, offset=0.0, weave=0.0, speed_gain=0.12, speed_base=8.0):
    """Hand-written driver through the steering increments: holds a lateral offset (m), slow enough to finish laps."""
    raw = env._raw
    width = np.maximum(raw[:, F["track_width"]], 1)
    want = 1.5 * (raw[:, F["angle"]] - 0.5 * (raw[:, F["to_middle"]] - offset) / width)
    want = want + weave * np.sin(step / 25 + np.arange(len(raw)))
    target = speed_base + speed_gain * raw[:, N_FIELDS + N_RAYS // 2]
    pedal = np.where(raw[:, F["speed_x"]] < target, 0.8, np.where(raw[:, F["speed_x"]] > target + 3, -0.6, 0))
    steer = np.clip((want - env._prev_action[:, 0]) / LIMIT, -1, 1)
    return np.concatenate([np.stack([steer, pedal], 1), np.zeros((len(raw), env.act_dim - 2))], 1).astype(np.float32)


@pytest.fixture(scope="module")
def laps():
    """One clean lap per track from the real engine: {track: dict(raw=[t, ...], obs=[t, ...])}, plus the layout."""
    env = SpeedDreamsVecEnv(proto7(n_workers=len(LAP_TRACKS), races=tuple(race_for(t) for t in LAP_TRACKS),
                                   max_episode_seconds=420, laps_per_episode=1))
    try:
        obs, _ = env.reset()
        rows = [[(env._raw[i].copy(), obs[i])] for i in range(len(LAP_TRACKS))]
        done = np.zeros(len(LAP_TRACKS), bool)
        ends = [None] * len(LAP_TRACKS)
        step = 0
        while not done.all():
            obs, _, term, trunc, info = env.step(controller(env, step, weave=0.1))
            step += 1
            for i in np.flatnonzero(~done):
                if term[i] or trunc[i]:
                    done[i] = True
                    ends[i] = env.term_reasons[info["stats"]["term_reason"][i]] if term[i] else "time_limit"
                else:
                    rows[i].append((env._raw[i].copy(), obs[i]))
        assert ends == ["lap_done"] * len(LAP_TRACKS), dict(zip(LAP_TRACKS, ends))
        out = {t: dict(raw=np.stack([r for r, _ in rows[i]]), obs=np.stack([o for _, o in rows[i]]))
               for i, t in enumerate(LAP_TRACKS)}
        return out, env.obs_layout, env._edges
    finally:
        env.close()


# ---------------------------------------------------------------- layout


def test_layout_is_the_single_source_of_truth():
    layout = make_layout(("steer", "pedal", "shift_up", "shift_down"), 9, 0, True, 3, POINTS)
    assert layout.dim == len(layout.names) == len(set(layout.names))
    a = 0
    for g in layout.groups:
        assert layout.slices[g.name] == slice(a, a + len(g.names))
        assert [layout.names[i] for i in range(a, a + len(g.names))] == list(g.names)
        assert g.scale
        a += len(g.names)
    assert a == layout.dim
    obs = np.arange(layout.dim, dtype=np.float32)
    for name in layout.names:
        assert layout.get(obs, name) == layout.index(name)
    assert layout.group(obs, "steer_history").tolist() == [layout.index(f"steer_change{i}") for i in range(3)]
    assert 12 + 19 + 4 + 9 + 3 + 3 + 4 * POINTS == layout.dim
    with pytest.raises(ValueError):
        ObsLayout((ObsGroup("a", ("x",), "-"), ObsGroup("b", ("x",), "-")))


def test_layout_keeps_the_proto6_order_for_old_checkpoints():
    layout = make_layout(("steer", "pedal", "shift_up", "shift_down"), 9, 8, True)
    assert layout.dim == 55
    assert layout.names[:12] == ("speed_x", "speed_y", "speed_z", "yaw_rate", "angle", "to_middle", "rpm", "gear",
                                 "wheel_spin0", "wheel_spin1", "wheel_spin2", "wheel_spin3")
    assert [g.name for g in layout.groups][-4:] == ["prev_action", "lookahead", "line", "car"]
    assert layout.slices["rays"] == slice(12, 31) and layout.slices["prev_action"] == slice(31, 35)
    assert layout.slices["lookahead"] == slice(35, 44) and layout.slices["line"] == slice(44, 52)


def test_assemble_rejects_missing_extra_and_wrong_width_groups():
    layout = ObsLayout((ObsGroup("prev_action", ("a", "b"), "-"), ObsGroup("c", ("d",), "-")))
    good = {"prev_action": np.ones((3, 2), np.float32), "c": np.zeros(3, np.float32)}
    assert assemble(layout, good).shape == (3, 3)
    with pytest.raises(ValueError):
        assemble(layout, {"prev_action": good["prev_action"]})
    with pytest.raises(ValueError):
        assemble(layout, good | {"extra": good["c"]})
    with pytest.raises(ValueError):
        assemble(layout, {"prev_action": np.ones((3, 1), np.float32), "c": good["c"]})  # would broadcast


def test_every_group_holds_its_documented_value_on_the_real_engine():
    env = SpeedDreamsVecEnv(proto7(n_workers=2, races=(race_for("jarama"), race_for("braga")), max_episode_seconds=60,
                                   car_grip=(0.9, 1.1), car_mass=(0.9, 1.1), car_power=(0.9, 1.1)))
    try:
        env.reset()
        assert env.single_observation_space.shape == (env.obs_layout.dim,)
        for t in range(120):
            obs, *_ = env.step(controller(env, t))
        r, o, L = env._raw, obs, env.obs_layout
        half = r[:, F["track_width"]] * 0.5
        for name, want in (("speed_x", r[:, F["speed_x"]] / 50), ("speed_y", r[:, F["speed_y"]] / 10),
                           ("speed_z", r[:, F["speed_z"]] / 10), ("yaw_rate", r[:, F["yaw_rate"]] / 2),
                           ("angle", r[:, F["angle"]] / np.pi), ("to_middle", r[:, F["to_middle"]] / half),
                           ("rpm", r[:, F["rpm"]] / r[:, F["rpm_redline"]]), ("gear", r[:, F["gear"]] / 6)):
            np.testing.assert_allclose(L.get(o, name), want, rtol=1e-5, err_msg=name)
        np.testing.assert_allclose(L.group(o, "rays"), r[:, N_FIELDS:N_FIELDS + N_RAYS] / 200, rtol=1e-5)
        np.testing.assert_allclose(L.group(o, "car"), (env.car_factors - 1) * 10, rtol=1e-5)
        np.testing.assert_array_equal(L.group(o, "prev_action"), env.applied)
        look = LookaheadTable(env.races, "peak")
        np.testing.assert_array_equal(L.group(o, "lookahead"),
                                      look(r[:, F["dist_from_start"]], *look.rows(env.client.races)))
    finally:
        env.close()


# ---------------------------------------------------------------- synthetic geometry


def circle_track(radius=100.0, half_width=5.0, turn="left"):
    """Constant-radius loop, sampled every ~2 m like the engine's export. Starts at (radius, 0) heading +y (left turn)."""
    n = int(2 * np.pi * radius / 2)
    a = np.arange(n) * 2 * np.pi / n
    sign = 1 if turn == "left" else -1
    ring = lambda r: np.stack([r * np.cos(a), sign * r * np.sin(a)], 1)
    # left of the direction of travel is towards the centre in a left turn
    left, right = (ring(radius - half_width), ring(radius + half_width)) if turn == "left" else \
        (ring(radius + half_width), ring(radius - half_width))
    return left, right, 2 * np.pi * radius


def stadium_track(straight=1000.0, radius=50.0, half_width=5.0):
    """Counter-clockwise stadium, starting on the first straight along +x from the origin: straight, half circle, straight,
    half circle. Sampled every 2 m."""
    xs = np.arange(0, straight, 2.0)
    arc_angles = np.arange(0, np.pi, 2.0 / radius)
    mid = np.vstack([
        np.stack([xs, np.zeros_like(xs)], 1),
        np.stack([straight + radius * np.sin(arc_angles), radius - radius * np.cos(arc_angles)], 1),
        np.stack([straight - xs, np.full_like(xs, 2 * radius)], 1),
        np.stack([-radius * np.sin(arc_angles), radius + radius * np.cos(arc_angles)], 1),
    ])
    seg = np.diff(np.vstack([mid, mid[:1]]), axis=0)
    tangent = seg / np.hypot(*seg.T)[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
    length = 2 * straight + 2 * np.pi * radius
    return mid + half_width * normal, mid - half_width * normal, length


def synthetic_edges(left, right, length, n_points=POINTS, horizon=6.0, floor=60.0):
    """An EdgeTable over one synthetic track, with no reference files involved."""
    edges = TrackEdges(left, right, length)
    t = EdgeTable.__new__(EdgeTable)
    t.horizon, t.floor = horizon, floor
    t.fractions = np.arange(n_points, dtype=np.float64)[None, :] / (n_points - 1)
    t.table = edges.table
    rows = (np.zeros(1, np.int64), np.array([edges.cells], np.int64), np.array([edges.spacing], np.float64))
    return t, rows


def view(t, rows, s, speed, x=0.0, y=0.0, yaw=0.0):
    """Left and right points in the car's frame (m), [n_points, 2] each."""
    a = np.float32
    world = t.world(np.array([s]), np.array([speed], a), *rows)
    out = t.car_frame(np.array([x], a), np.array([y], a), np.array([yaw], a), world)[0]
    return out[:, :2], out[:, 2:]


def test_straight_gives_exact_edge_points():
    t, rows = synthetic_edges(*stadium_track())
    speed = 20.0  # span 120 m, from s = 100
    left, right = view(t, rows, 100.0, speed, x=100.0, y=0.0, yaw=0.0)
    ahead = np.arange(POINTS) * 120.0 / (POINTS - 1)
    np.testing.assert_allclose(left[:, 0], ahead, atol=1e-3)
    np.testing.assert_allclose(right[:, 0], ahead, atol=1e-3)
    np.testing.assert_allclose(left[:, 1], 5.0, atol=1e-3)
    np.testing.assert_allclose(right[:, 1], -5.0, atol=1e-3)
    # offset 2 m to the left of the centreline: left edge 3 m away, right edge 7 m
    left, right = view(t, rows, 100.0, speed, x=100.0, y=2.0, yaw=0.0)
    np.testing.assert_allclose(left[:, 1], 3.0, atol=1e-3)
    np.testing.assert_allclose(right[:, 1], -7.0, atol=1e-3)
    # yawed 0.1 rad to the left, the road ahead appears rotated to the right
    left, right = view(t, rows, 100.0, speed, x=100.0, y=0.0, yaw=0.1)
    c, s = np.cos(0.1), np.sin(0.1)
    np.testing.assert_allclose(left[:, 0], c * ahead + s * 5, atol=1e-3)
    np.testing.assert_allclose(left[:, 1], c * 5 - s * ahead, atol=1e-3)


@pytest.mark.parametrize("turn", ["left", "right"])
def test_constant_radius_circle_gives_exact_edge_points(turn):
    r, hw = 100.0, 5.0
    t, rows = synthetic_edges(*circle_track(r, hw, turn))
    sign = 1 if turn == "left" else -1
    span = 150.0  # 25 m/s x 6 s
    for theta0 in (0.0, 1.0, 4.0):  # along the loop; the expected answer doesn't depend on it
        s0 = theta0 * r
        x0, y0 = r * np.cos(theta0), sign * r * np.sin(theta0)
        yaw = sign * (theta0 + np.pi / 2)
        left, right = view(t, rows, s0, 25.0, x0, y0, yaw)
        dtheta = (np.arange(POINTS) * span / (POINTS - 1)) / r
        # in the car's frame, with the circle's centre on the side it turns to (sign x r along y)
        r_left, r_right = (r - hw, r + hw) if turn == "left" else (r + hw, r - hw)
        for pts, rad in ((left, r_left), (right, r_right)):
            np.testing.assert_allclose(pts[:, 0], rad * np.sin(dtheta), atol=0.02)
            np.testing.assert_allclose(pts[:, 1], sign * (r - rad * np.cos(dtheta)), atol=0.02)


def test_the_frame_ignores_where_the_track_is_in_the_world():
    left, right, length = circle_track()
    a = 2.1
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    shift = np.array([1500.0, -900.0])
    t, rows = synthetic_edges(left, right, length)
    t2, rows2 = synthetic_edges(left @ rot.T + shift, right @ rot.T + shift, length)
    base = view(t, rows, 0.0, 30.0, 100.0, 0.0, np.pi / 2)
    moved_pos = rot @ np.array([100.0, 0.0]) + shift
    moved = view(t2, rows2, 0.0, 30.0, moved_pos[0], moved_pos[1], np.pi / 2 + a)
    for p, q in zip(base, moved):
        np.testing.assert_allclose(p, q, atol=3e-3)


def test_a_map_that_does_not_close_stays_anchored_at_the_lap_length():
    """Imola: 3 m too long a last gap. Distances before the seam stay true rather than shrinking over the whole lap."""
    left, right, length = stadium_track()
    edges = TrackEdges(left[:-2], right[:-2], length)  # the last gap is now ~6 m in the map, while the lap is unchanged
    np.testing.assert_allclose(edges.table[100, :2], [100 * edges.spacing, 5.0], atol=1e-2)
    assert edges.table.shape[0] == edges.cells + 1
    np.testing.assert_array_equal(edges.table[0], edges.table[-1])
    with pytest.raises(ValueError):
        TrackEdges(left, right, 100.0)  # a lap shorter than its map is a broken file


def test_distance_mapping_matches_the_engine_on_all_tracks():
    """Engine truth from the bot's traces (here only, as a measuring tape): the point of the map at a car's reported
    distance is abreast of the car. Bound: 0.5 m at p99 on every track, 3.5 m at most (Imola's seam)."""
    tracks = [line.split(",")[0] for line in (workdirs.ROOT / "reference" / "tracks.csv").read_text().splitlines()[1:]]
    assert len(tracks) == 41
    for track in tracks:
        ref = reference.load(reference_for(race_for(track)), laps=3)
        tr = ref.trace
        ok = tr["dist_raced"] > 0
        e = TrackEdges.for_race(race_for(track))
        mid = 0.5 * (e.table[:, :2] + e.table[:, 2:])
        u = np.mod(tr["dist_from_start"][ok], e.length) / e.spacing
        i = np.minimum(u.astype(np.int64), e.cells - 1)
        tangent = mid[i + 1] - mid[i]
        tangent /= np.hypot(*tangent.T)[:, None]
        here = mid[i] + (u - i)[:, None] * (mid[i + 1] - mid[i])
        along = ((np.stack([tr["x"][ok], tr["y"][ok]], 1) - here) * tangent).sum(1)
        assert np.percentile(np.abs(along), 99) < 0.5, track
        assert np.abs(along).max() < 3.5, track


# ---------------------------------------------------------------- engine agreement


def test_world_frame_conventions_are_what_the_code_assumes(laps):
    """Position and yaw, checked from the motion: the car moves along (cos yaw, sin yaw); y = left is (-sin yaw, cos yaw)."""
    data, _, _ = laps
    raw = data["jarama"]["raw"]
    dpos = np.diff(raw[:, [F["pos_x"], F["pos_y"]]], axis=0)
    dt = np.diff(raw[:, F["time"]])
    yaw = raw[:-1, F["yaw"]] + 0.5 * np.diff(np.unwrap(raw[:, F["yaw"]]))
    vx, vy = (raw[:-1, F["speed_x"]] + raw[1:, F["speed_x"]]) / 2, (raw[:-1, F["speed_y"]] + raw[1:, F["speed_y"]]) / 2
    fwd = dpos[:, 0] * np.cos(yaw) + dpos[:, 1] * np.sin(yaw)
    lat = -dpos[:, 0] * np.sin(yaw) + dpos[:, 1] * np.cos(yaw)
    moving = (vx > 5) & (dt > 0)
    assert moving.sum() > 2000
    assert np.percentile(np.abs(fwd[moving] / dt[moving] - vx[moving]), 95) < 0.3
    assert np.percentile(np.abs(lat[moving] / dt[moving] - vy[moving]), 95) < 0.3
    # + yaw rate = yaw growing = turning left; ray and steering signs are tested below
    dyaw = np.diff(np.unwrap(raw[:, F["yaw"]]))[moving] / dt[moving]
    assert np.corrcoef(dyaw, raw[:-1, F["yaw_rate"]][moving])[0, 1] > 0.95


@pytest.mark.parametrize("track", LAP_TRACKS)
def test_nearest_edge_points_match_to_left_and_to_right(laps, track):
    data, layout, _ = laps
    raw, obs = data[track]["raw"], data[track]["obs"]
    left, right = decode_edges(layout, obs)
    on = (raw[:, F["to_left"]] > 0.3) & (raw[:, F["to_right"]] > 0.3)
    assert on.mean() > 0.95
    err_l = np.hypot(*left[on, 0].T) - raw[on, F["to_left"]]
    err_r = np.hypot(*right[on, 0].T) - raw[on, F["to_right"]]
    for e in (err_l, err_r):
        assert np.percentile(np.abs(e), 99) < 0.05, track
        assert np.abs(e).max() < 1.0, track


def _first_hit(poly, angle):
    """Distance from the origin along `angle` to polylines [m, 2], or nan."""
    d = np.array([np.cos(angle), np.sin(angle)])
    a, e = poly[:-1], np.diff(poly, axis=0)
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a[:, 0] * e[:, 1] - a[:, 1] * e[:, 0]) / den
        u = (a[:, 0] * d[1] - a[:, 1] * d[0]) / den
    hit = t[(t > 0) & (u >= 0) & (u <= 1)]
    return hit.min() if len(hit) else np.nan


@pytest.mark.parametrize("track", LAP_TRACKS)
def test_edge_points_agree_with_the_engine_rays(laps, track):
    """Intersecting the rays with what the policy sees gives the engine's distances: 0.25 m at p99. (A border behind point 0
    or past the last point isn't in the observation: rarely, in a tight corner, a ray can only hit one further on, so
    the worst case isn't bounded here. The next test covers those rays.)"""
    data, layout, _ = laps
    raw, obs = data[track]["raw"][::5], data[track]["obs"][::5]
    left, right = decode_edges(layout, obs)
    errs = []
    for t in range(len(raw)):
        if raw[t, F["speed_x"]] < 3:
            continue
        for j, deg in enumerate(RAY_DEG):
            engine = raw[t, N_FIELDS + j]
            if engine >= 199:
                continue
            hits = [_first_hit(left[t], np.deg2rad(deg)), _first_hit(right[t], np.deg2rad(deg))]
            if not np.isnan(hits).all():
                errs.append(np.nanmin(hits) - engine)
    errs = np.abs(errs)
    assert len(errs) > 5000
    assert np.percentile(errs, 99) < 0.25, track
    assert (errs > 2).mean() < 0.002, track


@pytest.mark.parametrize("track", ["jarama", "corkscrew"])
def test_engine_rays_end_on_the_map(laps, track):
    """All rays under 60 m end at a border of the map we built (sampled every ~1 m)."""
    data, _, _ = laps
    raw = data[track]["raw"][::5]
    table = TrackEdges.for_race(race_for(track)).table
    worst = 0.0
    for r in raw[raw[:, F["speed_x"]] > 3]:
        for j, deg in enumerate(RAY_DEG):
            d = r[N_FIELDS + j]
            if d < 60:
                a = r[F["yaw"]] + np.deg2rad(deg)
                x, y = r[F["pos_x"]] + d * np.cos(a), r[F["pos_y"]] + d * np.sin(a)
                worst = max(worst, min(np.hypot(table[:, 0] - x, table[:, 1] - y).min(),
                                       np.hypot(table[:, 2] - x, table[:, 3] - y).min()))
    assert worst < 1.5


def test_angle_matches_the_road_direction_the_points_show(laps):
    """The engine's angle (+ = road points left of the car) is the direction the points give, from two different sources."""
    data, layout, _ = laps
    raw, obs = data["pinabashi"]["raw"], data["pinabashi"]["obs"]
    left, right = decode_edges(layout, obs)
    mid = 0.5 * (left + right)
    d = mid[:, 1] - mid[:, 0]
    err = np.angle(np.exp(1j * (np.arctan2(d[:, 1], d[:, 0]) - raw[:, F["angle"]])))
    assert np.percentile(np.abs(err), 95) < 0.05


# ---------------------------------------------------------------- signs


@pytest.mark.parametrize("sign,name", [(1, "left"), (-1, "right")])
def test_points_bend_towards_the_corner(laps, sign, name):
    """In corners read from the curvature lookahead (+ = left), the road ahead bends that way in the car's frame, on both
    edges, measured relative to the road's direction at the car."""
    data, layout, _ = laps
    for track in ("jarama", "corkscrew", "ardennen-spa"):
        raw, obs = data[track]["raw"], data[track]["obs"]
        # bending the same way over all of the next 75 m (0.2 = 125 m radius), so that the road can only go that way
        corner = (sign * layout.group(obs, "lookahead")[:, :4] > 0.2).all(axis=1) & (raw[:, F["speed_x"]] > 5)
        assert corner.sum() > 20, (track, name)
        left, right = decode_edges(layout, obs[corner])
        a = raw[corner, F["angle"]]
        speed = np.hypot(raw[corner, F["speed_x"]], raw[corner, F["speed_y"]])
        k = np.minimum((50.0 / np.maximum(6 * speed, 60.0) * (POINTS - 1)).astype(int), POINTS - 1)  # ~50 m ahead
        for pts in (left, right):
            far = pts[np.arange(len(k)), k]
            road_y = lambda p: p[:, 1] * np.cos(a) - p[:, 0] * np.sin(a)  # in the road's direction at the car
            assert (sign * (road_y(far) - road_y(pts[:, 0])) > 0).mean() > 0.97, (track, name)


def test_car_offset_moves_the_points_the_expected_way():
    env = SpeedDreamsVecEnv(proto7(n_workers=2, max_episode_seconds=30))
    try:
        env.reset()
        offset = np.array([2.0, -2.0], np.float32)
        for t in range(220):
            obs, *_ = env.step(controller(env, t, offset=offset, speed_gain=0.04))
        r = env._raw
        assert r[0, F["to_middle"]] > 1.0 and r[1, F["to_middle"]] < -1.0  # the car left of centre, the car right of it
        left, right = decode_edges(env.obs_layout, obs)
        gap = r[0, F["to_middle"]] - r[1, F["to_middle"]]
        # the car further left sees the left edge closer and the right edge further, by the gap between the two cars
        assert 0 < left[0, 0, 1] < left[1, 0, 1] and right[0, 0, 1] < right[1, 0, 1] < 0
        np.testing.assert_allclose(left[1, 0, 1] - left[0, 0, 1], gap, atol=0.25)
        np.testing.assert_allclose(right[1, 0, 1] - right[0, 0, 1], gap, atol=0.25)
    finally:
        env.close()


# ---------------------------------------------------------------- lap wrap


def test_wrap_is_exact_in_the_table():
    t, rows = synthetic_edges(*circle_track())
    length = 2 * np.pi * 100
    a = np.float32
    for s in (0.0, 5.0, length - 1.0, length + 5.0, -2.0):  # -2: a car in Imola's seam reports a small negative distance
        here = t.world(np.array([s]), np.array([25.0], a), *rows)
        there = t.world(np.array([s + length]), np.array([25.0], a), *rows)
        np.testing.assert_allclose(here, there, atol=5e-3)
    # points just before and after the line are neighbours
    w = t.world(np.array([length - 75.0]), np.array([25.0], a), *rows)[0]  # the line is half way along the 150 m span
    assert np.hypot(*np.diff(w[:, :2], axis=0).T).max() < 150.0 / (POINTS - 1) * 1.02


@pytest.mark.parametrize("track", LAP_TRACKS)
def test_no_jump_across_the_start_line_including_the_grid(laps, track):
    data, layout, edges = laps
    raw, obs = data[track]["raw"], data[track]["obs"]
    length = raw[0, F["track_length"]]
    assert raw[0, F["dist_raced"]] < 0 and raw[0, F["dist_from_start"]] > length - 30  # on the grid, before the line
    # back to world coordinates, from what the policy sees
    left, right = decode_edges(layout, obs)
    c, s = np.cos(raw[:, F["yaw"]])[:, None], np.sin(raw[:, F["yaw"]])[:, None]
    world_x = lambda p: raw[:, F["pos_x"]][:, None] + c * p[..., 0] - s * p[..., 1]
    world_y = lambda p: raw[:, F["pos_y"]][:, None] + s * p[..., 0] + c * p[..., 1]
    speed = np.hypot(raw[:, F["speed_x"]], raw[:, F["speed_y"]])
    # continuous along the path, whereas dist_from_start jumps at the line. The fastest a point can move in the world is the
    # rate of its distance along the lap, times 2.6 (the outside of a bend; Imola's seam is 5 m of map in 2 m of lap)
    path = np.maximum.accumulate(raw[:, F["dist_raced"]])
    grid = raw[:, F["dist_raced"]] < 0
    path = np.where(grid, raw[:, F["dist_from_start"]] - length, path)
    moved = np.diff(path)[:, None] + np.diff(edges.span(speed))[:, None] * edges.fractions
    for p in (left, right):
        jump = np.hypot(np.diff(world_x(p), axis=0), np.diff(world_y(p), axis=0))
        assert (jump <= 2.6 * np.abs(moved) + 0.25).all(), (track, jump.max())
    # the grid's first observation sees across the line with no gap between neighbouring points
    for p in (left[0], right[0]):
        assert np.hypot(*np.diff(p, axis=0).T).max() < 60.0 / (POINTS - 1) * 2.6
    # ... and the lap did cross the line
    assert (np.diff(raw[:, F["dist_from_start"]]) < -length / 2).sum() == 1


def test_figure_eight_never_jumps_to_the_other_branch(laps):
    data, layout, edges = laps
    raw = data["karwada"]["raw"]
    table = TrackEdges.for_race(race_for("karwada"))
    mid = 0.5 * (table.table[:, :2] + table.table[:, 2:])
    # where the two branches cross: close in the world, far along the lap
    coarse = np.arange(0, len(mid), 3)
    d = np.hypot(mid[coarse, None, 0] - mid[coarse, 0][None], mid[coarse, None, 1] - mid[coarse, 1][None])
    far = np.abs(coarse[:, None] - coarse[None]) * table.spacing
    far = np.minimum(far, table.length - far) > 500
    i, j = np.unravel_index(np.argmin(np.where(far, d, np.inf)), d.shape)
    s1, s2 = coarse[i] * table.spacing, coarse[j] * table.spacing
    assert d[i, j] < 15, "karwada should cross itself"
    # the car passed both branches, and at each the nearest points lie on its own branch
    obs = data["karwada"]["obs"]
    left, right = decode_edges(layout, obs)
    for s0 in (s1, s2):
        t = int(np.argmin(np.abs(raw[:, F["dist_from_start"]] - s0)))
        assert abs(raw[t, F["dist_from_start"]] - s0) < 2
        # the road ahead continues in the car's direction on its own branch, not across it
        np.testing.assert_allclose(np.hypot(*left[t, 0]), raw[t, F["to_left"]], atol=0.2)
        np.testing.assert_allclose(np.hypot(*right[t, 0]), raw[t, F["to_right"]], atol=0.2)
        for p in (left[t], right[t]):
            assert np.hypot(*np.diff(p, axis=0).T).max() < edges.span(raw[[t], F["speed_x"]])[0] / (POINTS - 1) * 2.6


# ---------------------------------------------------------------- speed scaling


def test_span_follows_speed_and_uses_the_floor():
    t, rows = synthetic_edges(*stadium_track())
    a = np.float32
    for speed, span in ((0.0, 60.0), (5.0, 60.0), (10.0, 60.0), (10.5, 63.0), (20.0, 120.0), (70.0, 420.0), (-20.0, 120.0)):
        left, _ = view(t, rows, 0.0, speed, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(left[-1, 0], span, atol=1e-3, err_msg=str(speed))
    np.testing.assert_allclose(t.span(np.array([3.0, 30.0], a)), [60.0, 180.0])


def test_the_same_corner_appears_at_an_earlier_index_when_faster():
    """On a known straight of 1000 m, at 800 m: the corner starts 200 m ahead, which is index 200 / span x (n - 1)."""
    t, rows = synthetic_edges(*stadium_track(straight=1000.0, radius=50.0))
    corner_index = {}
    for speed in (40.0, 60.0, 80.0):
        left, right = view(t, rows, 800.0, speed, 800.0, 0.0, 0.0)
        mid_y = 0.5 * (left[:, 1] + right[:, 1])
        first = int(np.argmax(np.abs(mid_y) > 0.05))
        expected = int(np.ceil(200.0 / (6.0 * speed) * (POINTS - 1)))
        assert first in (expected, expected + 1, expected + 2), (speed, first, expected)
        corner_index[speed] = first
    assert corner_index[80.0] < corner_index[60.0] < corner_index[40.0]


def test_span_matches_speed_on_the_real_engine_and_a_real_corner_comes_earlier():
    """Mid-line length of the points = the span. Two cars pass the same spot at different speeds: the faster car has the
    corner ahead at an earlier point index."""
    env = SpeedDreamsVecEnv(proto7(n_workers=2, max_episode_seconds=40))
    try:
        obs, _ = env.reset()
        L = env.obs_layout
        left, right = decode_edges(L, obs)
        np.testing.assert_allclose(left[:, -1, 0], 60.0, atol=1.5)  # near standstill on the grid: the floor, ahead
        table = TrackEdges.for_race(race_for("jarama")).table
        mid = 0.5 * (table[:, :2] + table[:, 2:])
        # a spot where the road has left its direction by 8 m within 85 m, in view of both cars (span >= 90 m)
        for s0 in range(10, 1000, 5):
            h = mid[s0 + 5] - mid[s0]
            n = np.array([-h[1], h[0]]) / np.hypot(*h)
            lateral = (mid[s0 + 85] - mid[s0]) @ n
            if abs(lateral) > 8:
                break
        assert abs(lateral) > 8
        seen: dict[int, np.ndarray] = {}
        speed_at: dict[int, float] = {}
        for t in range(1000):
            a = controller(env, t, speed_gain=0.0, speed_base=np.array([15.0, 45.0], np.float32))
            obs, *_ = env.step(a)
            r = env._raw
            for i in range(2):
                if i not in seen and 0 < r[i, F["dist_raced"]] and r[i, F["dist_from_start"]] >= s0:
                    seen[i], speed_at[i] = obs[i].copy(), float(np.hypot(r[i, F["speed_x"]], r[i, F["speed_y"]]))
            if len(seen) == 2:
                break
        assert len(seen) == 2 and speed_at[1] > speed_at[0] + 5
        for i in range(2):
            l, rr = decode_edges(L, seen[i])
            m = 0.5 * (l + rr)
            arc = np.hypot(*np.diff(m, axis=0).T).sum()
            np.testing.assert_allclose(arc, 6.0 * speed_at[i], rtol=0.03)
        first = []
        for i in range(2):
            l, rr = decode_edges(L, seen[i])
            m = 0.5 * (l + rr)
            first.append(int(np.argmax(np.abs(m[:, 1] - m[0, 1]) > 6.0)))
        assert 0 < first[1] < first[0], (first, speed_at)
    finally:
        env.close()


# ---------------------------------------------------------------- episode boundaries


def test_first_observations_and_final_obs_across_a_same_step_autoreset():
    env = SpeedDreamsVecEnv(proto7(n_workers=2, max_episode_seconds=2.0))
    L = env.obs_layout
    try:
        first, _ = env.reset()
        for o in first:
            assert L.get(o, "prev_steer") == 0 and (L.group(o, "steer_history") == 0).all()
            assert (L.group(o, "prev_action") == 0).all()
        steps = int(round(2.0 / env.cfg.step_seconds))
        actions = np.array([[0.25, 0.5, 0.0, 0.0], [-0.125, 0.5, 0.0, 0.0]], np.float32)  # stays short of the lock
        for _ in range(steps):
            obs, _, term, trunc, info = env.step(actions)
        assert (term | trunc).all()
        # the new episode's first observation: zeros, and the same as after reset() (the sim is deterministic)
        for i in range(2):
            assert L.get(obs[i], "prev_steer") == 0
            assert (L.group(obs[i], "steer_history") == 0).all()
            np.testing.assert_array_equal(obs[i], first[i])
            # final_obs: still the pre-reset state, with the steering it had
            final = info["final_obs"][i]
            np.testing.assert_allclose(L.get(final, "prev_steer"), actions[i, 0] * LIMIT * steps, atol=1e-5)
            np.testing.assert_allclose(L.group(final, "steer_history"), actions[i, 0], atol=1e-6)
            raw = info["final_raw"][i]
            np.testing.assert_allclose(L.get(final, "speed_x"), raw[F["speed_x"]] / 50, rtol=1e-6)
            assert raw[F["dist_raced"]] > raw[F["track_length"]] * -1 and L.get(final, "speed_x") > 0.05
            assert not np.array_equal(final, first[i])
            # the last observed controls survive for evaluation; the next step starts from 0
            np.testing.assert_allclose(env.applied[i, 0], actions[i, 0] * LIMIT * steps, atol=1e-5)
        env.step(actions)
        np.testing.assert_allclose(env._act[:, A["steer"]], actions[:, 0] * LIMIT, atol=1e-7)
    finally:
        env.close()


def test_partial_batch_autoreset_leaves_the_other_envs_alone():
    """Env 0 ends every 1 s while env 1 goes on: env 1 behaves as if nothing happened, env 0 restarts cleanly each time."""
    def run(limits):
        env = SpeedDreamsVecEnv(proto7(n_workers=2, max_episode_seconds=20))
        env.set_time_limits(np.array(limits))
        L = env.obs_layout
        try:
            first, _ = env.reset()
            out, resets = [], 0
            for t in range(120):
                a = np.array([[0.1 * np.sin(t / 7), 0.6, 0, 0], [0.1 * np.cos(t / 9), 0.6, 0, 0]], np.float32)
                obs, _, term, trunc, info = env.step(a)
                out.append(obs[1].copy())
                if (term | trunc)[0]:
                    resets += 1
                    np.testing.assert_array_equal(obs[0], first[0])
                    assert not (term | trunc)[1] and info["final_obs"][1] is None and info["final_raw"][1] is None
                    assert L.get(obs[1], "prev_steer") != 0  # untouched
            return np.stack(out), resets
        finally:
            env.close()

    alone, r0 = run([20.0, 20.0])
    partial, r1 = run([1.0, 20.0])
    assert r0 == 0 and r1 == 4
    np.testing.assert_array_equal(alone, partial)


def test_worker_recycling_keeps_the_track_and_its_tables():
    races = (race_for("jarama"), race_for("braga"))

    def run(recycle):
        env = SpeedDreamsVecEnv(proto7(n_workers=2, races=races, max_episode_seconds=1.0, recycle_resets=recycle))
        try:
            first, _ = env.reset()
            pids = [p.pid for p in env.client.procs]
            rows = [r.copy() for r in env._edge_rows]
            out = []
            for t in range(4 * 25 + 5):
                obs, *_ = env.step(np.array([[np.sin(t / 5), 0.5, 0, 0]] * 2, np.float32))
                out.append(obs.copy())
                assert list(env.client.races) == list(races)
                for a, b in zip(rows, env._edge_rows):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_allclose(env._raw[:, F["track_length"]], [env._edges.cells[r] * env._edges.spacing[r]
                                                                            for r in races], atol=1e-2)
            return np.stack(out), pids != [p.pid for p in env.client.procs], env.client.races
        finally:
            env.close()

    normal, respawned0, _ = run(10_000)
    recycled, respawned, races_after = run(1)
    assert respawned and not respawned0 and list(races_after) == list(races)
    np.testing.assert_array_equal(normal, recycled)


def test_track_switch_moves_every_table_to_the_new_track():
    races = (race_for("jarama"), race_for("braga"), race_for("pinabashi"))
    env = SpeedDreamsVecEnv(proto7(n_workers=3, races=races, max_episode_seconds=0.4, track_switch_steps=1))
    L = env.obs_layout
    try:
        obs, _ = env.reset()
        changed = False
        for t in range(40):
            obs, _, term, trunc, _ = env.step(np.zeros((3, env.act_dim), np.float32))
            done = term | trunc
            changed |= list(env.client.races) != list(races)
            if done.any():  # first observations of the (maybe new) tracks: the tables must agree with the engine
                left, right = decode_edges(L, obs)
                r = env._raw
                np.testing.assert_allclose(np.hypot(*left[:, 0].T)[done], r[done, F["to_left"]], atol=0.15)
                np.testing.assert_allclose(np.hypot(*right[:, 0].T)[done], r[done, F["to_right"]], atol=0.15)
                look = env._look(r[:, F["dist_from_start"]], env._look_off, env._look_size)
                np.testing.assert_array_equal(L.group(obs[done], "lookahead"), look[done])
                for i in np.flatnonzero(done):
                    race = env.client.races[i]
                    assert env._race_index[i] == env.races.index(race)
                    assert env._edge_rows[0][i] == env._edges.offset[race] and env._look_off[i] == env._look.offset[race]
                    assert env._line_off[i] == env._line.offset[race]
        assert changed
    finally:
        env.close()


def test_workers_must_cover_tracks_equally_without_switching():
    with pytest.raises(ValueError, match="multiple"):
        SpeedDreamsVecEnv(EnvConfig(n_workers=3, races=(race_for("jarama"), race_for("braga"))))


def test_track_coverage_counts_every_worker_per_step():
    races = (race_for("jarama"), race_for("braga"))
    env = SpeedDreamsVecEnv(proto7(n_workers=4, races=races, max_episode_seconds=10))
    try:
        env.reset()
        for _ in range(7):
            env.step(np.zeros((4, env.act_dim), np.float32))
        np.testing.assert_array_equal(env.track_steps, [14, 14])
        env.step(np.zeros((4, env.act_dim), np.float32), np.array([True, False, False, False]))
        np.testing.assert_array_equal(env.track_steps, [15, 14])
    finally:
        env.close()


# ---------------------------------------------------------------- steering


def test_steering_increment_limit_and_clipping_are_exact(tmp_path, monkeypatch):
    record = tmp_path / "controls.bin"
    monkeypatch.setenv("RLP_RECORD_CONTROLS", str(record))
    env = SpeedDreamsVecEnv(proto7(n_workers=1, max_episode_seconds=60))
    L = env.obs_layout
    try:
        obs, _ = env.reset()
        assert L.get(obs[0], "prev_steer") == 0
        sequence = [1.0, 1.0, -0.5, 0.25, 0.0, 5.0, -5.0]  # actions: +-5 are clipped to +-1 before scaling
        expected_steer = [0.0625, 0.125, 0.09375, 0.109375, 0.109375, 0.171875, 0.109375]
        expected_change = [0.0625, 0.0625, -0.03125, 0.015625, 0.0, 0.0625, -0.0625]
        history = [0.0, 0.0, 0.0]
        applied = []
        for action, steer, change in zip(sequence, expected_steer, expected_change):
            obs, *_ = env.step(np.array([[action, 0.5, 0, 0]], np.float32))
            assert env._act[0, A["steer"]] == steer                        # the shared-memory slot, exactly
            assert env.applied[0, 0] == steer and L.get(obs[0], "prev_steer") == steer
            history = [change / LIMIT] + history[:2]
            assert L.group(obs[0], "steer_history").tolist() == history    # most recent first
            applied.append(steer)
        # up to full lock, and it stays there: the change at the lock is 0, whatever is requested
        for _ in range(18):
            obs, *_ = env.step(np.array([[1.0, 0.5, 0, 0]], np.float32))
        assert env.applied[0, 0] == 1.0 and env._act[0, A["steer"]] == 1.0
        assert L.group(obs[0], "steer_history").tolist() == [0.0, 0.0, 0.0]
        applied += [min(1.0, applied[-1] + LIMIT * (i + 1)) for i in range(18)]
        for _ in range(3):
            obs, *_ = env.step(np.array([[-1.0, 0.5, 0, 0]], np.float32))
        assert env.applied[0, 0] == 1.0 - 3 * LIMIT
        applied += [1.0 - (i + 1) * LIMIT for i in range(3)]
    finally:
        env.close()
    # the engine's car control at every robot tick, not just what Python wrote
    ticks = np.fromfile(record, dtype=[("steer", "f4"), ("accel", "f4"), ("brake", "f4"), ("clutch", "f4"), ("gear", "i4"),
                                       ("x", "f4"), ("y", "f4")])
    # the steering of a step is applied from the tick the worker receives it, for frame_skip = 2 ticks
    np.testing.assert_array_equal(ticks["steer"][:2 * len(applied)][1::2], applied)


def test_steering_is_limited_with_noisy_actions_and_matches_the_slot_across_episodes():
    env = SpeedDreamsVecEnv(proto7(n_workers=3, max_episode_seconds=1.0))
    L = env.obs_layout
    rng = np.random.default_rng(3)
    try:
        env.reset()
        prev = np.zeros(3, np.float32)
        hist = np.zeros((3, 3), np.float32)
        ended = 0
        for _ in range(200):
            a = rng.normal(0, 1.5, (3, env.act_dim)).astype(np.float32)
            obs, _, term, trunc, info = env.step(a)
            slot = env._act[:, A["steer"]].copy()
            done = term | trunc
            np.testing.assert_array_equal(slot, np.clip(prev + np.clip(a[:, 0], -1, 1) * LIMIT, -1, 1))
            change = slot - prev
            assert (np.abs(change) <= LIMIT * (1 + 1e-6)).all()
            hist = np.concatenate([change[:, None] / LIMIT, hist[:, :2]], 1)
            for i in np.flatnonzero(done):
                np.testing.assert_array_equal(L.group(info["final_obs"][i], "steer_history"), hist[i])
                assert L.get(info["final_obs"][i], "prev_steer") == slot[i]
                ended += 1
            prev = np.where(done, 0, slot)
            hist = np.where(done[:, None], 0, hist)
            np.testing.assert_array_equal(L.get(obs, "prev_steer"), prev)
            np.testing.assert_array_equal(L.group(obs, "steer_history"), hist)
        assert ended >= 3
    finally:
        env.close()


def test_positive_steering_turns_the_car_left():
    env = SpeedDreamsVecEnv(proto7(n_workers=2, max_episode_seconds=30))
    try:
        env.reset()
        for _ in range(100):
            env.step(np.array([[0, 1.0, 0, 0]] * 2, np.float32))
        yaw0 = env._raw[:, F["yaw"]].copy()
        for _ in range(5):
            env.step(np.array([[1, 0.0, 0, 0], [-1, 0.0, 0, 0]], np.float32))
        d = np.angle(np.exp(1j * (env._raw[:, F["yaw"]] - yaw0)))
        assert d[0] > 0.005 and d[1] < -0.005
        assert env._raw[0, F["yaw_rate"]] > 0 > env._raw[1, F["yaw_rate"]]
    finally:
        env.close()


def test_paused_env_keeps_its_steering():
    env = SpeedDreamsVecEnv(proto7(n_workers=2, max_episode_seconds=20))
    L = env.obs_layout
    try:
        env.reset()
        a = np.array([[1, 0.5, 0, 0], [1, 0.5, 0, 0]], np.float32)
        env.step(a)
        obs, *_ = env.step(a, np.array([True, False]))
        assert L.get(obs, "prev_steer").tolist() == [2 * LIMIT, LIMIT]
        assert L.group(obs, "steer_history")[1].tolist() == [1.0, 0.0, 0.0]
    finally:
        env.close()


def test_absolute_steering_is_unchanged_without_a_limit():
    env = SpeedDreamsVecEnv(EnvConfig(n_workers=1, max_episode_seconds=20))
    L = env.obs_layout
    try:
        assert "steer_history" not in L and "edge_left_x" not in L
        env.reset()
        obs, *_ = env.step(np.array([[0.8, -0.4]], np.float32))
        assert np.isclose(env._act[0, A["steer"]], 0.8) and np.isclose(env._act[0, A["brake"]], 0.4)
        np.testing.assert_allclose(L.group(obs, "prev_action"), [[0.8, -0.4]])
    finally:
        env.close()
    with pytest.raises(ValueError):
        SpeedDreamsVecEnv(EnvConfig(n_workers=1, steer_history=3))


# ---------------------------------------------------------------- penalties


def _tracker(**kw):
    raw = np.zeros((1, N_FIELDS + N_RAYS), np.float32)
    raw[:, F["speed_x"]] = 20.0
    t = RewardTracker(1, RewardConfig(steer_rate=0.0, **kw), 0.04)
    t.reset(raw)
    return t, raw


def _penalties(t, raw, changes):
    change_i, rev_i = REWARD_COMPONENTS.index("steer_change"), REWARD_COMPONENTS.index("steer_reversal")
    out = []
    for c in changes:
        t.step(raw, np.array([c], np.float32))
        out.append((float(t.components[change_i][0]), float(t.components[rev_i][0]), bool(t.reversed[0])))
    return out


def test_change_and_reversal_penalties_match_hand_computed_sequences():
    t, raw = _tracker(steer_change=0.5, steer_reversal=0.03, steer_deadband=0.015)
    got = _penalties(t, raw, [0.0625, -0.0625, -0.03125, 0.03125, 0.0, 0.03125, -0.0625])
    want_change = [-0.03125, -0.03125, -0.015625, -0.015625, 0.0, -0.015625, -0.03125]
    want_rev = [0.0, -0.03, 0.0, -0.03, 0.0, 0.0, -0.03]
    np.testing.assert_allclose([g[0] for g in got], want_change)
    np.testing.assert_allclose([g[1] for g in got], want_rev)


def test_reversal_dead_band():
    db = 0.015625  # dyadic, so the comparison at the boundary is exact
    t, raw = _tracker(steer_change=0.0, steer_reversal=1.0, steer_deadband=db)
    # below it, on it, and just above it on either side: only the last pair counts
    changes = [db / 2, -db / 2, db, -db, db * 1.5, -db / 2, db * 1.5, -db * 1.5, db, -db * 1.5]
    want = [0, 0, 0, 0, 0, 0, 0, -1, 0, 0]
    np.testing.assert_allclose([g[1] for g in _penalties(t, raw, changes)], want)
    # a pause between two opposite moves is not a reversal
    t, raw = _tracker(steer_reversal=1.0, steer_deadband=db)
    assert [g[2] for g in _penalties(t, raw, [0.05, 0.0, -0.05])] == [False, False, False]


def test_progress_ignores_lap_sized_jumps():
    length = 4000.0
    raw = np.zeros((1, N_FIELDS + N_RAYS), np.float32)
    raw[:, F["speed_x"]] = 5.0
    raw[:, F["track_length"]] = length
    raw[:, F["dist_raced"]] = -3.0
    t = RewardTracker(1, RewardConfig(max_damage=1e9), 0.04)
    t.reset(raw)
    i = REWARD_COMPONENTS.index("progress")
    # forward over the line, back across it (the engine jumps a lap ahead), forward again (and back)
    path = [-1.0, 1.0, 0.5, length - 0.5, length - 1.0, length - 0.5, 0.5, 1.5]
    moved = [2.0, 2.0, -0.5, -1.0, -0.5, 0.5, 1.0, 1.0]
    for d, want in zip(path, moved):
        raw[:, F["dist_raced"]] = d
        t.step(raw)
        np.testing.assert_allclose(t.components[i][0], 0.1 * want, atol=1e-4)
    np.testing.assert_allclose(t.raced[0], 1.5, atol=1e-3)


def test_a_u_turn_across_the_start_line_earns_nothing_on_the_real_engine():
    """The exploit a smoke run found: turn around at full lock just before the line, and cross it backwards. The raw distance
    must jump a lap here, or this test proves nothing; the reward and the stats must not."""
    env = SpeedDreamsVecEnv(proto7(n_workers=1, race=race_for("allondaz"), max_episode_seconds=20,
                                   reward=RewardConfig(steer_rate=0.0)))
    try:
        env.reset()
        length = env._raw[0, F["track_length"]]
        progress = REWARD_COMPONENTS.index("progress")
        jumped = False
        for t in range(500):
            r = env._raw[0]
            turning = r[F["dist_raced"]] > -2  # just before the line (the grid is 4 m before it here)
            pedal = np.clip(0.5 * (5.0 - r[F["speed_x"]]), -1, 1)
            before = r[F["dist_raced"]]
            *_, term, trunc, info = env.step(np.array([[-1.0 if turning else 0.0, pedal, 0, 0]], np.float32))
            end = term[0] or trunc[0]
            after = (info["final_raw"][0] if end else env._raw[0])[F["dist_raced"]]
            jumped |= after - before > length / 2
            assert abs(env._rew.components[progress][0]) < 1.0  # 10 m of road in one step
            if end:
                assert jumped, "the car never crossed the line backwards"
                assert after > length / 2
                assert info["stats"]["dist_raced"][0] < 50 and info["stats"]["progress"][0] < 0.01
                assert info["stats"]["laps"][0] == 0
                return
        raise AssertionError("the episode didn't end")
    finally:
        env.close()


def test_reset_clears_the_previous_change_for_those_envs_only():
    raw = np.zeros((2, N_FIELDS + N_RAYS), np.float32)
    raw[:, F["speed_x"]] = 20.0
    t = RewardTracker(2, RewardConfig(steer_rate=0.0, steer_reversal=1.0), 0.04)
    t.reset(raw)
    t.step(raw, np.array([0.05, 0.05], np.float32))
    t.reset(raw, np.array([0]))
    t.step(raw, np.array([-0.05, -0.05], np.float32))
    np.testing.assert_allclose(t.components[REWARD_COMPONENTS.index("steer_reversal")], [0.0, -1.0])


def test_penalties_in_the_env_match_the_actions_it_was_given():
    cfg = proto7(n_workers=1, max_episode_seconds=1.0, reward=RewardConfig(steer_rate=0.0, steer_change=0.5, steer_reversal=0.03))
    env = SpeedDreamsVecEnv(cfg)
    try:
        env.reset()
        actions = np.array([1, -1, -1, 1, 0.1, -0.1, 1, -1] * 4, np.float32)[:25]  # 25 steps = 1 s
        change = reversal = 0.0
        prev = 0.0
        for a in actions:
            c = float(a) * LIMIT
            change -= 0.5 * abs(c)
            if c * prev < 0 and abs(c) > 0.015 and abs(prev) > 0.015:
                reversal -= 0.03
            prev = c
            *_, info = env.step(np.array([[a, 0.5, 0, 0]], np.float32))
        st = info["stats"]
        np.testing.assert_allclose(st["reward_steer_change"][0], change, rtol=1e-5)
        np.testing.assert_allclose(st["reward_steer_reversal"][0], reversal, rtol=1e-5)
        n_rev = -reversal / 0.03
        np.testing.assert_allclose(st["steer_reversal_frac"][0], n_rev / 25, rtol=1e-6)
        np.testing.assert_allclose(st["steer_requested"][0], np.mean(np.abs(actions)) * LIMIT, rtol=1e-5)
        np.testing.assert_allclose(st["steer_delta"][0], st["steer_requested"][0], rtol=1e-5)  # no lock reached
        total = sum(st[f"reward_{c}"][0] for c in REWARD_COMPONENTS)
        np.testing.assert_allclose(total, info["episode"]["r"][0], atol=1e-3)
    finally:
        env.close()


def test_requested_and_applied_changes_differ_at_the_lock():
    env = SpeedDreamsVecEnv(proto7(n_workers=1, max_episode_seconds=2.0, reward=RewardConfig(steer_rate=0.0, steer_change=0.5)))
    try:
        env.reset()
        for _ in range(50):
            *_, info = env.step(np.array([[1.0, 0.5, 0, 0]], np.float32))
        st = info["stats"]
        np.testing.assert_allclose(st["steer_requested"][0], LIMIT, rtol=1e-5)
        np.testing.assert_allclose(st["steer_delta"][0], 1.0 / 50, rtol=1e-5)  # 16 steps to the lock, then nothing
        np.testing.assert_allclose(st["steer_limit_frac"][0], 1.0)
        np.testing.assert_allclose(st["steer_lock_frac"][0], 35 / 50)
    finally:
        env.close()


# ---------------------------------------------------------------- determinism


def test_same_actions_give_identical_observations_across_workers_and_resets():
    env = SpeedDreamsVecEnv(proto7(n_workers=3, max_episode_seconds=60))
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1, 1, (300, env.act_dim)).astype(np.float32)
    actions[:, 1] = 0.4 + 0.1 * actions[:, 1]

    def rollout():
        out = [env.reset()[0].copy()]
        for a in actions:
            obs, *_ = env.step(np.tile(a, (3, 1)))
            out.append(obs.copy())
        return np.stack(out)

    try:
        a, b = rollout(), rollout()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, 0], a[:, 1])
        np.testing.assert_array_equal(a[:, 0], a[:, 2])
        assert np.isfinite(a).all()
    finally:
        env.close()


# ---------------------------------------------------------------- nothing depends on the bot's driving


def test_policy_inputs_never_read_the_bots_laps(monkeypatch):
    """All geometry reaches the observation without the bot's trace."""
    def boom(*_, **__):
        raise AssertionError("a policy input read the bot's laps")

    for cls in (TrackEdges, TrackProfile):
        cls.__dict__["for_race"].__func__.cache_clear()
    monkeypatch.setattr(reference, "load", boom)
    race = race_for("braga")
    LookaheadTable((race,))
    t = EdgeTable((race,), POINTS, 6.0, 60.0)
    assert t.table.shape[1] == 4


def test_scrambling_the_bots_line_changes_no_input_or_reward():
    def rollout(scramble):
        env = SpeedDreamsVecEnv(proto7(n_workers=2, races=(race_for("jarama"), race_for("braga")),
                                       max_episode_seconds=4.0))
        if scramble:
            env._line.table = np.random.default_rng(0).uniform(-50, 50, env._line.table.shape).astype(np.float32)
        try:
            out = [env.reset()[0]]
            rewards = []
            for t in range(160):
                obs, rew, *_ = env.step(np.tile(np.array([[np.sin(t / 9), 0.5, 0, 0]], np.float32), (2, 1)))
                out.append(obs)
                rewards.append(rew)
            return np.stack(out), np.stack(rewards)
        finally:
            env.close()

    (o1, r1), (o2, r2) = rollout(False), rollout(True)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(r1, r2)


def test_no_group_in_the_proto7_layout_comes_from_the_bot():
    layout = make_layout(("steer", "pedal", "shift_up", "shift_down"), 9, 0, True, 3, POINTS)
    assert "line" not in layout
    assert set(EDGE_GROUPS) <= set(layout.slices)
    for g in layout.groups:
        assert not re.search(r"\bbot\b", g.scale), g.name


# ---------------------------------------------------------------- scales, split, flags


def test_edge_scales_decode_back_to_metres():
    layout = make_layout(("steer", "pedal"), edge_points=2)
    obs = np.zeros(layout.dim, np.float32)
    obs[layout.slices["edge_left_x"]] = [1.0, 2.0]
    obs[layout.slices["edge_left_y"]] = [0.1, 0.2]
    obs[layout.slices["edge_right_y"]] = [-0.1, -0.2]
    left, right = decode_edges(layout, obs)
    np.testing.assert_allclose(left, [[EDGE_X_SCALE, 0.1 * EDGE_Y_SCALE], [2 * EDGE_X_SCALE, 0.2 * EDGE_Y_SCALE]])
    np.testing.assert_allclose(right[:, 1], [-0.1 * EDGE_Y_SCALE, -0.2 * EDGE_Y_SCALE])


def test_the_split_is_the_recorded_draw_and_partitions_the_tracks():
    known = {line.split(",")[0] for line in (workdirs.ROOT / "reference" / "tracks.csv").read_text().splitlines()[1:]}
    train, val, test = set(PROTO7_TRAIN), set(PROTO7_VALIDATION), set(PROTO7_TEST)
    assert (len(train), len(val), len(test)) == (29, 6, 6)
    assert train | val | test == known and not (train & val or train & test or val & test)
    assert set(PROTO7_EVAL_TRAIN) <= train
    pool = sorted((train | test) - set(PROTO7_EVAL_TRAIN))
    assert sorted(random.Random(20261103).sample(pool, 6)) == sorted(PROTO7_TEST)


def test_a_run_config_round_trips_through_its_logged_params():
    """scripts/eval_test_tracks.py rebuilds a run's EnvConfig from its logged params: all values become strings there."""
    cfg = proto7(n_workers=2, race="rlpilot-braga", races=(race_for("jarama"), race_for("braga")), run_dir=None,
                 reward=RewardConfig(steer_change=0.3, steer_reversal=0.02, gamma=0.99),
                 car_grip=(0.9, 1.1), max_episode_seconds=12.5, upshift_hold_gap=None)
    params = {f"env.{k}": str(v) for k, v in cfg.flat().items()} | {"timesteps": "1", "algo": "ppo"}
    assert EnvConfig.from_flat(params, "env.") == cfg
    assert EnvConfig.from_flat(params | {"env.steer_filter": "1.0", "env.pedal_filter": "1.0"}, "env.") == cfg  # older runs


@pytest.mark.parametrize("key,value", [("env.steer_filter", "0.25"), ("env.something_new", "1")])
def test_a_config_this_code_cannot_reproduce_is_refused(key, value):
    params = {f"env.{k}": str(v) for k, v in proto7(n_workers=2).flat().items()}
    with pytest.raises(ValueError, match="reproduce"):
        EnvConfig.from_flat(params | {key: value}, "env.")


@pytest.mark.parametrize("flags", [["--heldout-tracks", "alicante"], ["--tracks", "alicante"], ["--eval-train-tracks", "spring"]])
def test_a_split_refuses_explicit_track_flags(flags):
    """They would otherwise be silently replaced. Refused before anything starts."""
    script = workdirs.ROOT / "scripts" / "train_sb3.py"
    r = subprocess.run([sys.executable, str(script), "--split", "proto7", *flags], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "can't be combined" in r.stderr
