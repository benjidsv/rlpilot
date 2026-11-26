"""Corners of a track, found from the road's curvature, with features from the track table: shape, the bot's speed
through it (braking zone or flat out), elevation, banking, runoff, and the corner before it. Used by the failure map
(scripts/analysis/corner_map.py) and the corner scenarios (scripts/setup/make_corners.py).

A corner is a stretch where the centreline's curvature (smoothed over ~SMOOTH m) stays above 1 / MAX_RADIUS on one
side, turning at least MIN_TURN in total. Stretches of the same direction closer than MERGE_GAP are one corner; a change
of direction starts a new one (a chicane is two corners, the second with a 0 m gap to the first), and so does a
valley in the curvature below half the peaks around it (a long complex becomes its bends).

A braking zone: the bot loses at least BRAKING m/s from its braking point (its last speed maximum before its slowest
point in the corner, searched back at most APPROACH m and never past the previous corner's slowest point) to its
slowest point. The corner's `window`, where its losses of control are counted, runs from the braking point (at least
LEAD m before the corner, never before the previous corner's slowest point) to EXIT m after it.
"""

from __future__ import annotations

import numpy as np

from .tracks import C, Track

SMOOTH = 8.0         # m, sd of the gaussian smoothing the curvature
MAX_RADIUS = 500.0   # m: tighter than this is a corner
MIN_TURN = 10.0      # deg of total turn
MERGE_GAP = 15.0     # m between two stretches of the same direction that make one corner
MIN_PIECE = 20.0     # m: a long stretch is split at a curvature valley only into pieces at least this long
BRAKING = 5.0        # m/s the bot loses from its approach speed to its slowest point: a braking zone
APPROACH = 300.0     # m before the corner where the bot's braking point is looked for
LEAD = 50.0          # m before a flat-out corner where its window begins
EXIT = 50.0          # m after the corner where its window ends
# A long straight (straight_map): no bend tighter than STRAIGHT_RADIUS from STRAIGHT_BEHIND m behind to STRAIGHT_AHEAD
# m ahead. ~16% of a training track's length (median; 0-50% by track): the corner approaches, where a car sets up its
# line, are never part of one (2026-11-25).
STRAIGHT_RADIUS = 800.0
STRAIGHT_BEHIND = 30.0
STRAIGHT_AHEAD = 150.0


def _circ_smooth(x: np.ndarray, sd_cells: float) -> np.ndarray:
    n = len(x)
    k = np.arange(-int(4 * sd_cells), int(4 * sd_cells) + 1)
    w = np.exp(-0.5 * (k / sd_cells) ** 2)
    w /= w.sum()
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(np.roll(np.pad(w, (0, n - len(w))), -len(k) // 2))))


def profiles(track: Track) -> dict[str, np.ndarray]:
    """Per grid cell (cells, closed loop): signed curvature (1/m, + = left), centreline height, banking (rad, + = the
    left edge higher), half width, and the table's bot columns."""
    t = track.table[:-1].astype(np.float64)
    ds = track.spacing
    mid = 0.5 * (t[:, [C["left_x"], C["left_y"]]] + t[:, [C["right_x"], C["right_y"]]])
    d = np.roll(mid, -1, axis=0) - mid
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    dh = np.diff(np.concatenate([heading, heading[:1] + (heading[-1] - heading[0]) +
                                 (heading[-1] - heading[-2])]))  # closed: the last step wraps to the first
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    kappa = _circ_smooth(dh / ds, SMOOTH / ds)
    width = np.hypot(*(t[:, [C["left_x"], C["left_y"]]] - t[:, [C["right_x"], C["right_y"]]]).T)
    return {"kappa": kappa, "z": 0.5 * (t[:, C["left_z"]] + t[:, C["right_z"]]),
            "bank": np.arctan2(t[:, C["left_z"]] - t[:, C["right_z"]], np.maximum(width, 1.0)),
            "half_width": 0.5 * width, "bot_speed": t[:, C["bot_speed"]], "bot_steer": t[:, C["bot_steer"]],
            "side_left": t[:, C["side_left"]], "side_right": t[:, C["side_right"]],
            "grass_left": t[:, C["grass_left"]], "grass_right": t[:, C["grass_right"]],
            "loose_left": t[:, C["loose_left"]], "loose_right": t[:, C["loose_right"]]}


def straight_map(track: Track) -> np.ndarray:
    """[cells + 1] float32, per grid cell like Track.table (the last row is the start line again): 1 on a long
    straight (see STRAIGHT_RADIUS), else 0."""
    k = np.abs(profiles(track)["kappa"])
    n, ds = len(k), track.spacing
    window = np.arange(-int(STRAIGHT_BEHIND / ds), int(STRAIGHT_AHEAD / ds) + 1)
    tightest = np.zeros(n)
    for w in window:  # max over the window, without an [n, window] array
        np.maximum(tightest, np.roll(k, -w), out=tightest)
    straight = (tightest < 1.0 / STRAIGHT_RADIUS).astype(np.float32)
    return np.concatenate([straight, straight[:1]])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) cell ranges where mask is true, on a closed loop (a run across the start line wraps: end > n)."""
    n = len(mask)
    if mask.all():
        return [(0, n)]
    shift = int(np.flatnonzero(~mask)[0])  # start the scan at a false cell, so no run is cut by the start line
    m = np.roll(mask, -shift)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
    return [(a + shift, b + shift) for a, b in zip(edges[::2], edges[1::2])]


def find(track: Track) -> list[dict]:
    """The corners of a track, in lap order, each a dict of features (distances in m from the start line)."""
    p = profiles(track)
    n, ds, L = len(p["kappa"]), track.spacing, track.length
    k, v, z = p["kappa"], p["bot_speed"], p["z"]
    stretches = []
    for sign in (1, -1):
        for a, b in _runs(sign * k > 1 / MAX_RADIUS):
            stretches.append([a, b, sign])
    stretches.sort()
    merged: list[list[int]] = []
    for a, b, sign in stretches:
        if merged and merged[-1][2] == sign and (a - merged[-1][1]) * ds <= MERGE_GAP:
            merged[-1][1] = b
        else:
            merged.append([a, b, sign])
    if len(merged) > 1 and merged[0][2] == merged[-1][2] and (merged[0][0] + n - merged[-1][1]) * ds <= MERGE_GAP:
        merged[0][0] = merged[-1][0] - n  # across the start line
        merged.pop()
    pieces = []  # (cells as unwrapped indices, sign)
    for a, b, sign in merged:
        cells_ab = np.arange(a, b)
        for cells in _split(cells_ab, np.abs(k[cells_ab % n]), int(MIN_PIECE / ds)):
            if abs(np.degrees(k[cells % n].sum() * ds)) >= MIN_TURN:
                pieces.append((cells, sign))
    pieces.sort(key=lambda c: c[0][0])
    corners = []
    for i, (cells, sign) in enumerate(pieces):
        wrapped = cells % n
        a, b = int(cells[0]), int(cells[-1]) + 1
        apex = int(cells[np.argmax(np.abs(k[wrapped]))])
        slow = int(cells[np.argmin(v[wrapped])])               # the bot's slowest point in the corner
        # the braking point: the bot's last speed maximum before its slowest point, searched back to the previous
        # corner's slowest point (a chain's second corner brakes from the first one's exit) or APPROACH m
        prev_slow = corners[-1]["_slow"] if corners else None
        lo = a - int(APPROACH / ds)
        if prev_slow is not None:
            lo = max(lo, prev_slow + 1)
        if len(pieces) > 1 and i == 0:  # the last corner of the lap comes before the first one
            last_cells = pieces[-1][0]
            last_slow = int(last_cells[np.argmin(v[last_cells % n])]) - n
            lo = max(lo, last_slow + 1)
        before = np.arange(min(lo, slow), slow + 1)
        brake = int(before[len(before) - 1 - int(np.argmax(v[before % n][::-1]))])
        v_entry, v_min = float(v[brake % n]), float(v[slow % n])
        braking = v_entry - v_min >= BRAKING
        brake_dist = (slow - brake) * ds if braking else 0.0
        win0 = min(brake if braking else a, a - int(LEAD / ds))
        if prev_slow is not None:
            win0 = max(win0, prev_slow + 1)
        start_s, end_s = a * ds, b * ds
        out = "right" if sign > 0 else "left"                  # the outside of a left turn is on the right
        ex = np.arange(apex, b + int(EXIT / ds)) % n           # apex to exit: where a car runs wide
        hz = lambda c: float(z[int(round(c)) % n])
        lean = int(round(50 / ds))
        corners.append({
            "track": track.name, "start": round(start_s % L, 1), "end": round(end_s % L, 1),
            "apex": round((apex % n) * ds, 1), "slowest": round((slow % n) * ds, 1), "length": round((b - a) * ds, 1),
            "window_start": round((win0 * ds) % L, 1), "window_end": round((end_s + EXIT) % L, 1),
            "window_length": round(end_s + EXIT - win0 * ds, 1),
            "direction": "left" if sign > 0 else "right",
            "turn_deg": round(abs(float(np.degrees(k[wrapped].sum() * ds))), 1),
            "min_radius": round(float(1 / np.abs(k[apex % n])), 1),
            "half_width": round(float(p["half_width"][apex % n]), 2),
            "braking": bool(braking), "bot_entry_speed": round(v_entry, 1), "bot_min_speed": round(v_min, 1),
            "bot_drop": round(v_entry - v_min, 1), "brake_distance": round(brake_dist, 1),
            "bot_decel": round((v_entry ** 2 - v_min ** 2) / (2 * brake_dist), 2) if brake_dist > 0 else 0.0,
            "bot_steer_max": round(float(np.abs(p["bot_steer"][wrapped]).max()), 3),
            # elevation: the approach's slope (+ = uphill), the height gained from the window's start to the apex, and
            # the vertical curvature at the apex over +-50 m (+ = a dip / compression, - = a crest)
            "approach_slope": round((hz(a) - hz(a - 100 / ds)) / 100, 4),
            "height_change": round(hz(apex) - hz(win0), 2),
            "vertical_curv": round((hz(apex - lean) + hz(apex + lean) - 2 * hz(apex)) / (lean * ds) ** 2, 5),
            "bank_deg": round(-float(np.degrees(p["bank"][apex % n])) * sign, 2),  # + = banked into the turn
            "runoff_out": round(float(np.min(p[f"side_{out}"][ex])), 1),           # narrowest runoff outside
            "grass_out": round(float(np.mean(p[f"grass_{out}"][ex])), 2),
            "loose_out": round(float(np.mean(p[f"loose_{out}"][ex])), 2),
            "_slow": slow, "_a": a, "_b": b,
        })
    for i, c in enumerate(corners):  # the corner before: how far, which way, how slow
        prev = corners[i - 1] if len(corners) > 1 else None
        c["gap_before"] = round(((c["_a"] - prev["_b"]) % n) * ds, 1) if prev else float("nan")
        c["prev_same_dir"] = bool(prev and prev["direction"] == c["direction"])
        c["prev_bot_min_speed"] = prev["bot_min_speed"] if prev else float("nan")
        c["prev_turn_deg"] = prev["turn_deg"] if prev else float("nan")
        c["id"] = f"{track.name}:{int(c['apex'])}"
    for c in corners:
        for key in ("_slow", "_a", "_b"):
            del c[key]
    return corners


def _split(cells: np.ndarray, a: np.ndarray, min_cells: int) -> list[np.ndarray]:
    """Split a stretch (cells, |curvature| a) at its deepest valley while that valley is below half the peaks on both
    sides: a long complex of several bends becomes several corners."""
    if len(cells) <= 2 * min_cells:
        return [cells]
    inner = np.arange(min_cells, len(cells) - min_cells)
    v = inner[np.argmin(a[inner])]
    if a[v] < 0.5 * min(a[:v].max(), a[v:].max()):
        return _split(cells[:v], a[:v], min_cells) + _split(cells[v:], a[v:], min_cells)
    return [cells]
