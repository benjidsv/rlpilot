"""Run report: where a finished run fails, and since when. Everything comes from the run's saved checkpoints, so it also
works on old runs and training needs no extra logging.

    sweep: re-drive some checkpoints on all the run's training and validation tracks (never its test tracks)
    render: from those results, an HTML page with a tracks x steps heatmap, then crash maps, tables and GIFs per track

Results are cached in the report's data/ directory, so re-rendering takes no simulation. See scripts/report.py.
"""

from __future__ import annotations

import ast
import html
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import plots, reference, workdirs
from .config import EnvConfig
from .reference import Reference, lap_slice
from .track import race_for, reference_for, track_of
from .vec_env import SpeedDreamsVecEnv
from .evaluate import eval_config, eval_time_limits, rollout

MAX_WORKERS = 70  # 140 time out at startup
CLUSTER_GAP_M = 100.0  # crashes closer than this along the track are the same spot
ENTRY_M = 100.0  # the "entry speed" of a crash spot is taken this far before it
FINISHED = "lap_done"


def run_tracks(params: dict[str, str]) -> dict[str, str]:
    """race -> split, for the tracks a report may drive on: the run's training and validation tracks. A test split is
    neither, so it can't be spent here."""
    train = ast.literal_eval(params.get("env.races") or "()") or (params["env.race"],)
    held = [t for t in params.get("eval.heldout_tracks", "").split(",") if t]
    return {r: "train" for r in train} | {r: "validation" for r in map(race_for, held) if r not in train}


def pick_checkpoints(steps: list[int], n: int) -> list[int]:
    """n evenly spaced steps, always with the last one."""
    steps = sorted(steps)
    if n >= len(steps):
        return steps
    return [steps[i] for i in sorted({round(x) for x in np.linspace(len(steps) / n - 1, len(steps) - 1, n)})]


def crash_index(trace: dict[str, np.ndarray]) -> int:
    """The step where an episode's end began: where the car left the road for the last time if it never came back,
    else (a spin or a stop on the road) its furthest point."""
    off = trace["wheels_off"] > 0
    if off[-1]:
        back_on = np.flatnonzero(~off)
        return int(back_on[-1]) + 1 if len(back_on) else 0
    return int(np.argmax(trace["dist_raced"]))


def summarize_episode(trace: dict[str, np.ndarray], stats: dict, laps: int) -> dict:
    out = {"end": stats["end"], "driven": 1.0 if stats["end"] == FINISHED else min(stats["progress"] / laps, 1.0),
           "finished_strict": stats["end"] == FINISHED and stats.get("strict_end") is None}
    if stats["end"] != FINISHED:
        i = crash_index(trace)
        out |= {"crash_i": i, "crash_m": float(trace["dist_from_start"][i]), "lap": max(int(trace["laps"][i]), 1)}
    return out


def cluster_positions(positions: list[float], length: float, gap: float = CLUSTER_GAP_M) -> list[tuple[float, list[int]]]:
    """Groups positions on a closed track. Returns (median position, indices), ordered along the lap."""
    if not positions:
        return []
    order = np.argsort(positions)
    d = np.asarray(positions, float)[order]
    gaps = np.diff(np.append(d, d[0] + length))
    if gaps.max() <= gap:
        return [(float(np.median(d)), order.tolist())]
    first = int(np.argmax(gaps)) + 1  # start after the widest gap, so a spot across the line stays whole
    d, order = np.roll(d, -first), np.roll(order, -first)
    d[d < d[0]] += length  # unrolled across the line
    cuts = np.flatnonzero(np.diff(d) > gap) + 1
    groups = [(float(np.median(d[c])) % length, order[c].tolist()) for c in np.split(np.arange(len(d)), cuts)]
    return sorted(groups, key=lambda g: g[0])


# ---- sweep ---------------------------------------------------------------------------------------------------------

def _chunks(races: list[str], episodes: int) -> list[tuple[str, ...]]:
    per = max(MAX_WORKERS // episodes, 1)
    return [tuple(races[i:i + per]) for i in range(0, len(races), per)]


def sweep(params: dict[str, str], jobs: list[tuple[str, Path]], races: list[str], refs: dict[str, Reference], episodes: int,
          laps: int, on_chunk, keep_traces=None) -> None:
    """Drives each job (key, checkpoint dir) on every race, with `episodes` episodes per race, the first without launch
    noise. Calls on_chunk(key, race, summaries) as results come. keep_traces(race, traces) gets the raw traces."""
    from stable_baselines3 import PPO

    for chunk in _chunks(races, episodes):
        cfg = eval_config(EnvConfig.from_flat(params, "env."), chunk, episodes, laps)
        limits = eval_time_limits([refs[r].best_lap for r in chunk], cfg.n_workers, laps)
        env = SpeedDreamsVecEnv(cfg)
        try:
            for key, path in jobs:
                model = PPO.load(Path(path) / "model.zip", device="cpu")
                traces, _, stats = rollout(lambda o: model.predict(o, deterministic=True)[0], env,
                                           clean=range(len(chunk)), time_limits=limits)
                for t, race in enumerate(chunk):
                    idx = range(t, cfg.n_workers, len(chunk))
                    on_chunk(key, race, [summarize_episode(traces[i], stats[i], laps) for i in idx])
                    if keep_traces:
                        keep_traces(race, [traces[i] for i in idx])
        finally:
            env.close()


# ---- render --------------------------------------------------------------------------------------------------------

def _crash_rows(race: str, episodes: list[dict], traces: list[dict], ref: Reference) -> list[dict]:
    """The crash spots of one track at best/, most frequent first."""
    failed = [i for i, e in enumerate(episodes) if "crash_m" in e]
    rows = []
    for at, members in cluster_positions([episodes[i]["crash_m"] for i in failed], ref.track_length):
        eps = [failed[m] for m in members]
        rep = eps[0] if 0 not in eps else 0  # show the episode without launch noise when it's there
        tr, i = traces[rep], episodes[rep]["crash_i"]
        lap = episodes[rep]["lap"]
        entry = max(int(np.searchsorted(np.maximum.accumulate(tr["dist_raced"]), tr["dist_raced"][i] - ENTRY_M)) - 1, 0)
        bot = lap_slice(ref.trace, min(lap, 3))
        d = (tr["dist_from_start"][i] - ENTRY_M) % ref.track_length
        rows.append({"at": at, "n": len(eps), "total": len(episodes), "rep": rep, "i": i,
                     "laps": sorted({episodes[e]["lap"] for e in eps}),
                     "ends": Counter(episodes[e]["end"] for e in eps).most_common(),
                     "entry_kmh": float(tr["speed_x"][entry] * 3.6),
                     "bot_kmh": float(np.interp(d, bot["dist_from_start"], bot["speed_x"]) * 3.6)})
    return sorted(rows, key=lambda r: (-r["n"], r["at"]))


def _map_trace(traces: list[dict], episodes: list[dict]) -> dict:
    """The furthest episode's first lap, for the track map."""
    best = int(np.argmax([e["driven"] for e in episodes]))
    lap = lap_slice(traces[best], 1)
    return lap if len(lap["time"]) > 1 else traces[best]


BINS = (0.2, 0.4, 0.6, 0.8, 1.0)  # below each: a color step. 1.0 itself (every episode finished) gets the last step


def _bin(v: float) -> int:
    return int(np.searchsorted(BINS, v, side="right"))


def _pct(v: float) -> str:
    return f"{100 * v:.0f}%"


def _cell(track: str, label: str, episodes: list[dict]) -> str:
    v = float(np.mean([e["driven"] for e in episodes]))
    crashes = sorted({round(e["crash_m"]) for e in episodes if "crash_m" in e})
    where = "finished every episode" if not crashes else "off at " + ", ".join(f"{m:,} m" for m in crashes[:4]) + \
        (f" and {len(crashes) - 4} more" if len(crashes) > 4 else "")
    tip = html.escape(f"{track} · {label} · {_pct(v)} driven · {where}", quote=True)
    return f'<td class="c c{_bin(v)}" data-tip="{tip}"></td>'


def _heatmap(tracks: list[dict], steps: list[int], sweeps: dict) -> str:
    head = "".join(f'<th class="s">{s / 1e6:.0f}M</th>' if i % 5 == 0 or i == len(steps) - 1 else '<th class="s"></th>'
                   for i, s in enumerate(steps))
    rows = [f'<tr><th class="name"></th>{head}<th class="gap"></th><th class="b">best/</th></tr>']
    for split in ("train", "validation"):
        group = [t for t in tracks if t["split"] == split]
        if not group:
            continue
        means = [np.mean([np.mean([e["driven"] for e in sweeps[str(s)][t["race"]]]) for t in group]) for s in steps]
        best = np.mean([t["driven"] for t in group])
        mean_cells = "".join(f'<td class="c c{_bin(float(m))}" data-tip="{split} mean · {s / 1e6:.0f}M · {_pct(float(m))} driven"></td>'
                             for m, s in zip(means, steps))
        rows.append(f'<tr class="group"><th class="name">{split} mean</th>{mean_cells}<td class="gap"></td>'
                    f'<td class="b">{_pct(float(best))}</td></tr>')
        for t in group:
            cells = "".join(_cell(t["track"], f"{s / 1e6:.0f}M", sweeps[str(s)][t["race"]]) for s in steps)
            rows.append(f'<tr class="row" data-track="{html.escape(t["track"])}"><th class="name">'
                        f'<a href="#{html.escape(t["track"])}">{html.escape(t["track"])}</a></th>{cells}<td class="gap"></td>'
                        f'<td class="b">{_pct(t["driven"])}</td></tr>')
    return "<table class=\"heat\">" + "".join(rows) + "</table>"


def _track_section(t: dict) -> str:
    name = html.escape(t["track"])
    summary = f'{name} <span class="muted">{t["split"]} · {_pct(t["driven"])} driven · {t["finished"]}/{t["episodes"]} finished</span>'
    if not t["crashes"]:
        return (f'<details class="track" id="{name}"><summary>{summary}</summary><div class="body">'
                f'<img loading="lazy" src="{t["map"]}" alt="Track map of {name}"></div></details>')
    rows = "".join(
        f'<tr><td class="num">{i + 1}</td><td class="num">{c["at"]:,.0f} m</td><td class="num">{c["n"]}/{c["total"]}</td>'
        f'<td>{", ".join(str(x) for x in c["laps"])}</td><td>{", ".join(html.escape(e) for e, _ in c["ends"])}</td>'
        f'<td class="num">{c["entry_kmh"]:.0f}</td><td class="num">{c["bot_kmh"]:.0f}</td>'
        f'<td class="num">{c["entry_kmh"] - c["bot_kmh"]:+.0f}</td></tr>'
        for i, c in enumerate(t["crashes"]))
    gifs = "".join(f'<figure><img loading="lazy" src="{c["gif"]}" alt="Crash {i + 1} on {name}, top-down, against the bot">'
                   f'<figcaption>#{i + 1}, {c["at"]:,.0f} m</figcaption></figure>' for i, c in enumerate(t["crashes"]))
    return (f'<details class="track" id="{name}" open><summary>{summary}</summary><div class="body">'
            f'<img class="map" loading="lazy" src="{t["map"]}" alt="Track map of {name} with its crash spots">'
            f'<div class="side"><table class="crashes"><tr><th>#</th><th>Off at</th><th>Episodes</th><th>Lap</th><th>Ended by</th>'
            f'<th>Speed {ENTRY_M:.0f} m before</th><th>Bot</th><th>Diff</th></tr>{rows}</table>'
            f'<div class="gifs">{gifs}</div></div></div></details>')


CSS = """
:root { color-scheme: light dark; --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --tip:#0b0b0b; --tipink:#fcfcfb;
  --c0:#cde2fb; --c1:#9ec5f4; --c2:#6da7ec; --c3:#3987e5; --c4:#256abf; --c5:#0d366b; --hover:#0b0b0b; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a18; --ink:#f3f2ee; --ink2:#b5b4ae; --line:#383835; --tip:#f3f2ee; --tipink:#1a1a18;
  --c0:#104281; --c1:#1c5cab; --c2:#2a78d6; --c3:#5598e7; --c4:#86b6ef; --c5:#cde2fb; --hover:#f3f2ee; } }
* { box-sizing:border-box; }
body { margin:0 auto; max-width:1280px; padding:32px 24px 80px; background:var(--bg); color:var(--ink);
  font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
h1 { font-size:24px; margin:0; } h2 { font-size:17px; margin:40px 0 4px; }
.muted, .note { color:var(--ink2); font-weight:400; } .note { margin:0 0 14px; max-width:80ch; }
.stats { display:flex; gap:36px; margin:16px 0 0; flex-wrap:wrap; } .stats b { display:block; font-size:24px; font-variant-numeric:tabular-nums; }
.stats span { color:var(--ink2); font-size:12px; }
.heatwrap { overflow-x:auto; }
table.heat { border-collapse:separate; border-spacing:2px; }
.heat th, .heat td { padding:0; font-weight:400; }
.heat .name { text-align:left; padding-right:10px; white-space:nowrap; font-size:13px; position:sticky; left:0; background:var(--bg); }
.heat .name a { color:inherit; text-decoration:none; } .heat .row:hover .name a { text-decoration:underline; }
.heat .s { font-size:10px; color:var(--ink2); text-align:left; white-space:nowrap; width:16px; max-width:16px; overflow:visible; height:18px; }
.heat .c { width:16px; height:18px; border-radius:2px; cursor:pointer; }
.heat .c:hover { outline:2px solid var(--hover); outline-offset:-1px; }
.heat .gap { width:10px; }
.heat .b { text-align:right; font-size:12px; font-variant-numeric:tabular-nums; color:var(--ink2); padding-left:4px; min-width:42px; }
.heat .group .name { font-weight:600; padding-top:10px; } .heat .group .b { color:var(--ink); font-weight:600; padding-top:10px; }
.c0{background:var(--c0)} .c1{background:var(--c1)} .c2{background:var(--c2)} .c3{background:var(--c3)} .c4{background:var(--c4)} .c5{background:var(--c5)}
.legend { display:flex; gap:2px; margin:14px 0 0; font-size:11px; color:var(--ink2); }
.legend .k { width:72px; } .legend i { height:12px; display:block; border-radius:2px; margin-bottom:3px; }
#tip { position:fixed; pointer-events:none; background:var(--tip); color:var(--tipink); padding:5px 9px; border-radius:4px; font-size:12px;
  display:none; max-width:440px; z-index:5; }
details.track { border-top:1px solid var(--line); padding:12px 0; scroll-margin-top:12px; }
details.track summary { cursor:pointer; font-weight:600; font-size:15px; }
.body { display:flex; gap:24px; flex-wrap:wrap; align-items:flex-start; margin-top:12px; }
.body img { max-width:560px; width:100%; border-radius:6px; } .side { flex:1; min-width:380px; }
table.crashes { border-collapse:collapse; width:100%; font-size:13px; }
.crashes th { text-align:left; font-weight:600; color:var(--ink2); border-bottom:1px solid var(--line); padding:4px 8px; font-size:12px; }
.crashes td { padding:5px 8px; border-bottom:1px solid var(--line); } .num { font-variant-numeric:tabular-nums; }
.gifs { display:flex; gap:16px; flex-wrap:wrap; margin-top:16px; }
figure { margin:0; } figure img { width:440px; max-width:100%; display:block; border-radius:6px; } figcaption { font-size:12px; color:var(--ink2); margin-top:3px; }
"""

JS = """
const tip = document.getElementById('tip');
document.querySelector('.heat').addEventListener('mousemove', e => {
  const t = e.target.dataset && e.target.dataset.tip;
  tip.style.display = t ? 'block' : 'none';
  if (t) { tip.textContent = t; tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8) + 'px'; tip.style.top = (e.clientY + 16) + 'px'; }
});
document.querySelector('.heat').addEventListener('mouseleave', () => tip.style.display = 'none');
function go(track) { const d = document.getElementById(track); if (d) { d.open = true; d.scrollIntoView(); } }
document.querySelectorAll('.heat .row').forEach(r => r.addEventListener('click', () => go(r.dataset.track)));
"""


def render(out: Path, run_name: str, run_id: str, best_step: str | None, tracks_by_race: dict[str, str], results: dict,
           refs: dict[str, Reference], laps: int) -> Path:
    """Writes out/index.html and its images from the sweep results."""
    (out / "img").mkdir(parents=True, exist_ok=True)
    steps = sorted(int(k) for k in results["sweeps"])
    best = results["best"]
    tracks = []
    for race, split in tracks_by_race.items():
        name = track_of(race)
        with np.load(out / "data" / f"best_{race}.npz") as z:
            traces = [{k.split("_", 1)[1]: z[k] for k in z.files if k.startswith(f"ep{i}_")} for i in range(len(best[race]))]
        rows = _crash_rows(race, best[race], traces, refs[race])
        marks = []
        for n, c in enumerate(rows):
            tr = traces[c["rep"]]
            marks.append((float(tr["x"][c["i"]]), float(tr["y"][c["i"]]), f"#{n + 1}  {c['at']:,.0f} m"))
            c["gif"] = f"img/{name}-{n + 1}.gif"
            if not (out / c["gif"]).exists():
                plots.crash_gif(refs[race], tr, c["i"], out / c["gif"], f"{name}, #{n + 1}, {c['at']:,.0f} m")
        map_path = f"img/{name}.png"
        if not (out / map_path).exists():
            fig = plots.track_map(refs[race], _map_trace(traces, best[race]), f"{name}, furthest lap at best/", marks=marks)
            fig.savefig(out / map_path)
            plt.close(fig)
        tracks.append({"race": race, "track": name, "split": split, "crashes": rows, "map": map_path, "episodes": len(best[race]),
                       "finished": sum(e["end"] == FINISHED for e in best[race]),
                       # results cached before this field existed came from the strict rules
                       "finished_strict": sum(e.get("finished_strict", e["end"] == FINISHED) for e in best[race]),
                       "driven": float(np.mean([e["driven"] for e in best[race]]))})
    tracks.sort(key=lambda t: (t["split"] != "train", t["driven"], t["track"]))

    def finished(split):
        group = [t for t in tracks if t["split"] == split]
        return sum(t["finished"] for t in group) / max(sum(t["episodes"] for t in group), 1), len(group)

    stats = "".join(f'<div><b>{_pct(finished(s)[0])}</b><span>{s} episodes finished at best/ ({finished(s)[1]} tracks)</span></div>'
                    for s in ("train", "validation") if finished(s)[1])
    if any(t["finished_strict"] != t["finished"] for t in tracks):
        for s in ("train", "validation"):
            group = [t for t in tracks if t["split"] == s]
            share = sum(t["finished_strict"] for t in group) / max(sum(t["episodes"] for t in group), 1)
            stats += f'<div><b>{_pct(share)}</b><span>{s} under the 1 s rules of the runs up to proto-8</span></div>'
    n_spots = sum(len(t["crashes"]) for t in tracks)
    stats += f'<div><b>{n_spots}</b><span>crash spots at best/</span></div>'
    legend = "".join(f'<span class="k"><i class="c{i}"></i>{label}</span>'
                     for i, label in enumerate(("under 20%", "20-40%", "40-60%", "60-80%", "over 80%", "all finished")))
    episodes = len(results["sweeps"][str(steps[0])][next(iter(tracks_by_race))])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(run_name)} report</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>{CSS}</style></head><body>
<h1>{html.escape(run_name)}</h1><div class="muted">run {run_id[:8]} · best/ from step {int(best_step or 0) / 1e6:.0f}M</div>
<div class="stats">{stats}</div>
<h2>How far it gets</h2>
<p class="note">Share of the {laps} laps driven, per checkpoint and track, worst tracks at best/ on top. {episodes} episodes per cell,
{results["best_episodes"]} for best/. Only the darkest color means every episode finished. Click a track for its crash spots.</p>
<div class="heatwrap">{_heatmap(tracks, steps, results["sweeps"])}</div>
<div class="legend">{legend}</div>
<h2>Where it crashes at best/</h2>
<p class="note">"Off at" is where the car left the road for good, along the lap. GIFs run from 5 s before that spot, in real time, with the bot
starting from the same point. Tracks in the same order as the heatmap.</p>
{"".join(_track_section(t) for t in tracks)}
<div id="tip"></div><script>{JS}</script></body></html>"""
    index = out / "index.html"
    index.write_text(page)
    return index


def build(run, client, out: Path, n_checkpoints: int = 38, episodes: int = 2, best_episodes: int = 8) -> Path:
    """Runs what's missing from out/data/, then renders. `run` is an MLflow run."""
    os.chdir(workdirs.ROOT)  # artifact URIs are relative to it
    run_id, params = run.info.run_id, run.data.params
    laps = int(params.get("eval.laps", 2))
    tracks = run_tracks(params)
    races = list(tracks)
    refs = {r: reference.load(reference_for(r), laps=3) for r in races}
    (out / "data").mkdir(parents=True, exist_ok=True)
    cache = out / "data" / "results.json"
    results = json.loads(cache.read_text()) if cache.exists() else {"sweeps": {}, "best": {}, "best_episodes": best_episodes}

    def save():
        cache.write_text(json.dumps(results))

    all_steps = [int(Path(a.path).name) for a in client.list_artifacts(run_id, "checkpoints")]
    todo = [s for s in pick_checkpoints(all_steps, n_checkpoints)
            if set(results["sweeps"].get(str(s), {})) != set(races)]
    if todo:
        print(f"sweeping {len(todo)} checkpoints on {len(races)} tracks, {episodes} episodes each")
        jobs = [(str(s), Path(client.download_artifacts(run_id, f"checkpoints/{s:010d}"))) for s in todo]

        def add(key, race, summaries):
            results["sweeps"].setdefault(key, {})[race] = summaries
            save()
            print(f"\r  {int(key) / 1e6:.0f}M {track_of(race)}" + " " * 20, end="", flush=True)

        sweep(params, jobs, races, refs, episodes, laps, add)
        print()
    if set(results["best"]) != set(races):
        print(f"driving best/ on {len(races)} tracks, {best_episodes} episodes each")
        results["best_episodes"] = best_episodes

        def add_best(_, race, summaries):
            results["best"][race] = summaries

        def keep(race, traces):
            np.savez_compressed(out / "data" / f"best_{race}.npz",
                                **{f"ep{i}_{k}": v for i, tr in enumerate(traces) for k, v in tr.items()})

        sweep(params, [("best", Path(client.download_artifacts(run_id, "best")))], races, refs, best_episodes, laps, add_best, keep)
        save()
    return render(out, run.info.run_name, run_id, run.data.tags.get("best_step"), tracks, results, refs, laps)

