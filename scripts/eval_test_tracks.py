"""The test split: evaluate a finished run's best/ on the test tracks, which never select anything. Run it once, on the
final best/ of a run: looking at these numbers and then changing the run makes them validation numbers.

    python scripts/eval_test_tracks.py <run id> [--episodes 8] [--laps 2]

The env is rebuilt from the run's logged env.* params, set up like training evals. The results are logged to the run as
test/*, and a second run on the same one is refused.
"""

import argparse
import ast
import os
from pathlib import Path

import mlflow
import numpy as np
from stable_baselines3 import PPO

from rlpilot import EnvConfig, SpeedDreamsVecEnv, reference, workdirs
from rlpilot.evaluate import eval_config, eval_time_limits, rollout, summarize
from rlpilot.splits import PROTO7_TEST
from rlpilot.track import race_for, reference_for
from rlpilot.tracking import track_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--dry-run-tracks", nargs="+", default=None,
                   help="other tracks to run on instead, to check this script without spending the test set")
    a = p.parse_args()

    os.chdir(workdirs.ROOT)
    client = mlflow.MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{workdirs.ROOT / 'mlflow.db'}"))
    run = client.get_run(a.run_id)
    best = Path(client.download_artifacts(a.run_id, "best"))
    print(f"run {run.info.run_name}, best/ from step {run.data.tags.get('best_step')}")
    tracks = tuple(a.dry_run_tracks or PROTO7_TEST)
    races = tuple(race_for(t) for t in tracks)
    if not a.dry_run_tracks and "test/score" in run.data.metrics:
        p.error(f"this run's test set was already spent: test/score = {run.data.metrics['test/score']:.3f} (see its test/* metrics)")
    trained = set(ast.literal_eval(run.data.params.get("env.races", "()"))) | {run.data.params.get("env.race")}
    if trained & set(races):
        print(f"WARNING: this run trained on {sorted(trained & set(races))}: not a generalization result")
    cfg = eval_config(EnvConfig.from_flat(run.data.params, "env."), races, a.episodes, a.laps)
    model = PPO.load(best / "model.zip", device="cpu")
    refs = {r: reference.load(reference_for(r), laps=3) for r in races}
    env = SpeedDreamsVecEnv(cfg)
    try:
        traces, lap_times, stats = rollout(lambda o: model.predict(o, deterministic=True)[0], env, clean=range(len(races)),
                                           time_limits=eval_time_limits([refs[r].best_lap for r in races], cfg.n_workers, a.laps))
    finally:
        env.close()
    scores, logged = [], {}
    print(f"{'track':14s} {'completion':>10s} {'progress':>8s} {'flying vs bot %':>15s} {'score':>6s}")
    for t, race in enumerate(races):
        idx = range(t, cfg.n_workers, len(races))
        m = summarize([traces[i] for i in idx], [lap_times[i] for i in idx], [stats[i] for i in idx], refs[race]).metrics
        scores.append(track_score(m, refs[race].best_lap, a.laps))
        fly = m.get("eval/flying_lap_time_median")
        gap = 100 * (fly / refs[race].best_lap - 1) if fly else None
        print(f"{tracks[t]:14s} {m['eval/completion_rate']:10.2f} {m['eval/progress']:8.2f} "
              f"{f'{gap:+.1f}' if fly else '-':>15s} {scores[-1]:6.2f}")
        logged |= {f"test/{tracks[t]}/completion_rate": m["eval/completion_rate"], f"test/{tracks[t]}/progress": m["eval/progress"],
                   f"test/{tracks[t]}/score": scores[-1]} | ({f"test/{tracks[t]}/flying_gap_pct": gap} if fly else {})
    print(f"mean score {np.mean(scores):.3f}")
    if not a.dry_run_tracks:  # these numbers can only be had once: keep them with the run
        for key, value in (logged | {"test/score": float(np.mean(scores))}).items():
            client.log_metric(a.run_id, key, value)


if __name__ == "__main__":
    main()
