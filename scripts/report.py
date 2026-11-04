"""Report of a finished run: an HTML page with a tracks x steps heatmap, then each track's crash spots and GIFs at best/.
Runs by itself after training; takes a few minutes. See rlpilot/report.py.

    python scripts/report.py                     # the latest run
    python scripts/report.py proto-7             # a run name (its latest run) or a run id
    python scripts/report.py proto-7 --resweep   # drop the cached simulation results

Writes reports/<run name>-<run id>/index.html. Re-running only renders what is missing: delete img/ to redraw figures.
"""

import argparse
import os
import shutil

import mlflow

from rlpilot import report, workdirs


def find_run(client: mlflow.MlflowClient, ref: str | None):
    experiments = [e.experiment_id for e in client.search_experiments()]
    query = f"tags.`mlflow.runName` = '{ref}'" if ref else ""
    runs = client.search_runs(experiments, query, order_by=["attributes.start_time DESC"], max_results=1)
    if runs:
        return runs[0]
    if ref:
        return client.get_run(ref)  # raises if it isn't a run id either
    raise SystemExit("no runs in this MLflow store")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", help="run name or id (default: the latest run)")
    p.add_argument("--checkpoints", type=int, default=38, help="columns of the heatmap")
    p.add_argument("--episodes", type=int, default=2, help="per track and checkpoint (the first one has no launch noise)")
    p.add_argument("--best-episodes", type=int, default=8, help="per track on best/, for the crash spots")
    p.add_argument("--resweep", action="store_true", help="drop the cached results and images first")
    a = p.parse_args()

    os.chdir(workdirs.ROOT)
    client = mlflow.MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{workdirs.ROOT / 'mlflow.db'}"))
    run = find_run(client, a.run)
    out = workdirs.ROOT / "reports" / f"{run.info.run_name}-{run.info.run_id[:8]}"
    if a.resweep and out.exists():
        shutil.rmtree(out)
    print(f"report of {run.info.run_name} ({run.info.run_id})")
    print(report.build(run, client, out, a.checkpoints, a.episodes, a.best_episodes))


if __name__ == "__main__":
    main()
