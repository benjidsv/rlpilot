"""PPO (or SAC) on SpeedDreamsVecEnv with MLflow tracking.

    python scripts/train_sb3.py --algo ppo --workers 5 --timesteps 2000000 [--experiment rlpilot]
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mlflow
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import VecNormalize

from rlpilot import EnvConfig, SpeedDreamsVecEnv
from rlpilot.sb3 import SB3VecEnv
from rlpilot.tracking import EvalCallback, MLflowOutputFormat, TrainingMetricsCallback, log_provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--experiment", default="rlpilot")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cpu", help="small MLPs are usually fastest on CPU")
    p.add_argument("--laps-per-episode", type=int, default=0, help="training episodes end after N laps (0: time only)")
    p.add_argument("--eval-every", type=int, default=100_000, help="timesteps between evaluations")
    p.add_argument("--eval-episodes", type=int, default=4)
    p.add_argument("--eval-laps", type=int, default=3)
    p.add_argument("--log-every", type=int, default=20_000, help="timesteps between training-metric logs")
    a = p.parse_args()

    cfg = EnvConfig(n_workers=a.workers, frame_skip=a.frame_skip, laps_per_episode=a.laps_per_episode)
    algo_kwargs = {
        "ppo": dict(n_steps=512, batch_size=1024, n_epochs=10, gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                    ent_coef=0.0, clip_range=0.2, policy_kwargs=dict(net_arch=[256, 256])),
        "sac": dict(buffer_size=1_000_000, batch_size=256, learning_starts=10_000, train_freq=1,
                    gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256])),
    }[a.algo]

    if "MLFLOW_TRACKING_URI" not in os.environ:  # default: local SQLite store, artifacts in ./mlruns
        root = Path(__file__).resolve().parents[1]
        os.chdir(root)
        mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.run_name):
        log_provenance(cfg, **{f"algo.{k}": v for k, v in algo_kwargs.items()}, algo=a.algo,
                       timesteps=a.timesteps)
        env = SB3VecEnv(SpeedDreamsVecEnv(cfg))
        venv = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=algo_kwargs.get("gamma", 0.99))
        Model = {"ppo": PPO, "sac": SAC}[a.algo]
        model = Model("MlpPolicy", venv, device=a.device, verbose=0, **algo_kwargs)
        model.set_logger(Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout),
                                                             MLflowOutputFormat()]))
        try:
            model.learn(a.timesteps, callback=[
                TrainingMetricsCallback(every=a.log_every),
                EvalCallback(cfg, every=a.eval_every, episodes=a.eval_episodes, laps=a.eval_laps, verbose=1),
            ])
        finally:
            with tempfile.TemporaryDirectory() as d:
                model.save(Path(d) / "model.zip")
                venv.save(str(Path(d) / "vecnormalize.pkl"))
                mlflow.log_artifacts(d, "model")
            venv.close()


if __name__ == "__main__":
    main()
