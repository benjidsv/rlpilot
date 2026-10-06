"""PPO (or SAC) on SpeedDreamsVecEnv with MLflow tracking.

    python scripts/train_sb3.py --algo ppo --workers 5 --timesteps 2000000 [--experiment rlpilot]
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import VecNormalize

from rlpilot import EnvConfig, SpeedDreamsVecEnv
from rlpilot.sb3 import SB3VecEnv
from rlpilot.tracking import (CheckpointCallback, EvalCallback, MLflowOutputFormat, StdClampCallback, TrainingMetricsCallback,
                              log_provenance, save_model)


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
    p.add_argument("--gearbox", choices=["auto", "points", "agent"], default="auto",
                   help="auto: fixed shift points; points: the agent sets the up/downshift points; agent: the agent shifts")
    p.add_argument("--eval-every", type=int, default=100_000, help="timesteps between evaluations")
    p.add_argument("--eval-episodes", type=int, default=8, help="parallel eval envs (completion rate has 1/N steps)")
    p.add_argument("--eval-laps", type=int, default=3)
    p.add_argument("--log-every", type=int, default=20_000, help="timesteps between training-metric logs")
    p.add_argument("--checkpoint-every", type=int, default=1_000_000, help="timesteps between checkpoints")
    # PPO. Defaults changed after pilot-1: entropy collapsed (std 0.19 -> 0.016) while approx_kl rose to 0.044
    # and the clip fraction to 0.30, and eval completion fell from ~1.0 to ~0.2 after 4M steps.
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--target-kl", type=float, default=0.02, help="stop an update's epochs early above this (0: off)")
    p.add_argument("--min-std", type=float, default=0.135, help="PPO: floor on the action std (0.135 = exp(-2))")
    p.add_argument("--max-std", type=float, default=1.0,
                   help="PPO: cap on the action std; pilot-4's entropy bonus drove it to 127 (actions are clipped to [-1, 1])")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=["linear", "constant"], default="linear",
                   help="linear: decays to 10%% of --lr at the end of training")
    p.add_argument("--n-steps", type=int, default=2048, help="rollout length per env (x workers per update)")
    p.add_argument("--batch-size", type=int, default=2048)
    a = p.parse_args()

    cfg = EnvConfig(n_workers=a.workers, frame_skip=a.frame_skip, laps_per_episode=a.laps_per_episode,
                    gearbox=a.gearbox)
    lr = a.lr if a.lr_schedule == "constant" else (lambda progress_remaining: a.lr * (0.1 + 0.9 * progress_remaining))
    algo_kwargs = {
        "ppo": dict(n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    learning_rate=lr, ent_coef=a.ent_coef, clip_range=0.2, target_kl=a.target_kl or None,
                    policy_kwargs=dict(net_arch=[256, 256])),
        "sac": dict(buffer_size=1_000_000, batch_size=256, learning_starts=10_000, train_freq=1,
                    gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256])),
    }[a.algo]
    logged = algo_kwargs | {"learning_rate": f"{a.lr} ({a.lr_schedule})", "min_std": a.min_std,
                            "max_std": a.max_std} if a.algo == "ppo" else algo_kwargs

    if "MLFLOW_TRACKING_URI" not in os.environ:  # default: local SQLite store, artifacts in ./mlruns
        root = Path(__file__).resolve().parents[1]
        os.chdir(root)
        mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.run_name):
        log_provenance(cfg, **{f"algo.{k}": v for k, v in logged.items()}, algo=a.algo,
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
                CheckpointCallback(every=a.checkpoint_every),
                *([StdClampCallback(env.env.action_names, a.min_std, a.max_std, every=a.log_every)] if a.algo == "ppo" else []),
            ])
        finally:
            save_model(model, "model")  # also on Ctrl-C or a worker error
            venv.close()


if __name__ == "__main__":
    main()
