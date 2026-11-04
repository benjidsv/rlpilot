"""PPO (or SAC) on SpeedDreamsVecEnv with MLflow tracking.

    python scripts/train_sb3.py --algo ppo --workers 5 --timesteps 2000000 [--experiment rlpilot]
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import mlflow
from stable_baselines3 import SAC
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.vec_env import VecNormalize

from rlpilot import EnvConfig, SpeedDreamsVecEnv
from rlpilot.config import RewardConfig
from rlpilot.sb3 import CpuRolloutPPO, SB3VecEnv, resume_ppo
from rlpilot.splits import PROTO7_EVAL_TRAIN, PROTO7_TRAIN, PROTO7_VALIDATION
from rlpilot.track import race_for
from rlpilot.tracking import (CheckpointCallback, EvalCallback, LineShapingFadeCallback, MLflowOutputFormat,
                              ObsStatsCallback, StdClampCallback, TrainingMetricsCallback, log_provenance, save_model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    # Workers mostly wait for the Python side, so more than the core count pays: 20 workers sleeping after a
    # 50 us spin gave ~12k steps/s on an M1 Pro vs ~8.6k for 5 workers / 200 us (keep n_steps x workers ~10k).
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--spin-us", type=int, default=50, help="worker busy-wait before sleeping between steps")
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--experiment", default="rlpilot")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None, help="network init and action sampling (the sim is deterministic)")
    p.add_argument("--device", default="cpu", help="small MLPs are usually fastest on CPU")
    p.add_argument("--laps-per-episode", type=int, default=0, help="training episodes end after N laps (0: time only)")
    p.add_argument("--gearbox", choices=["auto", "points", "agent"], default="auto",
                   help="auto: fixed shift points; points: the agent sets the up/downshift points; agent: the agent shifts")
    p.add_argument("--tracks", nargs="+", default=["jarama"],
                   help="training tracks (data/raceman/rlpilot-<track>.xml); worker i drives track i modulo the number of tracks")
    p.add_argument("--split", choices=["proto7"], default=None,
                   help="sets the training, validation and evaluated tracks from rlpilot.splits; the test tracks stay out")
    p.add_argument("--heldout-tracks", nargs="*", default=[], help="evaluated at every eval, never trained on")
    p.add_argument("--eval-train-tracks", nargs="*", default=[],
                   help="training tracks to evaluate (default: all); the first gets eval/* and the plots")
    p.add_argument("--track-switch-steps", type=int, default=0,
                   help="a worker switches to a random track at its first reset after this many steps on it "
                        "(0: never, so --workers must be a multiple of the number of tracks; proto-3..6: 200000)")
    p.add_argument("--recycle-resets", type=int, default=EnvConfig.recycle_resets,
                   help="replace a worker process after this many resets (each costs ~0.14 MB until then): a worker peaks "
                        "at ~45 MB + 0.14 MB x this, and a respawn blocks the batch ~0.4 s")
    p.add_argument("--lookahead", action="store_true", help="observe the track curvature over the next 300 m")
    p.add_argument("--lookahead-stat", choices=["peak", "mean"], default="peak",
                   help="curvature per lookahead window: signed peak (default) or mean (proto-3/4)")
    p.add_argument("--line-obs", action="store_true",
                   help="observe the bot's racing line over the next 200 m, relative to the car")
    p.add_argument("--line-shaping", type=float, default=0.0,
                   help="potential-based shaping towards the bot's racing line, per metre of offset (0: off; ~0.1)")
    p.add_argument("--line-cap", type=float, default=RewardConfig.line_cap,
                   help="offsets beyond this (m) count as this in the shaping potential")
    p.add_argument("--line-fade", type=float, default=0.5,
                   help="line shaping fades linearly to 0 at this fraction of --timesteps (0: never)")
    p.add_argument("--edge-points", type=int, default=0,
                   help="observe this many points per road edge ahead, in the car's frame, from the track map (0: off)")
    p.add_argument("--edge-horizon", type=float, default=EnvConfig.edge_horizon, help="s of travel the edge points span")
    p.add_argument("--edge-floor", type=float, default=EnvConfig.edge_floor, help="m, what they span near standstill")
    p.add_argument("--steer-limit", type=float, default=None,
                   help="the steer action becomes a change of steering, up to this per step (0.0625 per 40 ms = 70 deg/s)")
    p.add_argument("--steer-history", type=int, default=0, help="observe the last N steering changes (needs --steer-limit)")
    p.add_argument("--steer-change", type=float, default=RewardConfig.steer_change,
                   help="reward penalty per step on |change of steer| (command units)")
    p.add_argument("--steer-reversal", type=float, default=RewardConfig.steer_reversal,
                   help="reward penalty per step where consecutive steering changes reverse, both above the dead-band")
    p.add_argument("--steer-deadband", type=float, default=RewardConfig.steer_deadband,
                   help="smallest steering change (command units per step) that counts as a reversal")
    p.add_argument("--resume", default=None,
                   help="PPO: continue from a directory with model.zip and vecnormalize.pkl (e.g. mlruns/.../best). "
                        "Steps restart at 0, the PPO flags here replace the saved values (--lr and --lr-schedule from there), "
                        "and step 0 is evaluated")
    p.add_argument("--steer-rate", type=float, default=RewardConfig.steer_rate,
                   help="reward penalty per step on (change of steer)^2, against steer chatter (0: off)")
    p.add_argument("--launch-share", type=float, default=EnvConfig.launch_share,
                   help="share of each worker's samples from episodes starting anywhere on the lap (0: grid only; eval: grid)")
    p.add_argument("--launch-max-speed", type=float, default=EnvConfig.launch_speed[1], help="m/s, launch speed drawn from 0 to this")
    p.add_argument("--car-random", type=float, default=0.0,
                   help="per episode, draw tyre grip, mass and engine power factors in [1 - x, 1 + x] (eval: stock car)")
    p.add_argument("--car-obs", action="store_true", help="observe the car factors (else the policy must be robust)")
    p.add_argument("--eval-every", type=int, default=100_000, help="timesteps between evaluations")
    p.add_argument("--eval-episodes", type=int, default=8,
                   help="eval envs per evaluated track, all tracks in parallel (completion rate has 1/N steps)")
    p.add_argument("--eval-laps", type=int, default=2,
                   help="laps per eval episode (2: one standing-start lap + one flying lap; proto-4 used 3)")
    p.add_argument("--eval-smooth", type=int, default=3,
                   help="best/ = best eval of the best window of this many consecutive evals (by mean score)")
    p.add_argument("--log-every", type=int, default=20_000, help="timesteps between training-metric logs")
    p.add_argument("--checkpoint-every", type=int, default=1_000_000, help="timesteps between checkpoints")
    # PPO. Defaults changed after pilot-1: entropy collapsed (std 0.19 -> 0.016) while approx_kl rose to 0.044
    # and the clip fraction to 0.30, and eval completion fell from ~1.0 to ~0.2 after 4M steps.
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--target-kl", type=float, default=0.02, help="stop an update's epochs early above this (0: off)")
    p.add_argument("--min-std", type=float, default=0.135, help="PPO: floor on the action std (0.135 = exp(-2))")
    p.add_argument("--max-std", type=float, default=1.0,
                   help="PPO: cap on the action std; pilot-4's entropy bonus drove it to 127 (actions are clipped to [-1, 1])")
    p.add_argument("--gamma", type=float, default=0.995,
                   help="discount per agent step: the horizon is ~1/(1-gamma) steps, e.g. 0.99 at 25 Hz = 4 s, "
                        "0.995 = 8 s (proto-4 braked too late for Spa's Bus Stop, ~4.2 s out at 0.99)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=["linear", "constant"], default="linear",
                   help="linear: decays to 10%% of --lr at the end of training")
    p.add_argument("--n-steps", type=int, default=512, help="rollout length per env (x workers per update)")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--no-report", action="store_true", help="don't run scripts/report.py when training ends (PPO only)")
    a = p.parse_args()

    if a.split == "proto7":
        if a.tracks != p.get_default("tracks") or a.heldout_tracks or a.eval_train_tracks:
            p.error("--split sets the tracks: it can't be combined with --tracks, --heldout-tracks or --eval-train-tracks")
        a.tracks, a.heldout_tracks, a.eval_train_tracks = PROTO7_TRAIN, PROTO7_VALIDATION, PROTO7_EVAL_TRAIN
    races = tuple(race_for(t) for t in a.tracks)
    cfg = EnvConfig(n_workers=a.workers, frame_skip=a.frame_skip, laps_per_episode=a.laps_per_episode,
                    gearbox=a.gearbox, race=races[0], spin_us=a.spin_us, races=races if len(races) > 1 else (),
                    track_switch_steps=a.track_switch_steps, recycle_resets=a.recycle_resets, lookahead=a.lookahead, lookahead_stat=a.lookahead_stat,
                    launch_share=a.launch_share, launch_speed=(0.0, a.launch_max_speed),
                    line_obs=a.line_obs, edge_points=a.edge_points, edge_horizon=a.edge_horizon, edge_floor=a.edge_floor,
                    steer_limit=a.steer_limit, steer_history=a.steer_history,
                    reward=RewardConfig(steer_rate=a.steer_rate, steer_change=a.steer_change,
                                        steer_reversal=a.steer_reversal, steer_deadband=a.steer_deadband,
                                        line_shaping=a.line_shaping, line_cap=a.line_cap, gamma=a.gamma),
                    car_grip=(1 - a.car_random, 1 + a.car_random), car_mass=(1 - a.car_random, 1 + a.car_random),
                    car_power=(1 - a.car_random, 1 + a.car_random), car_obs=a.car_obs)
    lr = a.lr if a.lr_schedule == "constant" else (lambda progress_remaining: a.lr * (0.1 + 0.9 * progress_remaining))
    algo_kwargs = {
        "ppo": dict(n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=10, gamma=a.gamma, gae_lambda=0.95,
                    learning_rate=lr, ent_coef=a.ent_coef, clip_range=0.2, target_kl=a.target_kl or None,
                    policy_kwargs=dict(net_arch=[256, 256])),
        "sac": dict(gamma=a.gamma, buffer_size=1_000_000, batch_size=256, learning_starts=10_000, train_freq=1,
                    gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256])),
    }[a.algo]
    logged = algo_kwargs | {"learning_rate": f"{a.lr} ({a.lr_schedule})", "min_std": a.min_std,
                            "max_std": a.max_std} if a.algo == "ppo" else algo_kwargs

    if "MLFLOW_TRACKING_URI" not in os.environ:  # default: local SQLite store, artifacts in ./mlruns
        root = Path(__file__).resolve().parents[1]
        os.chdir(root)
        mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.run_name) as run:
        log_provenance(cfg, **{f"algo.{k}": v for k, v in logged.items()}, algo=a.algo,
                       timesteps=a.timesteps, seed=a.seed, resume=a.resume)
        env = SB3VecEnv(SpeedDreamsVecEnv(cfg))
        Model = {"ppo": CpuRolloutPPO, "sac": SAC}[a.algo]
        if a.resume:
            if a.algo != "ppo":
                p.error("--resume is PPO only")
            model = resume_ppo(a.resume, env, a.device, algo_kwargs)
            venv = model.get_env()
            model.set_random_seed(a.seed)
        else:
            venv = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=a.gamma)
            model = Model("MlpPolicy", venv, device=a.device, verbose=0, seed=a.seed, **algo_kwargs)
        model.set_logger(Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout),
                                                             MLflowOutputFormat()]))
        try:
            model.learn(a.timesteps, callback=[
                TrainingMetricsCallback(every=a.log_every),
                *([ObsStatsCallback()] if a.algo == "ppo" else []),
                EvalCallback(cfg, every=a.eval_every, episodes=a.eval_episodes, laps=a.eval_laps, verbose=1,
                             heldout=tuple(race_for(t) for t in a.heldout_tracks),
                             eval_train=tuple(race_for(t) for t in a.eval_train_tracks), smooth=a.eval_smooth,
                             at_start=bool(a.resume)),
                CheckpointCallback(every=a.checkpoint_every),
                *([LineShapingFadeCallback(a.line_shaping, a.timesteps, a.line_fade)] if a.line_shaping > 0 else []),
                *([StdClampCallback(env.env.action_names, a.min_std, a.max_std, every=a.log_every)] if a.algo == "ppo" else []),
            ])
        finally:
            save_model(model, "model")  # also on Ctrl-C or a worker error
            venv.close()
    if a.algo == "ppo" and not a.no_report:  # only reached when training ended by itself. Its own process: it can't harm the run
        subprocess.run([sys.executable, str(Path(__file__).with_name("report.py")), run.info.run_id])


if __name__ == "__main__":
    main()
