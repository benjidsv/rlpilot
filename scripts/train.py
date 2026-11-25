# CleanRL's ppo_continuous_action.py (https://github.com/vwxyzjn/cleanrl, commit 35896b1), adapted to rlpilot.
# Every change is marked "rlpilot:". docs: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
#
#     python scripts/train.py --exp-name first --total-timesteps 50000000
#     metrics: Weights & Biases, project RLPilot (--no-track to skip it); eval results: runs/<run>/eval/*.csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import (
    Categorical,
)  # rlpilot: the gear shift is a discrete choice
from torch.distributions.normal import Normal
from tqdm import tqdm  # rlpilot
import wandb

from rlpilot import controls, evaluate, obs as observation  # rlpilot
from rlpilot.env import ENDS, RacingEnv  # rlpilot
from rlpilot.starts import KINDS  # rlpilot
from rlpilot.tracks import Track, allocate, load_split  # rlpilot


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True  # rlpilot: on by default (was False); off = no metrics logged (eval CSVs still written)
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RLPilot"  # rlpilot: was "cleanRL"; the entity defaults to the logged-in account
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    # rlpilot: removed TensorBoard, video capture and the Hugging Face upload; the model is always saved (checkpoints/)

    # Algorithm specific arguments
    # rlpilot: no env_id; the env is RacingEnv on the training tracks of data/split-v2.json
    total_timesteps: int = 50_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 48  # rlpilot: one worker process each, spread over the training tracks by road length (was 1)
    """the number of parallel game environments"""
    num_steps: int = 256  # rlpilot: was 2048; 48 x 256 = 12288 samples per update
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995  # rlpilot: was 0.99; at 25 Hz, 0.99 only looks ~4 s ahead, too short for braking zones
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8  # rlpilot: was 32 (minibatches of 1536 instead of 384)
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # rlpilot: our own arguments
    hidden: int = 256
    """width of the two hidden layers of the actor and the critic (CleanRL: 64)"""
    anneal_timesteps: int = 0
    """steps over which the learning rate anneals to 0 (0 = total-timesteps). Set it to a longer run's length to stop
    early on that run's schedule: a 3M test with 10M here follows a 10M run's first 3M exactly"""
    activation: str = "tanh"
    """activation of the hidden layers: tanh (CleanRL) or relu"""
    first_gain: float = float(np.sqrt(2))
    """orthogonal-init gain of the first layer of the actor and the critic (CleanRL: sqrt 2). With many correlated
    inputs, sqrt 2 starts tanh units saturated: 273 inputs gave a pre-activation std of 1.2, 16% of units past 0.95"""
    hold_bias: float = 3.0
    """initial logit of 'hold' over 'down' / 'up': exp(3) / (exp(3) + 2) = 91% hold, so ~4.5% per shift at the start"""
    episode_seconds: float = 30.0
    """training episodes are cut (and bootstrapped) after this much sim time"""
    lost_control_cost: float = 0.0
    """reward lost each time the car loses control: a new contact, starting to face backwards, or going 3 m beyond the
    road's edge (reward.py; 1 = 10 m of progress). Off by default: the reward sweep found 2 makes the car give up"""
    best_only_progress: bool = False
    """progress pays only past the furthest point reached this episode, so reversing costs nothing (default: signed,
    the phase-1 reward)"""
    lost_control_causes: tuple[str, ...] = ("contact", "backwards", "far_off")
    """which ways of losing control --lost-control-cost charges (all are still tracked and evaluated)"""
    terminal: float = 10.0
    """reward lost when a training episode fails (no progress for 10 s, wrecked). The eval's rules don't change"""
    progress_units: str = "metres"
    """metres (0.1 each) or bot_seconds (each metre pays 40 m/s / the bot's speed there: ~4 per second of the bot's
    lap, the same on average, but a second weighs the same on a straight as in a hairpin)"""
    steer_noise_power: float = 0.0
    """training's steering noise shrinks with speed: its std is multiplied by max(floor, min(1, (steer_noise_speed /
    speed) ^ power)), 1 = halved when the speed doubles, 2 = quartered. 0 = off. The steering itself (its rate) doesn't change,
    nor does the evaluation (it drives the mean)"""
    steer_noise_speed: float = 20.0
    """m/s: below this speed the steering noise is the policy's own std"""
    steer_noise_floor: float = 0.1
    """the steering std's multiplier never goes below this: at power 2 it reaches 0.1 at 63 m/s (the base runs' fastest
    1% of validation driving is above 58 m/s). A std near 0 would also make PPO's ratio explode for tiny changes of
    the mean"""
    obs_height: str = "none"
    """optional inputs (obs.py): the road's height at the edge points, "centre" (21), "edges" (42) or "none" (off)"""
    gearbox: str = "manual"
    """who shifts while the car moves: the policy ("manual") or an rpm rule ("auto", controls.auto_shift: up at 92% of
    the redline, down when the lower gear stays under 70%). With auto, the policy still shifts when the car is nearly
    stopped (1st <-> reverse), and its shift choice counts in the PPO loss only there. A diagnostic: is the car's 3rd
    gear cap (2026-11-24) a gearbox trap?"""
    learned_std: bool = True
    """the std of steer and pedal is an output of the actor (a linear head on its body, per state, within
    [exp(LOGSTD_MIN), exp(LOGSTD_MAX)]) instead of one learned number for every state (CleanRL, --no-learned-std).
    Starts at 1, like it. The steering round (3 seeds x 10M, docs/v2.md): it learns ~0.65 below 20 m/s and ~0.05
    above 40, with better nominal driving than one std"""
    fixed_gear_inputs: bool = False
    """the gear one-hots and the shift cooldown skip the running input normalization (obs.FIXED): with it, a gear the
    car rarely uses reads +7 sd or more when it engages (2026-11-24 speed round, arm A)"""
    pedal_std_floor: float = 0.0
    """the pedal's std never goes below this (learned std: the squashing's lower end, per dimension; it still starts at
    1). 0 = off (the steer and pedal floor is exp(LOGSTD_MIN) = 0.03). The pedal std at speed collapses to ~0.05, and
    past seeds' pace was ordered by how fast it did (2026-11-24 speed round, arm B)"""
    noise_beta: float = 0.0
    """colored exploration noise (Hollenstein et al., AAAI 2024): the Gaussian's standard normal draws z for
    `noise_dims` come from a per-env sequence with power spectrum 1 / f^beta, fresh at each episode, instead of white
    noise. Each step is still N(mean, std), so the log-probability is unchanged. 0 = off (white, as CleanRL); 1 = pink;
    the paper's best for PPO is 0.5. White noise at 25 Hz averages out within a second, so it barely tries "faster on
    this straight" (2026-11-24 speed round, arm C)"""
    noise_dims: tuple[str, ...] = ("pedal",)
    """which actions get the colored noise (steer, pedal): correlated steering noise drifts the car off the road at speed"""
    offroad_cost: float = 0.0
    """reward lost per metre driven and per metre beyond the road's edge, while all four wheels are off (progress
    already pays nothing there): 0.01 = a metre driven 10 m off (the cap) costs a metre of progress (reward.py)"""
    width_bonus: float = 0.0
    """reward per metre of progress at the road's edge, x |offset| / half width: a bonus for using the road's width
    (every policy so far enters corners from the middle). 0.004 = progress at the edge pays 4% more. 0 = off (reward.py)"""
    width_fade: int = 0
    """steps over which width_bonus fades linearly to 0 (0 = it doesn't fade): the bonus gets the car to wide lines,
    then the true reward decides whether it keeps them"""
    line_bonus: float = 0.0
    """reward per metre of progress on the bot's line, x closeness to it (1 on it, 0 a half width away): a curriculum
    towards the whole racing line (wide entry, apex, run-out on the exit). Training only: never an input. 0 = off"""
    line_fade: int = 0
    """steps over which line_bonus fades linearly to 0 (0 = it doesn't fade)"""
    auto_upshift: float = 0.92
    """with --gearbox auto, the upshift point, as a share of the redline (controls.configure). The gears are tall:
    0.995 is ~2% faster at the end of the long straights than 0.92 (2026-11-25)"""
    wall_cost: float = 0.0
    """reward lost per step of wall damage, at 40 m/s: scales with the speed squared (Fuchs et al. 2021: -c |v|^2 in
    wall contact; without it their car grinded along the walls, a fixed cost made it freeze). 1 = 10 m of progress
    at 40 m/s, 2.5 m at 20. 0 = off (reward.py)"""
    eval_every: int = 10_000_000
    """steps between two evaluations: the scenario suite's eval_tiers and 1 grid lap per track (capped at
    evaluate.LAP_CAP), validation and training tracks. The last one plays the whole suite and 2 full laps too"""
    eval_tiers: tuple[str, ...] = ("nominal", "crashed")
    """tiers of the suite played during the run (the last eval plays all). Over 225 checkpoints of the sweeps, mild
    tracked nominal at 0.97-0.98, perturbed and disturbed at 0.82-0.92; crashed is the one that differs (0.1)"""
    eval_workers: int = 48
    """workers of the eval env (results don't depend on it; 48 is the fastest on the Mac's 8 cores)"""
    rollout_threads: int = 1
    """torch threads while collecting rollouts (small batches: more threads only add overhead)"""
    update_threads: int = 4
    """torch threads during the PPO update"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class NormalizeObservation(gym.vector.VectorWrapper):
    # rlpilot: gymnasium's NormalizeObservation + TransformObservation(clip), for an env that resets in the same step
    # (gymnasium's refuse it, since they can't see info["final_obs"]): the final observations are normalized too.
    # rlpilot: inputs marked `fixed` are already at a fixed scale (obs.FIXED) and pass through unchanged.
    def __init__(self, env, epsilon=1e-8, fixed=None):
        super().__init__(env)
        self.obs_rms = gym.wrappers.utils.RunningMeanStd(
            shape=self.single_observation_space.shape
        )
        self.epsilon = epsilon
        self.fixed = (
            np.zeros(self.single_observation_space.shape, bool) if fixed is None else np.asarray(fixed, bool)
        )

    def normalize(
        self, obs
    ):  # without updating the statistics: the final observations, and evaluation
        mean = np.where(self.fixed, 0.0, self.obs_rms.mean)
        std = np.where(self.fixed, 1.0, np.sqrt(self.obs_rms.var + self.epsilon))
        return np.clip((obs - mean) / std, -10, 10).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_rms.update(obs)
        return self.normalize(obs), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.obs_rms.update(obs)
        if "final_obs" in info:
            info["final_obs"] = self.normalize(info["final_obs"])
        return self.normalize(obs), reward, terminated, truncated, info


def make_env(
    args,
):  # rlpilot: one vector env over the simulator's workers, instead of SyncVectorEnv(make_env(...))
    train = load_split()["train"]
    tracks = allocate(
        {t: Track.load(t).length for t in train}, args.num_envs
    )  # samples spread per metre of road
    env = RacingEnv(
        tracks,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        lost_cost=args.lost_control_cost,
        best_only_progress=args.best_only_progress,
        terminal=args.terminal,
        lost_causes=tuple(args.lost_control_causes),
        progress_units=args.progress_units,
        wall_cost=args.wall_cost,  # rlpilot
        offroad_cost=args.offroad_cost,  # rlpilot
        width_bonus=args.width_bonus,  # rlpilot
        line_bonus=args.line_bonus,  # rlpilot
    )
    env = norm = NormalizeObservation(
        env, fixed=observation.FIXED
    )  # kept to normalize the eval's observations the same way
    env = gym.wrappers.vector.NormalizeReward(env, gamma=args.gamma)
    env = gym.wrappers.vector.TransformReward(
        env, lambda reward: np.clip(reward, -10, 10)
    )
    return env, norm


def layer_init(layer, std=None, bias_const=0.0):
    if std is None:
        std = np.sqrt(2)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


LOGSTD_MIN, LOGSTD_MAX = -3.5, 0.5  # rlpilot: a learned std stays in [0.03, 1.65]
LOGSTD_BIAS = float(np.log(-LOGSTD_MIN / LOGSTD_MAX))  # rlpilot: sigmoid(bias) = 3.5 / 4, so the std starts at 1


def logstd_min(pedal_std_floor=0.0):
    """rlpilot: [steer, pedal] lower ends of the learned log std (Args.pedal_std_floor raises the pedal's)"""
    return np.array([LOGSTD_MIN, max(LOGSTD_MIN, float(np.log(pedal_std_floor)) if pedal_std_floor > 0 else LOGSTD_MIN)])


class Agent(nn.Module):
    # rlpilot: a hybrid policy. Actions are [steer, pedal, shift]: a Gaussian over steer and pedal (as in CleanRL) and a
    # categorical over the shift (down / hold / up), from the same actor body. The log-probability of an action is the
    # sum of the two, and so is the entropy.
    def __init__(
        self, envs, hidden=64, hold_bias=0.0, activation="tanh", first_gain=np.sqrt(2), learned_std=False,
        pedal_std_floor=0.0,
    ):  # rlpilot: activation, first_gain, learned_std and pedal_std_floor are our arguments (see Args)
        super().__init__()
        n_obs = np.array(envs.single_observation_space.shape).prod()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, hidden), std=first_gain),
            act(),
            layer_init(nn.Linear(hidden, hidden)),
            act(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_body = (
            nn.Sequential(  # rlpilot: was actor_mean, without its output layer
                layer_init(nn.Linear(n_obs, hidden), std=first_gain),
                act(),
                layer_init(nn.Linear(hidden, hidden)),
                act(),
            )
        )
        self.actor_mean = layer_init(
            nn.Linear(hidden, 2), std=0.01
        )  # rlpilot: steer, pedal
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.shift_logits = layer_init(
            nn.Linear(hidden, 3), std=0.01
        )  # rlpilot: down, hold, up
        with torch.no_grad():
            self.shift_logits.bias[1] = hold_bias
        self.learned_std = learned_std  # rlpilot: a per-state std (Args.learned_std), created last: off = as before
        self.logstd_min = torch.tensor(logstd_min(pedal_std_floor), dtype=torch.float32)  # rlpilot: not a parameter
        if learned_std:
            self.logstd_head = layer_init(nn.Linear(hidden, 2), std=0.01, bias_const=LOGSTD_BIAS)
            if pedal_std_floor > 0:  # rlpilot: a higher lower end for the pedal, its bias moved so its std still starts at 1
                lo = float(self.logstd_min[1])
                with torch.no_grad():
                    self.logstd_head.bias[1] = float(np.log(-lo / LOGSTD_MAX))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, steer_scale=None, shift_mask=None, noise=None):
        # rlpilot: steer_scale [n] multiplies the steering's std (steer_noise_scale); the update must pass the same
        # scale the rollout sampled with, so it is stored with the actions. shift_mask [n] (0 / 1): whether the shift
        # choice was used (--gearbox auto ignores it while moving): an unused choice adds nothing to the log-probability
        # or the entropy
        h = self.actor_body(x)
        action_mean = self.actor_mean(h)
        if self.learned_std:  # rlpilot: squashed into [LOGSTD_MIN, LOGSTD_MAX], so no state's noise can vanish
            if self.logstd_min[1] > LOGSTD_MIN:  # rlpilot: --pedal-std-floor
                lo = self.logstd_min.to(h.device)
                action_logstd = lo + (LOGSTD_MAX - lo) * torch.sigmoid(self.logstd_head(h))
            else:
                action_logstd = LOGSTD_MIN + (LOGSTD_MAX - LOGSTD_MIN) * torch.sigmoid(self.logstd_head(h))
        else:
            action_logstd = self.actor_logstd.expand_as(action_mean)
            if self.logstd_min[1] > LOGSTD_MIN:  # rlpilot: --pedal-std-floor without a learned std: a clamp
                action_logstd = torch.maximum(action_logstd, self.logstd_min.to(h.device))
        action_std = torch.exp(action_logstd)
        if steer_scale is not None:  # rlpilot
            action_std = action_std * torch.stack([steer_scale, torch.ones_like(steer_scale)], 1)
        self.last_mean, self.last_std = action_mean.detach(), action_std.detach()  # rlpilot: for the logs
        probs = Normal(action_mean, action_std)
        shift = Categorical(logits=self.shift_logits(h))
        if action is None:
            if noise is None:
                action = torch.cat([probs.sample(), shift.sample()[:, None].float()], 1)
            else:  # rlpilot: colored noise, noise [n, 2] standard normal draws (ColoredNoise)
                action = torch.cat([action_mean + action_std * noise, shift.sample()[:, None].float()], 1)
        mask = 1.0 if shift_mask is None else shift_mask  # rlpilot
        logprob = probs.log_prob(action[:, :2]).sum(1) + shift.log_prob(
            action[:, 2].long()
        ) * mask
        entropy = probs.entropy().sum(1) + shift.entropy() * mask
        return action, logprob, entropy, self.critic(x)

    def act_deterministic(
        self, x
    ):  # rlpilot: for evaluation: the mean and the most likely shift
        h = self.actor_body(x)
        return torch.cat(
            [self.actor_mean(h), self.shift_logits(h).argmax(1, keepdim=True).float()],
            1,
        )


SPEED = observation.NAMES.index("speed_x / 50")  # rlpilot


def powerlaw_noise(beta, n, size, rng):
    """rlpilot: [size, n] Gaussian sequences with power spectrum 1 / f^beta along the last axis, unit variance
    (Timmer & Koenig 1995, as in the colorednoise package that Hollenstein et al. use). beta 0 = white."""
    f = np.fft.rfftfreq(n)
    f[0] = 1.0 / n  # the lowest frequency stands in for f = 0 (no infinite power)
    scale = f ** (-beta / 2.0)
    w = scale[1:].copy()
    w[-1] *= (1 + (n % 2)) / 2.0
    sigma = 2 * np.sqrt(np.sum(w**2)) / n  # the sequences' theoretical std
    sr, si = rng.normal(size=(size, len(f))) * scale, rng.normal(size=(size, len(f))) * scale
    if n % 2 == 0:  # the Nyquist component is real
        si[:, -1] = 0
        sr[:, -1] *= np.sqrt(2)
    si[:, 0] = 0  # so is the mean
    sr[:, 0] *= np.sqrt(2)
    return np.fft.irfft(sr + 1j * si, n=n, axis=-1) / sigma


class ColoredNoise:
    """rlpilot: per-env standard normal draws for [steer, pedal] (Args.noise_beta): colored along time for the
    dimensions in `dims`, white for the others. Each env reads its own sequence, a fresh one at the start of each
    episode (and when it runs out). Its own generator, so the policy's torch draws don't change."""

    LENGTH = 1024  # steps per sequence: ~43 s, longer than a 30 s training episode

    def __init__(self, n_envs, beta, dims, seed):
        self.n, self.beta = n_envs, beta
        if set(dims) - {"steer", "pedal"}:
            raise ValueError(f"noise_dims: steer and/or pedal, not {dims}")
        self.colored = np.array([d in dims for d in ("steer", "pedal")])
        self.rng = np.random.default_rng(seed)
        self.seq = np.zeros((n_envs, 2, self.LENGTH))
        self.pos = np.zeros(n_envs, np.int64)
        self.refresh(np.arange(n_envs))

    def refresh(self, envs):
        if len(envs) == 0:
            return
        self.seq[envs] = self.rng.normal(size=(len(envs), 2, self.LENGTH))
        for d in np.flatnonzero(self.colored):
            self.seq[envs, d] = powerlaw_noise(self.beta, self.LENGTH, len(envs), self.rng)
        self.pos[envs] = 0

    def draw(self):
        """[n, 2] this step's draws (call once per step)"""
        self.refresh(np.flatnonzero(self.pos >= self.LENGTH))
        z = self.seq[np.arange(self.n), :, self.pos]
        self.pos += 1
        return z


def car_speed(norm, x):
    """rlpilot: x [n, obs] normalized observations -> [n] the car's forward speed in m/s, unsigned (read back from the
    normalization: its statistics are the ones x was normalized with)"""
    m, s = norm.obs_rms.mean[SPEED], np.sqrt(norm.obs_rms.var[SPEED] + norm.epsilon)
    return (x[:, SPEED] * float(s) + float(m)).abs() * 50


def shift_mask(norm, x):
    """rlpilot: x [n, obs] normalized observations -> [n] 1 where the policy's shift choice is used, else 0
    (controls.policy_shifts: always with the manual gearbox, only nearly stopped with the auto one)"""
    return torch.as_tensor(controls.policy_shifts(car_speed(norm, x).cpu().numpy()), dtype=torch.float32,
                           device=x.device)


def steer_noise_scale(norm, x, args):
    """rlpilot: x [n, obs] normalized observations -> [n] multiplier of the steering's std, from the car's speed"""
    if args.steer_noise_power == 0:
        return torch.ones(len(x), device=x.device)
    speed = car_speed(norm, x)
    scale = torch.clamp(args.steer_noise_speed / speed.clamp(min=1e-3), max=1.0) ** args.steer_noise_power
    return scale.clamp(min=args.steer_noise_floor)


# rlpilot: the metrics are logged to W&B in numbered sections: 2_ppo (health), 3_eval (every tier of both splits),
# 4_driving (training episodes), 5_perf. The workspace's charts are laid out by scripts/setup/wandb_workspace.py.


def run_eval(
    agent, norm, args, step, run_dir, best, suite, laps, final
):  # rlpilot: the scenario suite and grid laps, validation + train (final: the whole suite and 2 full laps too)
    torch.set_num_threads(
        1
    )  # a fixed thread count: the same checkpoint must always give the same actions

    def policy(o):
        with torch.no_grad():
            return agent.act_deterministic(torch.as_tensor(norm.normalize(o))).numpy()

    t0 = time.time()
    if final:  # every tier: the scenarios eval_tiers shares play the same (each is a fresh race)
        full = evaluate.Suite(evaluate.load_suite(("validation", "train")), args.eval_workers)
        try:
            rows = full.run(policy)
        finally:
            full.close()
    else:
        rows = suite.run(policy)
    evaluate.write_results(rows, run_dir / "eval" / f"{step:010d}.csv")
    split = load_split()
    split_of = {t: s for s in ("validation", "train") for t in split[s]}
    lap_rows = laps.run(policy)
    evaluate.write_laps(lap_rows, run_dir / "eval" / f"{step:010d}-laps.csv")
    metrics = evaluate.summarize(rows) | evaluate.summarize_laps(lap_rows, split_of)
    if final:
        laps2 = evaluate.grid_laps(policy, laps.tracks, laps=2, cap=None)
        evaluate.write_laps(laps2, run_dir / "eval" / f"{step:010d}-laps2.csv")
        metrics |= evaluate.summarize_laps(laps2, split_of, "laps2")
    wandb.log(
        {f"3_eval/{k}": v for k, v in metrics.items()}
        | {"5_perf/eval_seconds": time.time() - t0, "global_step": step}
    )
    ckpt = {
        "agent": agent.state_dict(),
        "obs_mean": norm.obs_rms.mean,
        "obs_var": norm.obs_rms.var,
        "epsilon": norm.epsilon,
        "obs_fixed": norm.fixed,  # rlpilot: inputs that skip the normalization
        "args": vars(args),
        "step": step,
        "metrics": metrics,
    }
    torch.save(ckpt, run_dir / "checkpoints" / f"{step:010d}.pt")
    score = metrics[
        "validation/nominal/survival"
    ]  # best/ = nominal survival on validation tracks
    if score > best:
        torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
        best = score
    tqdm.write(  # rlpilot: above the progress bar
        f"eval at {step}: validation nominal survival {score:.3f} (best {best:.3f}), lap pace "
        f"{metrics['validation/lap/pace']:.3f}, lap clean {metrics['validation/lap/clean']:.3f}; train nominal survival "
        f"{metrics['train/nominal/survival']:.3f}, lap pace {metrics['train/lap/pace']:.3f}; {time.time() - t0:.0f} s"
    )
    torch.set_num_threads(args.rollout_threads)
    return best


if __name__ == "__main__":
    args = tyro.cli(Args)
    observation.configure(args.obs_height, args.fixed_gear_inputs)  # rlpilot: before any env or policy is made
    controls.configure(args.gearbox, args.auto_upshift)  # rlpilot
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"  # rlpilot: no env_id
    run_dir = Path("runs") / run_name  # rlpilot
    (run_dir / "checkpoints").mkdir(parents=True)  # rlpilot
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))  # rlpilot
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),  # rlpilot: replaces the hyperparameters text
        name=run_name,
        group=args.exp_name,  # rlpilot: the seeds of one setup share a group (mean + band in the UI)
        dir=str(run_dir),  # rlpilot: wandb's local files go in the run's folder
        save_code=True,
        mode=None if args.track else "disabled",  # rlpilot: disabled = wandb.log does nothing
    )  # rlpilot: no sync_tensorboard (no TensorBoard), no monitor_gym (no videos)
    for section in ("2_ppo", "3_eval", "4_driving", "5_perf"):  # rlpilot: plot against env steps
        wandb.define_metric(f"{section}/*", step_metric="global_step")
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs, norm = make_env(args)  # rlpilot

    agent = Agent(envs, args.hidden, args.hold_bias, args.activation, args.first_gain, args.learned_std,
                  args.pedal_std_floor).to(
        device
    )  # rlpilot: hidden width, hold bias
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    steer_scales = torch.ones((args.num_steps, args.num_envs)).to(device)  # rlpilot: see steer_noise_scale
    shift_masks = torch.ones((args.num_steps, args.num_envs)).to(device)  # rlpilot: see shift_mask
    stds = torch.zeros((args.num_steps, args.num_envs, 2)).to(device)  # rlpilot: the std applied, for the logs
    speeds = torch.zeros((args.num_steps, args.num_envs)).to(device)  # rlpilot: m/s, for the logs
    means = torch.zeros((args.num_steps, args.num_envs, 2)).to(device)  # rlpilot: the Gaussian's mean, for the logs

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    colored = ColoredNoise(args.num_envs, args.noise_beta, args.noise_dims, args.seed) if args.noise_beta > 0 \
        else None  # rlpilot: off = probs.sample(), as CleanRL
    torch.set_num_threads(args.rollout_threads)  # rlpilot
    next_eval, best = args.eval_every, -1.0  # rlpilot
    suite = evaluate.Suite(
        [r for r in evaluate.load_suite(("validation", "train")) if r["tier"] in args.eval_tiers], args.eval_workers
    )  # rlpilot: the eval's workers, up for the whole run (starting them takes ~12 s)
    laps = evaluate.GridLaps(load_split()["validation"] + load_split()["train"], laps=1)  # rlpilot: same
    bar = tqdm(total=args.num_iterations * args.batch_size, unit="step", unit_scale=True, desc="train",
               dynamic_ncols=True, smoothing=0.05)  # rlpilot: replaces the SPS print

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / (
                (args.anneal_timesteps or args.total_timesteps) // args.batch_size
            )  # rlpilot: was / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.width_fade > 0:  # rlpilot: --width-fade, set before the rollout (the env computes the reward)
            norm.env.reward.width_bonus = args.width_bonus * max(0.0, 1.0 - (iteration - 1) * args.batch_size
                                                                 / args.width_fade)
        if args.line_fade > 0:  # rlpilot: --line-fade, the same way
            norm.env.reward.line_bonus = args.line_bonus * max(0.0, 1.0 - (iteration - 1) * args.batch_size
                                                               / args.line_fade)

        ended = []  # rlpilot: statistics of the episodes that ended during this rollout
        logs = {}  # rlpilot: this iteration's metrics, one wandb.log at its end
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                steer_scales[step] = steer_noise_scale(norm, next_obs, args)  # rlpilot
                shift_masks[step] = shift_mask(norm, next_obs)  # rlpilot
                noise = None if colored is None else torch.as_tensor(colored.draw(), dtype=torch.float32,
                                                                     device=device)  # rlpilot
                action, logprob, _, value = agent.get_action_and_value(next_obs, steer_scale=steer_scales[step],
                                                                       shift_mask=shift_masks[step], noise=noise)
                values[step] = value.flatten()
                stds[step], speeds[step] = agent.last_std, car_speed(norm, next_obs)  # rlpilot
                means[step] = agent.last_mean  # rlpilot
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            if colored is not None:  # rlpilot: a fresh noise sequence for each new episode
                colored.refresh(np.flatnonzero(next_done))
            # rlpilot: a time-limit cut isn't the end of the task: bootstrap from the value of the state it cut at.
            # CleanRL treats it like a crash (done = terminated | truncated, no value after it). infos["final_obs"]
            # is the last observation (the returned one is already the next episode's first).
            if truncations.any():
                cut = np.flatnonzero(truncations)
                with torch.no_grad():
                    final = torch.Tensor(infos["final_obs"][cut]).to(device)
                    reward = reward.copy()
                    reward[cut] += (
                        args.gamma * agent.get_value(final).flatten().cpu().numpy()
                    )
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "episode_end" in infos:  # rlpilot: our env's per-episode statistics
                e, m = infos["episode_end"], infos["_episode_end"]
                ended += [
                    {k: float(v[i]) for k, v in e.items()} for i in np.flatnonzero(m)
                ]

        # rlpilot: log the rollout's episodes (CleanRL printed each episode's return)
        if ended:
            for k in (
                "return",
                "seconds",
                "distance",
                "failed_near_start",
                "shifts_per_min",
                "refused_shifts",
                "offroad_frac",
                "mean_speed",
                "reward_progress",
                "reward_damage",
                "reward_control",
                "reward_terminal",
                "reward_wall",
                "reward_offroad",
                "reward_width",
                "reward_line",
                "lost_contact",
                "lost_backwards",
                "lost_far_off",
                "damage",
            ):
                logs[f"4_driving/{k}"] = np.mean([x[k] for x in ended])
            for code, name in enumerate(ENDS):
                logs[f"4_driving/end/{name}"] = np.mean([x["end"] == code for x in ended])
            for code, name in enumerate(KINDS):
                mine = [x for x in ended if x["kind"] == code]
                if mine:
                    logs[f"4_driving/failed_near_start/{name}"] = np.mean(
                        [x["failed_near_start"] for x in mine]
                    )
        # rlpilot: the policy's std (before any speed scaling; with --learned-std, its mean over the rollout), then the
        # std applied (after it), overall, at low speed and at high speed
        policy_std = (stds / torch.stack([steer_scales, torch.ones_like(steer_scales)], -1)).reshape(-1, 2).mean(0)
        logs["2_ppo/std_steer"], logs["2_ppo/std_pedal"] = policy_std.tolist()
        logs["2_ppo/std_steer_applied"] = stds[..., 0].mean().item()
        for band, sel in (("slow", speeds < 20), ("fast", speeds > 40)):  # m/s
            if sel.any():
                logs[f"2_ppo/std_steer_{band}"] = stds[..., 0][sel].mean().item()
                logs[f"2_ppo/std_pedal_{band}"] = stds[..., 1][sel].mean().item()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

            # rlpilot: the std's pressure, per speed band: the mean of A * (z^2 - 1), A the normalized advantage and z
            # how far the sampled action strayed from the mean in stds. It's the policy gradient on log std (at ratio
            # 1): > 0 the noise paid and PPO raises the std there, < 0 it cost and PPO lowers it, ~0 no signal
            z = (actions[..., :2] - means) / stds
            adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pressure = adv[..., None] * (z**2 - 1)
            for band, sel in (("slow", speeds < 20), ("fast", speeds > 40)):  # m/s
                if sel.any():
                    logs[f"2_ppo/std_pressure_steer_{band}"] = pressure[..., 0][sel].mean().item()
                    logs[f"2_ppo/std_pressure_pedal_{band}"] = pressure[..., 1][sel].mean().item()

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_steer_scales = steer_scales.reshape(-1)  # rlpilot
        b_shift_masks = shift_masks.reshape(-1)  # rlpilot

        # Optimizing the policy and value network
        torch.set_num_threads(args.update_threads)  # rlpilot
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds], steer_scale=b_steer_scales[mb_inds],
                    shift_mask=b_shift_masks[mb_inds]
                )  # rlpilot: the steering noise's scale the actions were sampled with, which shifts were used
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        torch.set_num_threads(args.rollout_threads)  # rlpilot

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # rlpilot: grouped under 2_ppo and 5_perf, logged to W&B; old_approx_kl dropped
        logs["2_ppo/learning_rate"] = optimizer.param_groups[0]["lr"]
        logs["2_ppo/width_bonus"] = norm.env.reward.width_bonus  # rlpilot
        logs["2_ppo/line_bonus"] = norm.env.reward.line_bonus  # rlpilot
        logs["2_ppo/value_loss"] = v_loss.item()
        logs["2_ppo/policy_loss"] = pg_loss.item()
        logs["2_ppo/entropy"] = entropy_loss.item()
        logs["2_ppo/approx_kl"] = approx_kl.item()
        logs["2_ppo/clipfrac"] = np.mean(clipfracs)
        logs["2_ppo/explained_variance"] = explained_var
        bar.update(args.batch_size)  # rlpilot
        bar.set_postfix(
            sps=int(global_step / (time.time() - start_time)),
            ret=f"{np.mean([x['return'] for x in ended]):.0f}" if ended else "-",
            speed=f"{np.mean([x['mean_speed'] for x in ended]):.1f}" if ended else "-",
            best=f"{best:.3f}" if best >= 0 else "-",
            refresh=False,
        )
        logs["5_perf/SPS"] = int(global_step / (time.time() - start_time))
        wandb.log(logs | {"global_step": global_step})  # rlpilot

        if (
            global_step >= next_eval or iteration == args.num_iterations
        ):  # rlpilot: evaluate and checkpoint
            bar.set_description("eval")  # rlpilot
            best = run_eval(
                agent.cpu(), norm, args, global_step, run_dir, best, suite, laps,
                final=iteration == args.num_iterations,
            )
            bar.set_description("train")
            agent.to(device)
            next_eval += args.eval_every

    bar.close()  # rlpilot
    envs.close()
    suite.close()  # rlpilot
    laps.close()  # rlpilot
    wandb.finish()  # rlpilot: waits for the last uploads
