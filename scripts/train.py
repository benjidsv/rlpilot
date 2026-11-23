# CleanRL's ppo_continuous_action.py (https://github.com/vwxyzjn/cleanrl, commit 35896b1), adapted to rlpilot.
# Every change is marked "rlpilot:". docs: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
#
#     python scripts/train.py --exp-name first --total-timesteps 50000000
#     tensorboard --logdir runs
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # rlpilot

from rlpilot import evaluate, obs as observation  # rlpilot
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
    # rlpilot: removed wandb, video capture and the Hugging Face upload; the model is always saved (checkpoints/)

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
    eval_every: int = 10_000_000
    """steps between two evaluations on the scenario suite (validation and training tracks)"""
    eval_workers: int = 48
    """workers of the eval env; part of the suite's fixed layout, so keep it the same between runs"""
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
    env = RacingEnv(tracks, episode_seconds=args.episode_seconds, seed=args.seed)
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


class Agent(nn.Module):
    # rlpilot: a hybrid policy. Actions are [steer, pedal, shift]: a Gaussian over steer and pedal (as in CleanRL) and a
    # categorical over the shift (down / hold / up), from the same actor body. The log-probability of an action is the
    # sum of the two, and so is the entropy.
    def __init__(
        self, envs, hidden=64, hold_bias=0.0, activation="tanh", first_gain=np.sqrt(2)
    ):  # rlpilot: activation and first_gain are our arguments (see Args)
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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        h = self.actor_body(x)
        action_mean = self.actor_mean(h)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        shift = Categorical(logits=self.shift_logits(h))
        if action is None:
            action = torch.cat([probs.sample(), shift.sample()[:, None].float()], 1)
        logprob = probs.log_prob(action[:, :2]).sum(1) + shift.log_prob(
            action[:, 2].long()
        )
        entropy = probs.entropy().sum(1) + shift.entropy()
        return action, logprob, entropy, self.critic(x)

    def act_deterministic(
        self, x
    ):  # rlpilot: for evaluation: the mean and the most likely shift
        h = self.actor_body(x)
        return torch.cat(
            [self.actor_mean(h), self.shift_logits(h).argmax(1, keepdim=True).float()],
            1,
        )


# rlpilot: TensorBoard layout. The Scalars tab shows one metric per chart, in numbered sections (1_dashboard: the
# numbers that say whether a run is better; 2_ppo: health; 3_eval: every tier of both splits; 4_driving: training
# episodes; 5_perf), and compares runs well: one line per run. The Custom Scalars tab draws several metrics on one
# chart (LAYOUT): best for reading one run, since each run multiplies the lines.
DASHBOARD = {  # 1_dashboard/<name>: eval metric
    "validation_nominal_survival": "validation/nominal/survival",
    "train_nominal_survival": "train/nominal/survival",
    "validation_pace": "validation/pace",
    "validation_crashed_completion": "validation/crashed/completion",
    "validation_disturbed_survival": "validation/disturbed/survival",
}
LAYOUT = {
    "dashboard": {
        "nominal survival, validation (90% band over tracks)": [
            "Margin",
            [
                "3_eval/validation/nominal/survival",
                "3_eval/validation/nominal/survival_lo",
                "3_eval/validation/nominal/survival_hi",
            ],
        ],
        "nominal survival, train vs validation": [
            "Multiline",
            [r"3_eval/(train|validation)/nominal/survival$"],
        ],
        "recovery, validation": [
            "Multiline",
            [r"3_eval/validation/(crashed/completion|disturbed/survival)$"],
        ],
    },
    "eval by tier": {
        f"{m}, {split}": ["Multiline", [rf"3_eval/{split}/[a-z]+/{m}$"]]
        for split in ("validation", "train")
        for m in ("survival", "completion", "clean")
    },
    "driving": {
        "how episodes end": ["Multiline", [r"4_driving/end/"]],
        "failed near start, by kind of start": [
            "Multiline",
            [r"4_driving/failed_near_start/"],
        ],
        "reward components": ["Multiline", [r"4_driving/reward_"]],
    },
    "ppo": {"action std": ["Multiline", [r"2_ppo/std_"]]},
}


def run_eval(
    agent, norm, args, step, run_dir, writer, best
):  # rlpilot: the scenario suite, validation + train
    torch.set_num_threads(
        1
    )  # a fixed thread count: the same checkpoint must always give the same actions

    def policy(o):
        with torch.no_grad():
            return agent.act_deterministic(torch.as_tensor(norm.normalize(o))).numpy()

    t0 = time.time()
    rows = evaluate.run_suite(
        policy, evaluate.load_suite(("validation", "train")), args.eval_workers
    )
    evaluate.write_results(rows, run_dir / "eval" / f"{step:010d}.csv")
    metrics = evaluate.summarize(rows)
    for k, v in metrics.items():
        writer.add_scalar(f"3_eval/{k}", v, step)
    for name, k in DASHBOARD.items():
        writer.add_scalar(f"1_dashboard/{name}", metrics[k], step)
    writer.add_scalar("5_perf/eval_seconds", time.time() - t0, step)
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
        f"eval at {step}: validation nominal survival {score:.3f} (best {best:.3f}), pace "
        f"{metrics['validation/pace']:.3f}, train nominal survival {metrics['train/nominal/survival']:.3f}, "
        f"{time.time() - t0:.0f} s"
    )
    torch.set_num_threads(args.rollout_threads)
    return best


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"  # rlpilot: no env_id
    run_dir = Path("runs") / run_name  # rlpilot
    (run_dir / "checkpoints").mkdir(parents=True)  # rlpilot
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))  # rlpilot
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    writer.add_custom_scalars(LAYOUT)  # rlpilot
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs, norm = make_env(args)  # rlpilot

    agent = Agent(envs, args.hidden, args.hold_bias, args.activation, args.first_gain).to(
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

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    torch.set_num_threads(args.rollout_threads)  # rlpilot
    next_eval, best = args.eval_every, -1.0  # rlpilot
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

        ended = []  # rlpilot: statistics of the episodes that ended during this rollout
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
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
                "reward_terminal",
            ):
                writer.add_scalar(
                    f"4_driving/{k}", np.mean([x[k] for x in ended]), global_step
                )
            for code, name in enumerate(ENDS):
                writer.add_scalar(
                    f"4_driving/end/{name}",
                    np.mean([x["end"] == code for x in ended]),
                    global_step,
                )
            for code, name in enumerate(KINDS):
                mine = [x for x in ended if x["kind"] == code]
                if mine:
                    writer.add_scalar(
                        f"4_driving/failed_near_start/{name}",
                        np.mean([x["failed_near_start"] for x in mine]),
                        global_step,
                    )
        writer.add_scalar(
            "2_ppo/std_steer", agent.actor_logstd[0, 0].exp().item(), global_step
        )  # rlpilot
        writer.add_scalar(
            "2_ppo/std_pedal", agent.actor_logstd[0, 1].exp().item(), global_step
        )  # rlpilot

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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

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
                    b_obs[mb_inds], b_actions[mb_inds]
                )
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
        # rlpilot: grouped under 2_ppo and 5_perf (see LAYOUT); old_approx_kl dropped
        writer.add_scalar(
            "2_ppo/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("2_ppo/value_loss", v_loss.item(), global_step)
        writer.add_scalar("2_ppo/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("2_ppo/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("2_ppo/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("2_ppo/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("2_ppo/explained_variance", explained_var, global_step)
        bar.update(args.batch_size)  # rlpilot
        bar.set_postfix(
            sps=int(global_step / (time.time() - start_time)),
            ret=f"{np.mean([x['return'] for x in ended]):.0f}" if ended else "-",
            speed=f"{np.mean([x['mean_speed'] for x in ended]):.1f}" if ended else "-",
            best=f"{best:.3f}" if best >= 0 else "-",
            refresh=False,
        )
        writer.add_scalar(
            "5_perf/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if (
            global_step >= next_eval or iteration == args.num_iterations
        ):  # rlpilot: evaluate and checkpoint
            bar.set_description("eval")  # rlpilot
            best = run_eval(agent.cpu(), norm, args, global_step, run_dir, writer, best)
            bar.set_description("train")
            agent.to(device)
            next_eval += args.eval_every

    bar.close()  # rlpilot
    envs.close()
    writer.close()
