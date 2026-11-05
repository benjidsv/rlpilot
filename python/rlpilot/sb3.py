"""Stable-Baselines3 VecEnv adapter over SpeedDreamsVecEnv (no extra processes or copies), and a PPO with
single-threaded rollouts.

SB3 wants per-env info dicts; non-empty ones are only built for envs whose episode ended, carrying
"terminal_observation", "TimeLimit.truncated", the Monitor-style "episode" dict (so SB3 logs
rollout/ep_rew_mean and ep_len_mean) and our per-episode stats.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from torch.nn import functional as F

from .vec_env import SpeedDreamsVecEnv


class SB3VecEnv(VecEnv):
    def __init__(self, env: SpeedDreamsVecEnv):
        self.env = env
        super().__init__(env.num_envs, env.single_observation_space, env.single_action_space)
        self._actions = None
        self.env_seconds = 0.0  # wall time spent inside env.step (for env-only throughput)
        self.env_steps = 0

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        t = time.perf_counter()
        obs, rew, term, trunc, info = self.env.step(self._actions)
        dones = term | trunc
        infos = [{} for _ in range(self.num_envs)]
        if info:
            ep, st = info["episode"], info["stats"]
            reasons = self.env.term_reasons
            for i in np.flatnonzero(dones):
                stats = {k: float(v[i]) for k, v in st.items() if k != "term_reason"}
                r = int(st["term_reason"][i])
                stats["end"] = reasons[r] if r >= 0 else "time_limit"
                infos[i] = {
                    "terminal_observation": info["final_obs"][i],
                    "TimeLimit.truncated": bool(trunc[i]),
                    "episode": {"r": float(ep["r"][i]), "l": int(ep["l"][i]), "t": float(ep["t"][i])},
                    "stats": stats,
                }
        self.env_seconds += time.perf_counter() - t
        self.env_steps += self.num_envs
        return obs, rew, dones, infos

    def close(self):
        self.env.close()

    # The remaining VecEnv API is about per-env wrappers, which this env does not have.
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.env, method_name)(*args, **kwargs)] * len(self._get_indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def get_images(self):
        return [None] * self.num_envs


class CpuRolloutPPO(PPO):
    """PPO whose rollouts always run on the CPU, with one torch thread: the policy runs on a batch of n_envs observations per
    step, where waking a thread pool costs more than it saves (and the threads compete with the sim workers). The updates keep
    all threads, or run on `device` if it's a GPU. Measured on a 7700X + RTX 4070 Ti, per ~10k samples: CPU only 0.42 s rollout +
    0.43 s update, GPU only 0.62 s + 0.18 s. Saved models load with PPO.load. (Updates on MPS were slower: SB3 syncs to the CPU
    after every minibatch.)"""

    def collect_rollouts(self, *args, **kwargs):
        threads, update_device = th.get_num_threads(), self.device
        th.set_num_threads(1)
        cpu = th.device("cpu")
        # .to() keeps the parameter objects, so the optimizer stays valid, and its state stays on update_device for train().
        # Evaluations run in this method's callbacks, so they're on the CPU too.
        self.policy.to(cpu)
        self.device = cpu
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            self.device = update_device
            self.policy.to(update_device)
            th.set_num_threads(threads)

    def train(self) -> None:
        if self.device.type == "cpu" or self.clip_range_vf is not None or not isinstance(self.action_space, spaces.Box):
            return super().train()
        return self._train_on_gpu()

    def _train_on_gpu(self) -> None:
        """PPO.train with the same updates and the same random draws, minus its GPU overhead: the rollout goes to the GPU once
        rather than per minibatch, and only the KL early stop waits for the GPU, rather than five values per minibatch."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        b = self.rollout_buffer
        n = b.buffer_size * b.n_envs
        if not b.generator_ready:
            for name in ("observations", "actions", "values", "log_probs", "advantages", "returns"):
                b.__dict__[name] = b.swap_and_flatten(b.__dict__[name])
            b.generator_ready = True
        obs, actions = b.to_torch(b.observations), b.to_torch(b.actions.astype(np.float32, copy=False))
        old_log_probs, all_advantages, returns = (b.to_torch(x.flatten()) for x in (b.log_probs, b.advantages, b.returns))

        logged = {"entropy_loss": [], "policy_gradient_loss": [], "value_loss": [], "clip_fraction": []}
        continue_training = True
        for _ in range(self.n_epochs):
            approx_kl_divs = []
            order = th.as_tensor(np.random.permutation(n), device=self.device)  # as RolloutBuffer.get
            for start in range(0, n, self.batch_size):
                batch = order[start:start + self.batch_size]
                old_log_prob = old_log_probs[batch]
                values, log_prob, entropy = self.policy.evaluate_actions(obs[batch], actions[batch])
                advantages = all_advantages[batch]
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - old_log_prob)
                policy_loss = -th.min(advantages * ratio, advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                value_loss = F.mse_loss(returns[batch], values.flatten())
                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                with th.no_grad():
                    logged["policy_gradient_loss"].append(policy_loss.detach())
                    logged["value_loss"].append(value_loss.detach())
                    logged["entropy_loss"].append(entropy_loss.detach())
                    logged["clip_fraction"].append(th.mean((th.abs(ratio - 1) > clip_range).float()))
                    log_ratio = log_prob - old_log_prob
                    approx_kl_divs.append(th.mean((th.exp(log_ratio) - 1) - log_ratio))
                if self.target_kl is not None and approx_kl_divs[-1].item() > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break

        for name, values in logged.items():
            self.logger.record(f"train/{name}", th.stack(values).mean().item())
        self.logger.record("train/approx_kl", th.stack(approx_kl_divs).mean().item())
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_variance(b.values.flatten(), b.returns.flatten()))
        self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)


def resume_ppo(directory: str | Path, env: VecEnv, device: str, settings: dict) -> CpuRolloutPPO:
    """Continue from a directory with model.zip and vecnormalize.pkl. `settings` are the keyword arguments of a new PPO:
    they all replace the saved values, also in VecNormalize (gamma), so what a run logs is what it uses. Only the
    network comes from the checkpoint, and a different policy_kwargs is refused."""
    directory = Path(directory)
    venv = VecNormalize.load(str(directory / "vecnormalize.pkl"), env)
    venv.training = True
    venv.gamma = settings["gamma"]
    # custom_objects replaces the saved values before the schedules are rebuilt; the saved ones are closures
    replaced = {k: settings[k] for k in ("learning_rate", "clip_range")}
    rest = {k: v for k, v in settings.items() if k not in replaced}
    return CpuRolloutPPO.load(directory / "model.zip", env=venv, device=device,
                              custom_objects={"lr_schedule": None} | replaced, **rest)
