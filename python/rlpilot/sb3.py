"""Stable-Baselines3 VecEnv adapter over SpeedDreamsVecEnv (no extra processes or copies), and a PPO with
single-threaded rollouts.

SB3 wants per-env info dicts; non-empty ones are only built for envs whose episode ended, carrying
"terminal_observation", "TimeLimit.truncated", the Monitor-style "episode" dict (so SB3 logs
rollout/ep_rew_mean and ep_len_mean) and our per-episode stats.
"""

from __future__ import annotations

import time

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

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
    """PPO whose rollouts use one torch thread: the policy runs on a batch of n_envs observations per step,
    where waking a thread pool costs more than it saves (and the threads compete with the sim workers).
    The updates keep all threads. Measured +6% end to end with 5 workers. Saved models load with PPO.load.
    (Running the updates on MPS was slower: SB3 syncs to the CPU after every minibatch.)"""

    def collect_rollouts(self, *args, **kwargs):
        threads = th.get_num_threads()
        th.set_num_threads(1)
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            th.set_num_threads(threads)
