"""CpuRolloutPPO's GPU update must train exactly like SB3's own."""

import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from rlpilot.sb3 import CpuRolloutPPO


def trained(train, target_kl, device="cpu"):
    class Model(CpuRolloutPPO):
        pass

    Model.train = train
    model = Model("MlpPolicy", "Pendulum-v1", n_steps=64, batch_size=32, n_epochs=3, target_kl=target_kl, seed=3, device=device,
                  policy_kwargs=dict(net_arch=[16, 16]))
    model.learn(192)
    return model


@pytest.mark.parametrize("target_kl", [None, 1e-6])  # 1e-6: the early stop fires
def test_same_weights_as_sb3(target_kl):
    ours, theirs = trained(CpuRolloutPPO._train_on_gpu, target_kl), trained(PPO.train, target_kl)
    assert ours._n_updates == theirs._n_updates
    for a, b in zip(ours.policy.parameters(), theirs.policy.parameters()):
        assert th.equal(a, b)
    for key, value in theirs.logger.name_to_value.items():
        assert ours.logger.name_to_value[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key


@pytest.mark.skipif(not th.cuda.is_available(), reason="needs a GPU")
def test_updates_on_the_gpu_and_rollouts_on_the_cpu():
    model = trained(CpuRolloutPPO.train, None, "cuda")
    assert model.policy.device.type == "cuda" and model.device.type == "cuda"
    assert next(iter(model.policy.optimizer.state.values()))["exp_avg"].device.type == "cuda"
    assert np.isfinite(model.logger.name_to_value["train/loss"])
