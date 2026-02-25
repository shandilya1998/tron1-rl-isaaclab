# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.modules import ActorCritic

from himloco.modules.him_estimator import HIMEstimator, get_activation


class HIMActorCritic(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ):

        super(HIMActorCritic, self).__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization,
            critic_obs_normalization,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            noise_std_type,
            state_dependent_std,
            **kwargs,
        )

        # Estimator
        num_one_step_obs = obs["policy"].shape[-1]
        self.history_size = int(obs["obsHistory"].shape[-1] / num_one_step_obs)
        assert self.history_size * num_one_step_obs == obs["obsHistory"].shape[-1]
        self.estimator = HIMEstimator(
            temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs
        )
        self.estimator.actor_obs_normalizer = self.actor_obs_normalizer
        self.estimator.critic_obs_normalizer = self.critic_obs_normalizer

        print(f"Estimator: {self.estimator.encoder}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_we/ights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict):
        obs_history = torch.flatten(
            self.actor_obs_normalizer(obs["obsHistory"]), start_dim=1
        )
        actor_obs = self.get_actor_obs(obs["policy"])
        actor_obs = self.actor_obs_normalizer(actor_obs)
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        actor_input = torch.cat((actor_obs, vel, latent), dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        obs_history = torch.flatten(
            self.actor_obs_normalizer(obs["obsHistory"]), start_dim=1
        )
        vel, latent = self.estimator(obs_history)
        actions_mean = self.actor(torch.cat(actor_obs, vel, latent), dim=-1)
        return actions_mean
