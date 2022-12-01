from publish.utils.constants import FORWARD, BACKWARD
from hive.agents.agent import Agent
from hive.envs.base import BaseEnv
from torchtyping import TensorType
from typing import Tuple
import ray
import torch

class RolloutWorker:
    def __init__(self, env: BaseEnv, agent: Agent, batch_size: int = 1):
        self.env, self.agent, self.batch_size = env, agent, batch_size
        self.horizon = self.env.horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def sample(self, batch_size: int = None) -> Tuple[
        TensorType['batch_size', 'horizon', 'ndim_times_side_len', int], # states
        TensorType['batch_size', 'horizon', int], # actions
        TensorType['batch_size', 'horizon', int], # backwards actions
        TensorType['batch_size', 'horizon', bool], # dones
        TensorType['batch_size', float], # rewards
    ]:
        batch_size = batch_size or self.batch_size
        all_states, all_dones, all_rewards, all_actions = \
            self._get_init_sample_tensors(batch_size)

        dones = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        states = self.env.reset()

        i = 0
        all_states[:, i] = states
        all_dones[:, i] = dones
        while not dones.all():
            with torch.no_grad():
                agent_states = self.agent.get_agent_state(
                    all_states,
                    all_actions,
                    all_dones,
                    all_rewards,
                    i
                )

                actions = self.agent.act(agent_states)

            actions[dones] = -1
            all_actions[:, i] = actions

            i += 1
            states, all_rewards, dones = self.env.step(actions)

            idx_to_insert = i if i != self.horizon else self.horizon - 1
            all_states[:, idx_to_insert] = states
            all_dones[:, idx_to_insert] = dones

        return (
            all_states,
            all_actions,
            self.env.get_backward_actions(all_states, all_actions),
            all_dones,
            all_rewards,
        )

    def _get_init_sample_tensors(self, batch_size: int = None) -> Tuple[
        TensorType['batch_size', 'horizon', 'ndim_times_side_len', int], # states
        TensorType['batch_size', 'horizon', int], # actions
        TensorType['batch_size', 'horizon', int], # backwards actions
        TensorType['batch_size', float], # rewards
        TensorType['batch_size', 'horizon', bool], # dones
    ]:
        all_states = torch.full(
            (batch_size, self.horizon, self.env.obs_dim),
            fill_value=-1,
            dtype=torch.float,
            device=self.device
        )

        all_dones = torch.ones(
            (batch_size, self.horizon),
            dtype=torch.int,
            device=self.device
        )

        all_rewards = torch.zeros(
            (batch_size),
            dtype=torch.float,
            device=self.device
        )

        all_actions = torch.full(
            (batch_size, self.horizon),
            fill_value=-1,
            dtype=torch.int,
            device=self.device
        )

        return all_states, all_dones, all_rewards, all_actions


    def sample_backward(
        self,
        terminal_states: TensorType['num_terminal_states', 'obs_dim'],
        num_to_sample_per_terminal_state: int = 1
    ) -> Tuple[
        TensorType['num_terminal_states', 'horizon', 'ndim_times_side_len', int], # states
        TensorType['num_terminal_states', 'horizon', int], # actions
        TensorType['num_terminal_states', 'horizon', int], # backwards actions
        TensorType['num_terminal_states', 'horizon', bool], # dones
        TensorType['num_terminal_states', float], # rewards
    ]:
        repeated_terminal_states = terminal_states.repeat_interleave(
            num_to_sample_per_terminal_state,
            dim=0
        )

        states, actions, back_actions, ep_lens, rewards = \
            self._do_backward_rollout(repeated_terminal_states)

        device = states.device

        # Roll all the states, actions, back actions, and create dones
        # before returning the results
        dones = []
        return_states, return_actions, return_back_actions = [], [], []

        ep_lens = ep_lens + 1
        n_samp = num_to_sample_per_terminal_state
        return_states = torch.cat([
            torch.cat([
                states[i * n_samp : (i + 1) * n_samp, -ep_lens[i * n_samp]:],
                terminal_states[i].repeat(n_samp, self.horizon - ep_lens[i * n_samp], 1)
            ], dim=1)
            for i in range(len(terminal_states))
        ], dim=0)

        return_actions = torch.cat([
            torch.cat([
                actions[i * n_samp : (i + 1) * n_samp, -ep_lens[i * n_samp]:],
                torch.full((n_samp, self.horizon - ep_lens[i * n_samp]), -1, device=device)
            ], dim=1)
            for i in range(len(terminal_states))
        ], dim=0)

        return_back_actions = torch.cat([
            torch.cat([
                back_actions[i * n_samp : (i + 1) * n_samp, -ep_lens[i * n_samp]:],
                torch.full((n_samp, self.horizon - ep_lens[i * n_samp]), -1, device=device)
            ], dim=1)
            for i in range(len(terminal_states))
        ], dim=0)

        dones = torch.cat([
            torch.cat([
                torch.zeros((n_samp, ep_lens[i * n_samp]), device=device),
                torch.ones((n_samp, self.horizon - ep_lens[i * n_samp]), device=device)
            ], dim=1)
            for i in range(len(terminal_states))
        ], dim=0)

        return (
            return_states,
            return_actions,
            return_back_actions,
            dones,
            rewards
        )

    def _do_backward_rollout(
        self,
        terminal_states: TensorType['num_terminal_states', 'obs_dim']
    ) -> Tuple[
        TensorType['num_terminal_states', 'horizon', 'ndim_times_side_len', int], # states
        TensorType['num_terminal_states', 'horizon', int], # actions
        TensorType['num_terminal_states', 'horizon', int], # backwards actions
        TensorType['num_terminal_states', int], # episode_lens
        TensorType['num_terminal_states', float], # rewards
    ]:
        batch_size = len(terminal_states)
        states = terminal_states
        all_states, all_dones, all_rewards, all_actions = \
            self._get_init_sample_tensors(batch_size)

        all_back_actions = all_actions.clone()
        dones = self.env.is_origin_state(states)

        # We define episode lengths here by the number of
        # transitions in a trajectory
        episode_lens = torch.zeros_like(dones).long()

        i = self.horizon - 1
        all_states[:, i] = states
        all_dones[:, i] = True
        all_actions[:, i] = self.env.done_action
        while not dones.all():
            with torch.no_grad():
                agent_states = self.agent.get_agent_state(
                    all_states,
                    all_actions,
                    all_dones,
                    all_rewards,
                    i
                )

                back_actions = self.agent.act(agent_states, mode=BACKWARD)

            back_actions[dones] = -1
            all_back_actions[:, i] = back_actions
            episode_lens[~dones] = episode_lens[~dones] + 1

            i -= 1
            states = self.env.step_backward(back_actions, states)

            all_states[:, i] = states
            all_actions[~dones, i] = self.env.get_action_ids(
                all_states[~dones, i],
                all_states[~dones, i + 1],
                mode=FORWARD
            ).int()

            dones = self.env.is_origin_state(states)

        return (
            all_states,
            all_actions,
            all_back_actions,
            episode_lens,
            self.env.get_rewards(terminal_states),
        )


    def set_agent_weights(self, weights):
        self.agent.load_state_dict(weights)

@ray.remote
class RolloutWorkerRemote(RolloutWorker):
    pass
