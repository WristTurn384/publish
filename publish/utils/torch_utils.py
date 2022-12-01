from publish.utils.constants import LOG_ZERO
from functools import reduce
from operator import mul
from torchtyping import TensorType
from typing import Tuple, List, Dict
from enum import Enum
from contextlib import nullcontext
import torch
import math

def build_mlp(
    obs_dim: int,
    action_dim: int,
    hidden_layer_dim: int,
    num_hidden_layers: int,
    should_model_state_flow: bool = False,
    should_param_backward_policy: bool = False,
    activation: torch.nn.Module = torch.nn.LeakyReLU()
) -> torch.nn.Module:
    out_dim = (
        action_dim +
        (should_param_backward_policy * (action_dim - 1)) +
        int(should_model_state_flow)
    )

    layer_dims = [obs_dim]
    layer_dims.extend([hidden_layer_dim] * num_hidden_layers)
    layer_dims.append(out_dim)

    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
        layers.append(activation)

    return torch.nn.Sequential(*layers)

def observation_to_multi_one_hot(
    observations: TensorType,
    side_length: int,
    num_dims: int,
    token_val: int = -1
) -> TensorType:
    new_dim = side_length * num_dims
    new_shape = tuple(list(observations.shape)[:-1] + [new_dim])
    if len(observations.shape) == 1:
        new_shape = (1, new_dim)

    device = observations.device
    multi_onehot = torch.zeros(
        new_shape,
        dtype=torch.float32,
        device=device
    )

    if len(observations.shape) == 1:
        tile_shape = (1, 1)
    else:
        tile_shape = tuple(observations.shape[:-1]) + (1,)

    idx_arr_pre = torch.tile(torch.arange(num_dims) * side_length, tile_shape).to(
        device=device
    )

    dtype = observations.dtype
    token_obs_idx = torch.isclose(observations, torch.tensor(token_val, dtype=dtype))

    add_obs = observations.clone()
    add_obs[token_obs_idx] = 0

    idx_arr_pre = idx_arr_pre + add_obs

    idx_arrs = _to_multidim_idx_arrs(idx_arr_pre)
    multi_onehot[idx_arrs] = 1

    multi_onehot[token_obs_idx.all(axis=-1)] = token_val

    if multi_onehot.shape[0] == 1:
        multi_onehot = multi_onehot.reshape(-1)

    return multi_onehot

def _to_multidim_idx_arrs(idx_arr: TensorType) -> Tuple[TensorType, TensorType]:
    idxs, shape = [], idx_arr.shape
    for i in reversed(range(len(shape) - 1)):
        num_later_dim_repeats = reduce(mul, shape[i + 1:], 1)
        num_earlier_dim_repeats = reduce(mul, shape[:i], 1)

        idx_pre = torch.cat([
            torch.full((num_later_dim_repeats,), fill_value=j)
            for j in range(shape[i])
        ])

        idxs.append(idx_pre.repeat(num_earlier_dim_repeats))

    return tuple(reversed(idxs)) + (idx_arr.flatten(),)

def get_grid_one_hot_int_encoder_mask(
    num_dims: int,
    side_length: int
) -> TensorType['num_dims_times_side_length']:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_mask_part1 = torch.arange(
        side_length,
        device=device
    ).repeat(num_dims)

    encoder_mask_part2 = (
        side_length ** torch.arange(
            num_dims,
            device=device
        ).flip(0).repeat_interleave(side_length)
    )

    return encoder_mask_part1 * encoder_mask_part2

def get_int_encoded_terminal_states(
    states: TensorType['batch_size', 'horizon', 'obs_dim'],
    dones: TensorType['batch_size', 'horizon'],
    side_length: int,
    num_dims: int
) -> TensorType['batch_size']:
    encoder_mask = get_grid_one_hot_int_encoder_mask(num_dims, side_length)

    first_done_idxs = get_first_done_idx(dones)
    return (
        states[torch.arange(len(states)), first_done_idxs] * encoder_mask
    ).sum(dim=1).long()

def get_first_done_idx(
    dones: TensorType['batch_size', 'horizon']
) -> TensorType['batch_size']:
    idx = torch.arange(
        start=dones.shape[1],
        end=0,
        step=-1,
        device=dones.device
    )

    tmp = dones * idx
    return tmp.argmax(dim=1)


class TrajectoryInfoTypes(Enum):
    LOSS       = 0
    P_FORWARD  = 1
    P_BACKWARD = 2


def get_trajectory_info(
    agent,
    states: TensorType['num_trajs', 'horizon', 'obs_dim'],
    fwd_actions: TensorType['num_trajs', 'horizon'],
    back_actions: TensorType['num_trajs', 'horizon'],
    dones: TensorType['num_trajs', 'horizon'],
    rewards: TensorType['num_trajs', 'horizon'],
    batch_size: int,
    compute_options: List[TrajectoryInfoTypes] = [TrajectoryInfoTypes.LOSS]
) -> Tuple[TensorType['num_trajs'], TensorType['num_trajs']]:
    actions, back_actions = fwd_actions.clone(), back_actions.clone()
    all_losses = torch.empty(len(states), device=states.device)
    all_pf = torch.empty(len(states), device=states.device)
    all_pb = torch.empty(len(states), device=states.device)
    with torch.no_grad():
        for i in range(math.ceil(len(states) / batch_size)):
            start_idx, end_idx = i * batch_size, (i + 1) * batch_size
            loss_input_dict = {
                'states': states[start_idx:end_idx],
                'fwd_actions': actions[start_idx:end_idx],
                'back_actions': back_actions[start_idx:end_idx],
                'dones': dones[start_idx:end_idx],
                'rewards': rewards[start_idx:end_idx],
                'do_regularization': False,
            }

            if TrajectoryInfoTypes.LOSS in compute_options:
                all_losses[start_idx:end_idx] = agent.loss(
                    loss_input_dict,
                    reduction='identity'
                )

            if TrajectoryInfoTypes.P_FORWARD in compute_options:
                all_pf[start_idx:end_idx] = agent.get_log_pf(
                    loss_input_dict['states'],
                    loss_input_dict['fwd_actions']
                ).sum(dim=1).exp()

            if TrajectoryInfoTypes.P_BACKWARD in compute_options:
                all_pb[start_idx:end_idx] = agent.get_log_pb(
                    loss_input_dict['states'],
                    loss_input_dict['back_actions']
                ).sum(dim=1).exp()

    return all_losses, all_pf, all_pb


def get_losses_and_pb(
    agent,
    states: TensorType['num_trajs', 'horizon', 'obs_dim'],
    fwd_actions: TensorType['num_trajs', 'horizon'],
    back_actions: TensorType['num_trajs', 'horizon'],
    dones: TensorType['num_trajs', 'horizon'],
    rewards: TensorType['num_trajs', 'horizon'],
    batch_size: int,
    requires_grad: bool = False
) -> Tuple[TensorType['num_trajs'], TensorType['num_trajs']]:
    actions, back_actions = fwd_actions.clone(), back_actions.clone()
    all_losses = torch.empty(len(states), device=states.device)
    all_pb = torch.empty(len(states), device=states.device)

    context = nullcontext if requires_grad else torch.no_grad
    with context():
        for i in range(math.ceil(len(states) / batch_size)):
            start_idx, end_idx = i * batch_size, (i + 1) * batch_size
            loss_input_dict = {
                'states': states[start_idx:end_idx],
                'fwd_actions': actions[start_idx:end_idx],
                'back_actions': back_actions[start_idx:end_idx],
                'dones': dones[start_idx:end_idx],
                'rewards': rewards[start_idx:end_idx],
                'do_regularization': False,
            }

            all_losses[start_idx:end_idx] = agent.loss(
                loss_input_dict,
                reduction='identity'
            )

            all_pb[start_idx:end_idx] = agent.get_log_pb(
                loss_input_dict['states'],
                loss_input_dict['back_actions']
            ).sum(dim=1).exp()

    return all_losses, all_pb

def get_p_given_terminal_state(
    all_p: TensorType['num_trajs'],
    num_terminal_states: int,
    int_encoded_states: TensorType['num_trajs'],
) -> TensorType['num_trajs']:
    px_marginal = torch.zeros(
        num_terminal_states,
        device=int_encoded_states.device
    )

    px_marginal = px_marginal.index_add(
        dim=0,
        index=int_encoded_states,
        source=all_p
    )

    return all_p / px_marginal[int_encoded_states]

def get_log_rewards(update_infos: Dict[str, object]) -> TensorType['batch_size']:
    log_rewards, rewards_pre = None, update_infos['rewards']
    is_zero_ind = torch.isclose(
        torch.tensor(0.0, device=rewards_pre.device),
        rewards_pre
    )

    if update_infos.get('mask_reward_close_to_zero', True):
        log_reward_pre = torch.nan_to_num(rewards_pre.log())
        log_rewards = (is_zero_ind * LOG_ZERO) + (~is_zero_ind * log_reward_pre)

    else:
        log_rewards = ((is_zero_ind * 1e-10) + (~is_zero_ind * rewards_pre)).log()

    return log_rewards

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
