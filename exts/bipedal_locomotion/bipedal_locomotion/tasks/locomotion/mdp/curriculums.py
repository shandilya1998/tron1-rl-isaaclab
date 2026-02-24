from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    env.command_manager.num_envs
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def compute_range(config, current_step, total_steps):
    start_step = config["start_frac"] * total_steps
    end_step = config["end_frac"] * total_steps

    if current_step < start_step:
        return tuple(config["min_range"]), 0.0
    elif current_step < end_step:
        alpha = (
            (current_step - start_step) / (end_step - start_step)
            if end_step > start_step
            else 1.0
        )
        start_range = config["min_range"]
        min_v = start_range[0] + alpha * (config["max_range"][0] - start_range[0])
        max_v = start_range[1] + alpha * (config["max_range"][1] - start_range[1])
        return (min_v, max_v), alpha
    else:
        return tuple(config["max_range"]), 1.0


def velocity_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    max_steps: int,
    x_config: dict[str, Any],
    y_config: dict[str, Any],
    z_config: dict[str, Any],
) -> torch.Tensor:
    """Curriculum that slowly increases the commanded velocity ranges based on training progress.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        command_name: The name of the command term.
        max_steps: Total number of steps for training (max_iterations * num_steps_per_env).
        x_config: Dictionary with 'start_frac', 'end_frac', 'min_range', 'max_range'.
        y_config: Dictionary with 'start_frac', 'end_frac', 'min_range', 'max_range'.
        z_config: Dictionary with 'start_frac', 'end_frac', 'min_range', 'max_range'.

    Returns:
        torch.Tensor: A tensor containing the current progress (0 to 1) for each component.
    """
    current_step = env.common_step_counter
    command_term = env.command_manager.get_term(command_name)

    # Linear velocity x
    range_x, progress_x = compute_range(x_config, current_step, max_steps)
    command_term.cfg.ranges.lin_vel_x = range_x

    # Linear velocity y
    range_y, progress_y = compute_range(y_config, current_step, max_steps)
    command_term.cfg.ranges.lin_vel_y = range_y

    # Angular velocity z
    range_z, progress_z = compute_range(z_config, current_step, max_steps)
    command_term.cfg.ranges.ang_vel_z = range_z

    progress = math.ceil(progress_x) or math.ceil(progress_y) or math.ceil(progress_z)

    if progress:
        return torch.ones(1)
    else:
        return torch.zeros(1)
