# ARCHITECTURE.md: Isaac Lab & RSL-RL Task Integration

This project utilizes a modular, inheritance-based configuration system to define bipedal locomotion tasks, bridging **Isaac Lab** (simulation/MDP) with **RSL-RL** (RL algorithms/runners).

## 1. Directory Roles & Responsibilities

| Directory/File | Responsibility |
| :--- | :--- |
| [`scripts/rsl_rl/train.py`](scripts/rsl_rl/train.py) | **Entry Point:** Parses `--task`, initializes the simulator, and invokes the RL Runner. |
| [`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/`](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/) | **MDP Logic:** Definition of Reward functions, Observation terms, and Events. |
| [`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py`](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py) | **MDP Templates:** Defines the robot-specific observation, reward, and action spaces. |
| [`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_env_cfg.py`](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_env_cfg.py) | **Scenario & Asset Config:** Defines USD asset paths, joint positions, and terrain scenarios (Flat, Rough, etc.). |
| [`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py) | **RL Hyperparameters:** Configures PPO algorithm settings and Actor-Critic network dimensions. |
| [`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py`](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py) | **The Registry:** Maps Task IDs to environment and agent configuration classes. |
| [`scripts/rsl_rl/play.py`](scripts/rsl_rl/play.py) | **Evaluation:** Script to load a trained checkpoint and visualize the policy. |
| [`scripts/rsl_rl/cli_args.py`](scripts/rsl_rl/cli_args.py) | **Arguments:** Centralized definition of command-line arguments for training and playback. |

## 2. Class Hierarchy & Relationships

The configuration follows a hierarchical structure using Isaac Lab's `@configclass` decorator.

### Environment Configuration Chain (Sole-Foot)
1.  **`ManagerBasedRLEnvCfg`** (external: `isaaclab.envs`): Foundation class.
2.  **`SFEnvCfg`** ([`limx_base_env_cfg.py`](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py)): Robot-specific MDP template (Scene, Rewards, Observations, Actions).
3.  **`SFBaseEnvCfg`** ([`limx_solefoot_env_cfg.py`](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_env_cfg.py)): Foundation for scenarios. Defines `SOLEFOOT_CFG` asset and default joint positions.
4.  **`SFBlindFlatEnvCfg`** ([`limx_solefoot_env_cfg.py`](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_env_cfg.py)): Final leaf class. Overrides parent to disable height scanners and set "plane" terrain.

## 3. Reward Definition and Formulation

The reward functions are defined in `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/rewards.py`. The total reward is a weighted sum of several terms, designed to encourage stable and efficient locomotion.

### Key Reward Terms:

-   **Velocity Tracking:**
    -   `rew_lin_vel_xy`: Rewards matching the target linear velocity in the x-y plane.
    -   `rew_ang_vel_z`: Rewards matching the target angular velocity around the z-axis.
-   **Penalties for undesired behavior:**
    -   `pen_lin_vel_z`: Penalizes vertical velocity.
    -   `pen_ang_vel_xy`: Penalizes angular velocity in the x-y plane.
    -   `pen_action_rate`: Penalizes large changes in actions between consecutive timesteps.
    -   `pen_flat_orientation`: Penalizes deviation from a flat orientation.
    -   `pen_undesired_contacts`: Penalizes contacts with parts of the robot other than the feet.
    -   `joint_powers_l1`: Penalizes high joint power consumption.
-   **Gait and Foot Placement:**
    -   `GaitReward`: A custom reward class that encourages a specific foot contact pattern based on a given gait command (frequency, offset, duration). It uses a von Mises distribution to create a smooth reward signal for being in the correct phase of the gait.
    -   `foot_landing_vel`: Penalizes high foot velocities upon landing.
    -   `feet_distance`: Penalizes if the distance between feet is too small or too large.
    -   `nominal_foot_position`: Rewards keeping the feet at a nominal position relative to the base.
-   **Stability and balance:**
    -   `unbalance_feet_air_time`: Penalizes large variance in the air time of the feet.
    -   `base_height_rough_l2`: Penalizes deviation from a target height, even on rough terrain.
    -   `stay_alive`: A constant reward for not terminating the episode.

The weights for these reward terms are specified in the environment configuration files (e.g., `PFBlindStairEnvCfg` in `limx_pointfoot_env_cfg.py`).

## 4. Data Flow & Environment Management

The project uses a "Manager-Based" architecture within the `ManagerBasedRLEnv` to orchestrate the simulation loop.

-   **`ObservationManager`**: Computes the policy and critic observations based on the `ObservationsCfg`.
-   **`RewardManager`**: Calculates individual reward terms and their weighted sum as defined in `RewardsCfg`.
-   **`ActionManager`**: Processes the actions output by the policy and applies them to the robot's actuators.
-   **`RslRlVecEnvWrapper`**: A critical shim that wraps the Isaac Lab environment to make it compatible with the expected input/output format of the RSL-RL library.

### Training Loop
1.  `train.py` creates the environment via `gym.make(task_id)`.
2.  The environment is wrapped by `RslRlVecEnvWrapper`.
3.  An `OnPolicyRunner` (or `HIMOnPolicyRunner`) is instantiated.
4.  `runner.learn()` is called, which handles data collection (rollouts) and PPO updates.

## 5. Environment Registration

Tasks are registered in [`robots/__init__.py`](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py) using `gym.register`.

```python
gym.register(
    id="Isaac-Limx-SF-HIM-v0",             # CLI Task Name
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": SFHIMBlindFlatEnvCfg,  # From limx_solefoot_env_cfg.py
        "rsl_rl_cfg_entry_point": SF_TRON1AFlatPPORunnerCfg(), # From limx_rsl_rl_ppo_cfg.py
    },
)
```

## 6. Steps to Create a New Task

1.  **Define MDP Logic:** If unique observations or rewards are needed, update robot `limx_base_env_cfg.py`.
2.  **Create Scenario Class:** In `limx_solefoot_env_cfg.py`, define a class inheriting from `SFBaseEnvCfg`. Use `__post_init__` for overrides.
3.  **Configure Agent:** Define a `RunnerCfg` in `limx_rsl_rl_ppo_cfg.py` for PPO hyperparameters.
4.  **Register Task:** Add a `gym.register` block in `robots/__init__.py`.
5.  **Select Runner:** Update `scripts/rsl_rl/train.py` if a custom runner (e.g., [`HIMOnPolicyRunner`](himloco/himloco/runners/him_on_policy_runner.py)) is required.

## 6. HIM Architecture Summary
*   **Observations:** Requires `policy` (1-step) and `obsHistory` (e.g., 25-step) groups.
*   **Format:** `flatten_history_dim` must be `True`.
*   **Network:** Uses [`HIMActorCritic`](himloco/himloco/modules/him_actor_critic.py) with a dedicated estimator.
*   **Toggle:** Switch via `--task Isaac-Limx-SF-HIM-v0 --policy_type HIMPPO`.
