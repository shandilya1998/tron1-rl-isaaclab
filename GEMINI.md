# GEMINI Code Companion Report

This document provides a detailed analysis of the `tron1-rl-isaaclab-cozum` repository, which is designed for training bipedal robot locomotion policies using Isaac Lab and RSL-RL.

## Repository Structure

The repository is organized into three main directories:

-   `exts/`: This directory contains the `bipedal_locomotion` extension, which is the core of the project. It defines the robot assets, environments, tasks, and MDPs (Markov Decision Processes).
-   `scripts/`: This directory contains the entry-point scripts for training and evaluating the policy.
-   `rsl_rl/`: As per the user's request, this directory is not used in the project, and the `rsl_rl` package installed in the Python environment is used instead.

### Key Files and Directories

-   `exts/bipedal_locomotion/bipedal_locomotion/`: The main source code for the Isaac Lab extension.
    -   `assets/`: Contains robot configurations (`pointfoot_cfg.py`, `solefoot_cfg.py`, `wheelfoot_cfg.py`) and USD files for the robot models.
    -   `tasks/locomotion/`: Defines the locomotion task.
        -   `cfg/`: Contains environment configurations for different robot types (PF, SF, WF for Point-Foot, Sole-Foot, and Wheel-Foot) and scenarios.
        -   `mdp/`: Defines the components of the Markov Decision Process:
            -   `rewards.py`: Reward functions.
            -   `observations.py`: Observation functions.
            -   `curriculums.py`: Curriculum learning configurations.
            -   `events.py`: Defines events that can happen during the simulation (e.g., pushing the robot).
            -   `commands/`: Defines the command generators for controlling the robot's target velocity and gait.
        -   `robots/`: Contains robot-specific environment configurations.
        -   `agents/`: Contains the configuration for the RSL-RL PPO agent.
-   `scripts/rsl_rl/`:
    -   `train.py`: The main script to start the training process.
    -   `play.py`: A script to evaluate a trained policy.
    -   `cli_args.py`: Defines command-line arguments for the training and evaluation scripts.

## Software Architecture

The architecture is based on Isaac Lab for the simulation and environment creation, and RSL-RL for the reinforcement learning algorithm.

1.  **Entry Point:** The training is initiated by running `scripts/rsl_rl/train.py`. This script uses `isaaclab.app.AppLauncher` to start the Isaac Sim application.

2.  **Environment Configuration:** The `train.py` script parses a task name from the command-line arguments and loads the corresponding environment configuration from `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/`. For example, the task `PF_TRON1A_flat` would load the configuration from `limx_pointfoot_env_cfg.py`.

3.  **Isaac Lab Environment:** An Isaac Lab environment is created using `gym.make()`. The environment is a `ManagerBasedRLEnv` from Isaac Lab, which manages the scene, actions, observations, and rewards.

4.  **RSL-RL Integration:** The Isaac Lab environment is wrapped with `RslRlVecEnvWrapper` to make it compatible with RSL-RL.

5.  **Training Loop:** An `OnPolicyRunner` from RSL-RL is instantiated with the wrapped environment and the agent configuration. The `runner.learn()` method is called to start the training loop, which collects rollouts from the environment and updates the policy using PPO.

### Class and Data Flow

-   `...EnvCfg` classes (e.g., `PFBaseEnvCfg` in `limx_pointfoot_env_cfg.py`): These are `configclass` objects that define the entire environment, including the robot, scene, observations, rewards, and terminations.
-   `ManagerBasedRLEnv`: The base class for the environment, which orchestrates the simulation.
-   `ObservationManager`, `RewardManager`, `ActionManager`: These managers are used within the `ManagerBasedRLEnv` to compute observations, rewards, and process actions.
-   `RslRlPpoAlgorithmMlpCfg`: A `configclass` that defines the hyperparameters for the PPO agent and the policy network architecture.
-   `OnPolicyRunner`: The main class from RSL-RL that drives the training process.

## Reward Definition and Formulation

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

## Logging Management

Logging is handled by the `train.py` script and the RSL-RL runner.

-   **Log Directory:** The `train.py` script creates a log directory for each experiment in `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>`.
-   **Configuration Logging:** Before starting the training, the script saves the environment and agent configuration files (`env.yaml`, `agent.yaml`, `env.pkl`, `agent.pkl`) in the log directory.
-   **RSL-RL Logging:** The `OnPolicyRunner` from RSL-RL handles the logging of training statistics. While not explicitly detailed in the provided files, RSL-RL typically logs data such as:
    -   Mean reward and individual reward terms.
    -   Episode length.
    -   PPO-specific metrics like value loss, policy loss, and entropy.
    -   This data is usually saved in a format that can be visualized with tools like TensorBoard.
-   **Video Recording:** If the `--video` flag is used, the `train.py` script wraps the environment with `gym.wrappers.RecordVideo` to save video recordings of the training process in the log directory.
-   **Checkpoints:** The RSL-RL runner saves model checkpoints (`.pt` files) periodically in the log directory. The frequency of saving is controlled by the `save_interval` parameter in the agent configuration.
