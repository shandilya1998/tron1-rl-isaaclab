"""Utilities for hot-swapping robot morphologies during training.

These helpers implement the 10-step respawn sequence required to replace all
robot articulations in an IsaacLab ``ManagerBasedRLEnv`` mid-training, and
the subsequent patching of ``IdentifiedActuator`` tensor attributes to reflect
per-individual actuator parameters.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv


def respawn_robots(env: ManagerBasedRLEnv, new_usd_paths: list[str]) -> None:
    """Replace all robot articulations with new USD variants.

    The function implements the full 10-step respawn sequence:

    1. ``sim.stop()``  — deactivate Fabric so USD attributes can be edited.
    2. Delete existing robot prim trees from the stage.
    3. Spawn new articulations from *new_usd_paths* (round-robin per env).
    4. Re-register contact / ray-cast sensor callbacks.
    5. ``sim.reset()``  — reactivate Fabric and cook physics.
    6. Create new ``Articulation`` objects for the env scene.
    7. Patch ``scene._articulations`` so the robot reference is updated.
    8. Re-bind ``ActionTerm._asset`` references.
    9. Re-bind ``EventManager`` term asset references.
    10. ``env.reset()``  — reset all environments to initial state.

    Args:
        env: The unwrapped ``ManagerBasedRLEnv`` instance (not the
            ``RslRlVecEnvWrapper``).
        new_usd_paths: List of USD file paths, one per individual.  Environments
            are assigned designs round-robin:
            ``individual_idx = env_idx % len(new_usd_paths)``.
    """

    sim = env.sim
    scene = env.scene
    num_envs: int = env.num_envs

    # ------------------------------------------------------------------
    # Step 1: Stop simulation (deactivate Fabric)
    # ------------------------------------------------------------------
    sim.stop()

    # ------------------------------------------------------------------
    # Step 2: Delete existing robot prims
    # ------------------------------------------------------------------
    # sim_utils.delete_prim fires the SimulationManager PRIM_DELETION event,
    # which triggers AssetBase._on_prim_deletion() → _clear_callbacks() on
    # the old articulation and any sensors whose prim paths are children of
    # the deleted path.  Raw stage.RemovePrim() does not fire that event.
    for env_idx in range(num_envs):
        prim_path = scene.articulations["robot"].cfg.prim_path.format(env_id=env_idx)
        sim_utils.delete_prim(prim_path)

    # ------------------------------------------------------------------
    # Step 3: Spawn new articulations (round-robin by individual)
    # ------------------------------------------------------------------
    # spawn_multi_usd_file wraps all m×k spawns in a single Sdf.ChangeBlock
    # transaction, applies rigid_props/articulation_props/contact-sensor
    # settings from the original spawn cfg, and sets the carb flag
    # /isaaclab/spawn/multi_assets=True required for heterogeneous envs.
    old_spawn_cfg = scene.articulations["robot"].cfg.spawn
    new_spawn_cfg = sim_utils.MultiUsdFileCfg(
        usd_path=new_usd_paths,
        random_choice=False,
        rigid_props=old_spawn_cfg.rigid_props,
        articulation_props=old_spawn_cfg.articulation_props,
        activate_contact_sensors=old_spawn_cfg.activate_contact_sensors,
    )
    sim_utils.spawn_multi_usd_file(
        prim_path=scene.env_regex_ns + "/Robot",
        cfg=new_spawn_cfg,
        replicate_physics=False,  # required: envs carry heterogeneous designs
    )

    # ------------------------------------------------------------------
    # Step 4: Re-register sensor callbacks
    # ------------------------------------------------------------------
    # _clear_callbacks() in Step 2 (at sim_utils.delete_prims(..)) deregistered
    # each sensor's PLAY subscription.  sim.reset() fires PLAY internally, so
    # sensors must be re-subscribed here — before reset() — or root_physx_view
    # is never rebuilt.
    # source: sensor_base.py _register_callbacks(), _clear_callbacks()
    for sensor in scene._sensors.values():
        sensor._register_callbacks()

    # ------------------------------------------------------------------
    # Step 5: Reset simulation (reactivate Fabric, cook physics)
    # ------------------------------------------------------------------
    sim.reset()

    # ------------------------------------------------------------------
    # Step 6–7: Re-create Articulation object and patch scene dict
    # ------------------------------------------------------------------
    # Build a new cfg pointing at the new USD variants so the Articulation's
    # spawn reference is not stale.  PLAY already fired in Step 5, so
    # _initialize_impl() runs immediately inside initialize().
    new_robot_cfg = scene.articulations["robot"].cfg.replace(
        prim_path=scene.env_regex_ns + "/Robot",
        spawn=new_spawn_cfg,
    )
    new_articulation = Articulation(new_robot_cfg)
    new_articulation.initialize(scene.env_regex_ns)

    scene._articulations["robot"] = new_articulation  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Step 8: Re-bind ActionTerm._asset
    # ------------------------------------------------------------------
    # ActionTerm.__init__ caches self._asset at construction; apply_actions()
    # calls self._asset.set_joint_position_target() — stale after swap.
    # _joint_ids / _num_joints are unchanged (same joint topology).
    # Re-read _offset from new articulation in case equilibrium pose changed.
    for term in env.action_manager._terms.values():  # type: ignore[attr-defined]
        term._asset = new_articulation
        if hasattr(term, "_offset") and getattr(term.cfg, "use_default_offset", False):
            term._offset = new_articulation.data.default_joint_pos[
                :, term._joint_ids
            ].clone()

    # ------------------------------------------------------------------
    # Step 9: Re-bind EventManager term asset references
    # ------------------------------------------------------------------
    # Class-based ManagerTermBase subclasses cache self.asset at __init__.
    # Function-based event terms call env.scene[name] per invocation — skip those.
    event_manager = getattr(env, "event_manager", None)
    if event_manager is not None:
        for mode_terms in event_manager._terms.values():  # type: ignore[attr-defined]
            for term in mode_terms:
                if hasattr(term, "asset"):
                    term.asset = new_articulation

    # ------------------------------------------------------------------
    # Step 10: Reset all environments
    # ------------------------------------------------------------------
    env.reset()


def apply_actuator_params(
    env: ManagerBasedRLEnv,
    actuator_params_list: list[dict[str, dict]],
) -> None:
    """Patch ``IdentifiedActuator`` tensor attributes per environment.

    Each environment is assigned an individual round-robin; its actuator
    tensor rows are overwritten with the corresponding design's parameter
    values.

    Args:
        env: The unwrapped ``ManagerBasedRLEnv``.
        actuator_params_list: List of actuator-param dicts, one per individual.
            Each dict maps actuator group name → scalar overrides.
    """
    num_envs: int = env.num_envs
    num_individuals = len(actuator_params_list)
    articulation = env.scene.articulations["robot"]

    for group_name, actuator in articulation.actuators.items():
        for env_idx in range(num_envs):
            individual_idx = env_idx % num_individuals
            overrides = actuator_params_list[individual_idx].get(group_name, {})

            for attr_name, value in overrides.items():
                tensor_attr = getattr(actuator, attr_name, None)
                if tensor_attr is None or not isinstance(tensor_attr, torch.Tensor):
                    continue
                # tensor_attr shape: (num_envs, num_joints_in_group)
                tensor_attr[env_idx, :] = value
