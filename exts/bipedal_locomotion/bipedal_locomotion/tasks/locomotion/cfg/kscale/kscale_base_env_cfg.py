import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.mdp.curriculums import reduce_tracking_rewards_std

# This file is a direct port of cfg/SF/brs_base_env_cfg.py (the SD_BRS1 env cfg) onto
# the kscale robot: same reward/termination/curriculum structure (proven to train a
# similar 6-DOF-per-leg biped), retargeted to kscale's URDF link and joint names. See
# kscale_identified_cfg.py for the actuator gains and standing pose, both of which are
# UNVERIFIED PLACEHOLDERS pending a first look at the robot in sim.
#
# Name mapping used throughout this file (kscale URDF names on the right):
#   Part_Torso        -> assy_formfg___kd_b_102b_torso_btm
#   Link6[LR] (feet)  -> foot_6061 (right) / foot_6061_2 (left), matched as "foot_6061.*"
#   Link[1-5][LR]     -> kd_d_102[rl]_6061 (hip pitch), kd_d_201r_6061 / rs03 (hip roll,
#                        the left part is named rs03 in the source CAD -- confirmed by
#                        matching mass, 2.434 kg, against its mirrored right counterpart),
#                        kd_d_301[rl]_6061 (hip yaw / thigh), kd_d_401[rl]_6061 (knee /
#                        shank), arb_uj111_cross_bearing[_2] (ankle universal joint)
#   HipRoll[LR]       -> (right|left)_hip_roll_03
#   HipPitch[LR]      -> (right|left)_hip_pitch_04
#   KneePitch[LR]     -> (right|left)_knee_04
#   AnkleRoll[LR]     -> (right|left)_foot_roll_02
#   AnklePitch[LR]    -> (right|left)_foot_pitch_02
#
# kscale additionally has a live hip yaw joint, (right|left)_hip_yaw_03, which SD_BRS1's
# URDF carries but disables via a zero-width limit (see project memory: SD_BRS1 NaN
# crash). It is left out of the action/observation space here exactly like SD_BRS1 --
# JointPositionActionCfg below still uses joint_names=[".*"] so it WILL be actuated,
# unlike SD_BRS1. This is intentional (kscale's hip yaw has a real, non-degenerate
# range), but means kscale trains one more DOF than SD_BRS1 and needs its own
# joint_deviation_l1 treatment if the extra axis wanders off in training (see
# pen_hip_deviation below).

_LEG_LINKS_NO_FOOT = "kd_d_102[rl]_6061|kd_d_201r_6061|rs03|kd_d_301[rl]_6061|kd_d_401[rl]_6061|arb_uj111_cross_bearing.*"
_FOOT_LINKS = "foot_6061.*"
_TORSO_LINK = "assy_formfg___kd_b_102b_torso_btm"

##################
# Scene Definition
##################

env_spacing = 2.5


@configclass
class KscaleSceneCfg(InteractiveSceneCfg):
    """Configuration for the kscale scene"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        env_spacing=env_spacing,
    )

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + _TORSO_LINK,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )


##############
# MDP settings
##############


@configclass
class CommandsCfg:
    """Command terms for the MDP"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        debug_vis=True,
        resampling_time_range=(7.5, 12.5),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.8),
            lin_vel_y=(-0.01, 0.01),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )

    # Periodic walking clock -- ported unchanged from SD_BRS1's brs_base_env_cfg.py,
    # see that file's CommandsCfg.gait_command docstring for the full derivation
    # (Siekmann/Walk-These-Ways/Humanoid-Gym clock family). frequency 1.0 Hz, stance
    # duration 0.6, anti-phase offset 0.5.
    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.0, 1.0),
            offsets=(0.5, 0.5),
            durations=(0.6, 0.6),
            swing_height=(0.08, 0.08),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.4,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (standard PPO)"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group"""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        gait_phase = ObsTerm(func=mdp.get_gait_phase)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for critic"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100.0, 100.0), scale=1.0)
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque, clip=(-500.0, 500.0))
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc, clip=(-2000.0, 2000.0))
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS)},
            clip=(-20.0, 20.0),
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos, clip=(-100.0, 100.0))
        robot_vel = ObsTerm(func=mdp.robot_vel, clip=(-20.0, 20.0))
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties, clip=(-10.0, 10.0))
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS)},
            clip=(-5000.0, 5000.0),
        )

        gait_phase = ObsTerm(func=mdp.get_gait_phase)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()


@configclass
class HIMObservationsCfg:
    """Observation specifications for the MDP (HIM architecture)"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group"""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HistoryObsCfg(ObsGroup):
        """Observations for history group (fed to HIM encoder)"""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class TargetEncCfg(ObsGroup):
        """Privileged observations for the target encoder"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100.0, 100.0), scale=1.0)
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for critic"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100.0, 100.0), scale=1.0)
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque, clip=(-500.0, 500.0))
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc, clip=(-2000.0, 2000.0))
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS)},
            clip=(-20.0, 20.0),
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos, clip=(-100.0, 100.0))
        robot_vel = ObsTerm(func=mdp.robot_vel, clip=(-20.0, 20.0))
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties, clip=(-10.0, 10.0))
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS)},
            clip=(-5000.0, 5000.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    @configclass
    class EstimatorGTCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            noise=GaussianNoise(mean=0.0, std=0.00),
            scale=1.0,
        )

    policy: PolicyCfg = PolicyCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    estimatorGT: EstimatorGTCfg = EstimatorGTCfg()
    targetEnc: TargetEncCfg = TargetEncCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    prepare_quantity_for_kscale = EventTerm(
        func=mdp.prepare_quantity_for_tron,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_TORSO_LINK),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_LEG_LINKS_NO_FOOT),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    hip_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_hip_roll_03", "(right|left)_hip_pitch_04"]
            ),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    knee_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(right|left)_knee_04"]),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    ankle_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_foot_roll_02", "(right|left)_foot_pitch_02"]
            ),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_offsets = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_friction_model,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    scale_all_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    # Joint-specific position randomisation. UNVERIFIED ranges -- copied structurally
    # from SD_BRS1's reset events (fractions of that robot's soft joint limits around its
    # nominal pose) but not re-derived from kscale's own URDF limits or its (placeholder)
    # nominal pose in kscale_identified_cfg.py. Revisit once the standing pose is fixed.
    reset_hip_roll_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "(right|left)_hip_roll_03",
            "position_range": (-0.1295, 0.1295),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_hip_pitch_r_joint = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "right_hip_pitch_04",
            "position_range": (-0.24, 0.24),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_hip_pitch_l_joint = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "left_hip_pitch_04",
            "position_range": (-0.24, 0.24),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_knee_pitch_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "(right|left)_knee_04",
            "position_range": (-0.24, 0.24),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_ankle_roll_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "(right|left)_foot_roll_02",
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_ankle_pitch_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "(right|left)_foot_pitch_02",
            "position_range": (-0.15, 0.15),
            "velocity_range": (-0.5, 0.5),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP"""

    keep_balance = RewTerm(func=mdp.stay_alive, weight=0.05)

    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=50,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=15,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )
    rew_no_fly = RewTerm(
        func=mdp.no_fly,
        weight=15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "threshold": 1.0,
            "history_index": 0,
        },
    )
    rew_keep_ankle_pitch_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_foot_pitch_02"]
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
        },
    )
    # Off-axis deviation penalties, mirroring the IsaacLab G1 recipe (joint_deviation_l1
    # on hip roll/yaw, not on hip pitch or the knee, which must swing freely). Unlike
    # SD_BRS1, kscale's hip yaw is a real actuated DOF (see module docstring), so it is
    # included here alongside hip roll -- SD_BRS1 only penalises hip roll because its hip
    # yaw is inert.
    pen_hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_hip_roll_03", "(right|left)_hip_yaw_03"]
            )
        },
    )
    # Penalise ankle joints deviating from the nominal crouch pose, same purpose as
    # pen_hip_deviation. Ported from SD_BRS1's pen_ankle_deviation.
    pen_ankle_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_foot_pitch_02", "(right|left)_foot_roll_02"]
            )
        },
    )

    # UNVERIFIED target_height. SD_BRS1's 1.15 m target was solved from a closed-chain IK
    # against that robot's own thigh/shank lengths (see sd_brs1_identified_cfg.py). kscale's
    # URDF zero-pose does not resolve to a clean straight-leg reference (see
    # kscale_identified_cfg.py docstring), so this 1.0 m is a rough guess pending visual
    # verification via `djinn start play kscale`, not a derived value. Keep this in sync
    # with kscale_identified_cfg.py's init_state.pos z and pen_feet_regulation's
    # base_height_target below once corrected.
    pen_base_height = RewTerm(
        func=mdp.base_height_rough_l2,
        params={
            "target_height": 1.0,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
        weight=-30.0,
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-5)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00001)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[_LEG_LINKS_NO_FOOT, _TORSO_LINK],
            ),
            "threshold": 10.0,
        },
    )
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.075)
    pen_joint_torque_rate = RewTerm(
        func=mdp.JointTorqueRatePenalty,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-50.0)
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-100,
        # UNVERIFIED. SD_BRS1's 0.21 m minimum came from its own hip geometry; kscale's
        # legs are noticeably narrower at the hip (see kscale_identified_cfg.py docstring
        # on the unresolved zero-pose), so this needs re-checking once the robot has been
        # inspected in sim.
        params={"min_feet_distance": 0.21, "feet_links_name": [_FOOT_LINKS]},
    )
    pen_feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[_FOOT_LINKS]),
            # kept in step with pen_base_height's target_height above -- same caveat applies
            "base_height_target": 1.0,
            # UNVERIFIED. Computed by transforming the foot_6061.stl collision mesh's
            # bounding box (in its local mesh frame) through the collision <origin> in the
            # kscale URDF (xyz="-0.0425 0.043 0.02" rpy="1.5708 1.5708 0") into the
            # foot_6061 link frame, which sits at the foot_roll joint. That places the
            # mesh's lowest point at z=-0.19 in the link frame -- much deeper than
            # SD_BRS1's flat sole plate (-0.124), suggesting this mesh may be a tall foot
            # bracket rather than a flat sole, or the transform above is misreading the
            # part. Treat this number as unverified until checked against the robot
            # visually.
            "foot_radius": 0.19,
            "height_decay_scale": 0.03,
        },
    )
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-05)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=12.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
        },
    )
    # rew_foot_clearance is DROPPED relative to SD_BRS1's brs_base_env_cfg.py. That term
    # needs a per-vertex "sole_offsets" table of the true lowest points of the sole mesh
    # under pitch/roll, measured against SD_BRS1's collision mesh; no equivalent
    # measurement has been done for kscale's foot_6061.stl. feet_air_time + feet_slide +
    # pen_feet_regulation are kept and should be enough to get a first walking gait;
    # re-add a foot_clearance_reward_v2 term with a real sole_offsets table once the foot
    # mesh geometry has been inspected properly (e.g. from inside Isaac Sim, not by
    # parsing the STL by hand).

    # Periodic phase reward -- ported unchanged from SD_BRS1's brs_base_env_cfg.py
    # RewardsCfg.rew_gait. See that file's docstring for the full weight derivation
    # (Siekmann/Walk-These-Ways clock construction); the derivation is generic to any
    # biped with a comparable stance/swing split and was not redone for kscale
    # specifically.
    rew_gait = RewTerm(
        func=mdp.GaitReward,
        weight=40.0,
        params={
            "tracking_contacts_shaped_force": -1.0,
            "tracking_contacts_shaped_vel": -1.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_TORSO_LINK),
            "threshold": 1.0,
        },
    )
    low_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        # UNVERIFIED. SD_BRS1 uses 0.4 against a 1.15 m stance (roughly 35%). Scaled here
        # to the same fraction of kscale's placeholder 1.0 m target_height above; revisit
        # together with that target once the real standing height is known.
        params={"minimum_height": 0.35},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel_delayed,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "starting_step": 200 * 24,
        },
    )

    modify_push_force = CurrTerm(
        func=mdp.modify_push_force_v2,
        params={
            "term_name": "push_robot",
            "max_velocity": (3.0, 3.0),
            "interval": 300 * 24,
            "starting_step": 800 * 24,
            "increment_rate": 1.1,
            "decrement_rate": 0.9,
        },
    )

    modify_command_velocity_lin_x = CurrTerm(
        func=mdp.modify_command_velocity_x,
        params={
            "term_name": "rew_lin_vel_xy",
            "max_velocity": (-0.5, 1.0),
            "interval": 300 * 24,
            "starting_step": 1000 * 24,
            "update_rate": 0.005,
            "update_threshold": 0.75,
        },
    )

    modify_command_velocity_lin_y = CurrTerm(
        func=mdp.modify_command_velocity_y,
        params={
            "term_name": "rew_lin_vel_xy",
            "max_velocity": (-0.2, 0.2),
            "interval": 300 * 24,
            "starting_step": 1000 * 24,
            "update_rate": 0.005,
            "update_threshold": 0.75,
        },
    )

    modify_command_velocity_ang_z = CurrTerm(
        func=mdp.modify_command_velocity_angular,
        params={
            "term_name": "rew_ang_vel_z",
            "max_velocity": (-0.9, 0.9),
            "interval": 300 * 24,
            "starting_step": 1000 * 24,
            "update_rate": 0.005,
            "update_threshold": 0.7,
        },
    )

    modify_linear_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_lin_vel_xy",
            "interval": 300 * 24,
            "starting_step": 900 * 24,
            "update_rate": 0.95,
            "update_threshold": 0.67,
            "minimum_std": 0.09,
        },
    )

    modify_angular_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_ang_vel_z",
            "interval": 300 * 24,
            "starting_step": 0,
            "update_rate": 0.975,
            "update_threshold": 0.5,
            "minimum_std": 0.09,
        },
    )


########################
# Environment definition
########################


@configclass
class KscaleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the kscale standard PPO environment"""

    scene: KscaleSceneCfg = KscaleSceneCfg(num_envs=4096, env_spacing=env_spacing)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        self.sim.dt = 0.005
        self.seed = 42
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class KscaleHIMEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the kscale HIM environment"""

    scene: KscaleSceneCfg = KscaleSceneCfg(num_envs=4096, env_spacing=env_spacing)
    observations: HIMObservationsCfg = HIMObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        self.sim.dt = 0.005
        self.seed = 42
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
