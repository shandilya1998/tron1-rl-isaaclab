import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from bipedal_locomotion.actuators import IdentifiedActuatorCfg

current_dir = os.path.dirname(__file__)
urdf_path = os.path.join(current_dir, "../urdf/solefoot/kscale/kscale.urdf")

# PLACEHOLDER GAINS. kscale's URDF carries no actuator spec sheet (config.json is just
# Onshape export metadata) -- these are copied from SD_BRS1's per-joint-type identified
# gains (sd_brs1_identified_cfg.py) as a starting point only, matched by role (hip
# pitch/roll/knee/ankle-pitch/ankle-roll). They have NOT been fit to kscale hardware and
# must be retuned. The rs04/rs03/rs02 mesh names at the hip/knee/ankle joints (see
# kscale/assets/*.stl) look like they could reference a real actuator line (e.g.
# Robstride RS02/03/04), which would be a better source for real gains if confirmed.
#
# kscale also has an actively-limited hip yaw joint (+-1.5708 rad), unlike SD_BRS1 where
# HipYaw is present in the URDF but disabled via a zero-width [0,0] limit (see
# project memory: SD_BRS1 NaN crash root-cause). kscale's hip yaw gains below are a
# straight guess, no SD_BRS1 analogue to copy since that axis is inert there.

KSCALE_HIP_YAW_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_hip_yaw_03"],
    effort_limit=60.0,
    velocity_limit=20.0,
    saturation_effort=60.0,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
    armature={".*": 0.01},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_HIP_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_hip_roll_03"],
    effort_limit=60.0,
    velocity_limit=20.0,
    saturation_effort=60.0,
    stiffness={".*": 150.0},
    damping={".*": 45.0},
    armature={".*": 0.01},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_HIP_PITCH_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_hip_pitch_04"],
    effort_limit=60.0,
    velocity_limit=20.0,
    saturation_effort=60.0,
    stiffness={".*": 200.0},
    damping={".*": 50.0},
    armature={".*": 0.01},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_KNEE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_knee_04"],
    effort_limit=60.0,
    velocity_limit=20.0,
    saturation_effort=60.0,
    stiffness={".*": 200.0},
    damping={".*": 22.0},
    armature={".*": 0.015},
    friction_static=0.8,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_FOOT_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_foot_roll_02"],
    effort_limit=17.0,
    velocity_limit=10.0,
    saturation_effort=17.0,
    stiffness={".*": 20.0},
    damping={".*": 4.0},
    armature={".*": 0.005},
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_FOOT_PITCH_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_foot_pitch_02"],
    effort_limit=17.0,
    velocity_limit=10.0,
    saturation_effort=17.0,
    stiffness={".*": 50.0},
    damping={".*": 4.0},
    armature={".*": 0.005},
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

rigid_props = sim_utils.RigidBodyPropertiesCfg(
    rigid_body_enabled=True,
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=2,
    solver_velocity_iteration_count=2,
)

# PLACEHOLDER STANDING POSE. Unlike SD_BRS1, kscale's URDF zero-pose is not a clean
# straight-leg reference frame -- the joint origins are raw Onshape/CAD frames with
# unexplained offsets (FK of the zero pose lands the foot ~0.13m off to the side of the
# hip, not straight below it), so the SD_BRS1 closed-chain IK trick used to derive its
# 1.15m crouch does not carry over without first inspecting the robot in sim. This pose
# is a mild, unverified knee-bend guess -- verify visually with
# `./djinn start play kscale <run> <seed>` and correct joint_pos / pos (spawn height)
# from what's actually observed before trusting this for real training.
init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.0),
    joint_pos={
        "right_hip_pitch_04": 0.0,
        "left_hip_pitch_04": 0.0,
        "right_hip_roll_03": 0.0,
        "left_hip_roll_03": 0.0,
        "right_hip_yaw_03": 0.0,
        "left_hip_yaw_03": 0.0,
        "right_knee_04": 0.3,
        "left_knee_04": 0.3,
        "right_foot_pitch_02": -0.15,
        "left_foot_pitch_02": -0.15,
        "right_foot_roll_02": 0.0,
        "left_foot_roll_02": 0.0,
    },
    joint_vel={".*": 0.0},
)

actuators = {
    "hip_yaw": KSCALE_HIP_YAW_ACTUATOR_CFG,
    "hip_roll": KSCALE_HIP_ROLL_ACTUATOR_CFG,
    "hip_pitch": KSCALE_HIP_PITCH_ACTUATOR_CFG,
    "knee": KSCALE_KNEE_ACTUATOR_CFG,
    "foot_roll": KSCALE_FOOT_ROLL_ACTUATOR_CFG,
    "foot_pitch": KSCALE_FOOT_PITCH_ACTUATOR_CFG,
}

KSCALE_IDENTIFIED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=urdf_path,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=None,
        self_collision=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=True,
    ),
    init_state=init_state,
    soft_joint_pos_limit_factor=0.9,
    actuators=actuators,
)
