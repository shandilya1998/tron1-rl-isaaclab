import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from environments.actuators import IdentifiedActuatorCfg

current_dir = os.path.dirname(__file__)
urdf_path = os.path.join(current_dir, "../urdf/solefoot/kscale/kscale.urdf")


KSCALE_HIP_YAW_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_hip_yaw_03"],
    effort_limit=60.0,
    velocity_limit=20.0,
    saturation_effort=60.0,
    # stiffness={".*": 15.0},
    # damping={".*": 0.9},
    stiffness={".*": 500.0},
    damping={".*": 18},
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
    # stiffness={".*": 150.0},
    # damping={".*": 17.3},
    stiffness={".*": 250.0},
    damping={".*": 10},
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
    damping={".*": 8},
    # stiffness={".*": 200.0},
    # damping={".*": 22.4},
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
    stiffness={".*": 300.0},
    damping={".*": 10.0},
    # stiffness={".*": 200.0},
    # damping={".*": 10.0},
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
    # stiffness={".*": 20.0},
    # damping={".*": 0.5},
    stiffness={".*": 170.0},
    damping={".*": 9},
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
    # stiffness={".*": 50.0},
    # damping={".*": 1.7},
    stiffness={".*": 170.0},
    damping={".*": 9},
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

init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.79),
    joint_pos={
        "right_hip_pitch_04": -0.1,
        "left_hip_pitch_04": 0.1,
        "right_hip_roll_03": 0.0,
        "left_hip_roll_03": 0.0,
        "right_hip_yaw_03": 0.0,
        "left_hip_yaw_03": 0.0,
        "right_knee_04": 0.4,
        "left_knee_04": 0.4,
        "right_foot_pitch_02": -0.3,
        "left_foot_pitch_02": -0.3,
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
