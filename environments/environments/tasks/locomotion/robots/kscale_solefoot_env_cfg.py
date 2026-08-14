from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.kscale_identified_cfg import KSCALE_IDENTIFIED_CFG
from bipedal_locomotion.tasks.locomotion.cfg.kscale.kscale_base_env_cfg import (
    KscaleEnvCfg,
    KscaleHIMEnvCfg,
)
from bipedal_locomotion.tasks.locomotion.cfg.SF.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
)

######################
# kscale Base Environments
######################


@configclass
class KscaleBaseEnvCfg(KscaleEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = KSCALE_IDENTIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-4.5, 0.0, 4.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class KscaleBaseEnvCfg_PLAY(KscaleBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


######################
# kscale HIM Base Environments
######################


@configclass
class KscaleHIMBaseEnvCfg(KscaleHIMEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = KSCALE_IDENTIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.1)


@configclass
class KscaleHIMBaseEnvCfg_PLAY(KscaleHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


######################
# kscale Blind Flat
######################


@configclass
class KscaleBlindFlatEnvCfg(KscaleBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class KscaleBlindFlatEnvCfg_PLAY(KscaleBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


######################
# kscale Blind Rough
######################


@configclass
class KscaleBlindRoughEnvCfg(KscaleBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class KscaleBlindRoughEnvCfg_PLAY(KscaleBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG


######################
# kscale HIM Blind Flat
######################


@configclass
class KscaleHIMBlindFlatEnvCfg(KscaleHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.heights = None
        self.curriculum.terrain_levels = None


@configclass
class KscaleHIMBlindFlatEnvCfg_PLAY(KscaleHIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.heights = None
        self.curriculum.terrain_levels = None
