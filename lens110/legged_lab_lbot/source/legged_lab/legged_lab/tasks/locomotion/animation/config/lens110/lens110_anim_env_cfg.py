import os

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.data.Robots.model_humanoid_lens110.lens110 import LENS110_CFG
from legged_lab.tasks.locomotion.animation.animation_env_cfg import AnimationEnvCfg


@configclass
class Lens110AnimEnvCfg(AnimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot_anim = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_anim")
        self.scene.robot_anim.spawn.rigid_props.disable_gravity = True  # type: ignore
        self.scene.robot_anim.spawn.articulation_props.enabled_self_collisions = False  # type: ignore
        self.scene.robot_anim.spawn.activate_contact_sensors = False  # type: ignore
        self.scene.robot_anim.spawn.collision_props = sim_utils.CollisionPropertiesCfg(  # type: ignore
            collision_enabled=False
        )

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "lens110_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "A1-_Stand_stageii": 1.0,
            "B10_-__Walk_turn_left_45_stageii": 1.0,
            "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            "B15_-__Walk_turn_around_stageii": 1.0,
            "B22_-__side_step_left_stageii": 1.0,
            "B23_-__side_step_right_stageii": 1.0,
        }

        self.animation.animation.random_initialize = True
        self.animation.animation.num_steps_to_use = 1
