import math
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.data.Robots.model_humanoid_lens110.lens110 import LENS110_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg


ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 3
LENS110_BASE_HEIGHT = 0.66
LENS110_TOTAL_MASS_KG = 22.92451
LENS110_BASE_MASS_RANDOMIZATION_KG = round(LENS110_TOTAL_MASS_KG * 0.033, 2)
LENS110_MOTION_DATA_DIR = os.path.join(LEGGED_LAB_ROOT_DIR, "data", "MotionData", "lens110_amp_lab")

# 定义下肢关键身体名称（移除上肢）
LENS110_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    # "left_elbow_link",      # 移除
    # "right_elbow_link",     # 移除
    # "left_shoulder_roll_link",  # 移除
    # "right_shoulder_roll_link", # 移除
]

# 定义下肢关节名称列表（12个）
LENS110_LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

# 躯干/上肢关节名称列表（不在动作空间中，由默认 PD 保持默认姿态；仅作记录）
LENS110_UPPER_JOINT_NAMES = [
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# 下肢关节在运动数据 dof 数组中的索引（0-11，顺序与 URDF/机器人 dof 顺序一致）
LENS110_LEG_JOINT_IDS = list(range(len(LENS110_LEG_JOINT_NAMES)))

# 踝部关键点在运动数据 key_body 数组中的索引
# （pkl 中 key_body_names 顺序: [left_ankle_roll_link, right_ankle_roll_link, left_elbow_link, ...]）
LENS110_ANKLE_BODY_IDS = [0, 1]

LENS110_MOTION_NAMES = [
    "lens110_01_in_place_step_ground_fixed",
    "lens110_02_steady_walk_ground_fixed",
    "lens110_03_stand_to_walk_start_ground_fixed",
    "lens110_04_walk_to_stand_stop_ground_fixed",
    "lens110_05_stand_walk_stop_ground_fixed",
]


@configclass
class Lens110AmpRewards:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.0)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    base_height = RewTerm(func=mdp.base_height, weight=0.0, params={"target_height": LENS110_BASE_HEIGHT})

    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0.0)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=0.0)
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=0.0)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    # 注释掉上肢相关的奖励
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_.*_joint",
    #                 ".*_elbow_joint",
    #             ],
    #         )
    #     },
    # )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_yaw_joint")},
    )
    # 注释掉上肢对称奖励
    # arm_symmetry = RewTerm(func=mdp.lens110_arm_symmetry_l2, weight=0.0)
    # arm_vel_symmetry = RewTerm(func=mdp.lens110_arm_vel_symmetry_l2, weight=0.0)
    hip_yaw_symmetry = RewTerm(func=mdp.lens110_hip_yaw_symmetry_l2, weight=0.0)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.0,
        },
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_y,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.18, "max": 0.46},
    )
    knee_distance = RewTerm(
        func=mdp.knee_distance_y,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee_link"]), "min": 0.14, "max": 0.40},
    )
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class Lens110AmpEnvCfg(LocomotionAmpEnvCfg):
    rewards: Lens110AmpRewards = Lens110AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2048
        self.scene.robot = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ========== 修改1: 动作空间只保留下肢 12 个关节 ==========
        # 策略输出 12 维动作；躯干/上肢不在动作空间中（由默认 PD 保持默认姿态）
        self.actions.joint_pos.joint_names = LENS110_LEG_JOINT_NAMES
        self.actions.joint_pos.scale = {
            joint_name: 0.25 for joint_name in LENS110_LEG_JOINT_NAMES
        }

        # ========== 修改2: 修改观察空间 - 只观察下肢关节 ==========
        # 策略网络观察
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", 
            joint_names=LENS110_LEG_JOINT_NAMES
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", 
            joint_names=LENS110_LEG_JOINT_NAMES
        )
        
        # Critic网络观察（特权观察）
        self.observations.critic.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", 
            joint_names=LENS110_LEG_JOINT_NAMES
        )
        self.observations.critic.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", 
            joint_names=LENS110_LEG_JOINT_NAMES
        )
        
        # Discriminator观察
        self.observations.disc.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", 
            joint_names=LENS110_LEG_JOINT_NAMES
        )
        self.observations.disc.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", 
            joint_names=LENS110_LEG_JOINT_NAMES
        )

        # ========== 修改3: 运动数据相关 ==========
        self.motion_data.motion_dataset.motion_data_dir = LENS110_MOTION_DATA_DIR
        self.motion_data.motion_dataset.motion_data_weights = {
            motion_name: (1.0, "auto") for motion_name in LENS110_MOTION_NAMES
        }

        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        
        # 只保留下肢关键点
        self.observations.disc.key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=LENS110_KEY_BODY_NAMES,
                    preserve_order=True,
                ),
            },
        )
        self.observations.disc_demo.ref_root_lin_vel_b = ObsTerm(
            func=mdp.ref_root_lin_vel_b,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
            },
        )
        self.observations.disc_demo.ref_key_body_pos_b = ObsTerm(
            func=mdp.ref_key_body_pos_b,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
                "body_ids": LENS110_ANKLE_BODY_IDS,
            },
        )
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        # 只取下肢关节（与 disc 组的 joint_pos/joint_vel 维度对齐）
        self.observations.disc_demo.ref_joint_pos.params["joint_ids"] = LENS110_LEG_JOINT_IDS
        self.observations.disc_demo.ref_joint_vel.params["joint_ids"] = LENS110_LEG_JOINT_IDS
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME

        # ========== 修改4: 事件随机化 - 只对下肢 ==========
        self.events.add_base_mass.params["asset_cfg"].body_names = "pelvis"
        self.events.add_base_mass.params["mass_distribution_params"] = (
            -LENS110_BASE_MASS_RANDOMIZATION_KG,
            LENS110_BASE_MASS_RANDOMIZATION_KG,
        )
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["pelvis", "torso_yaw_link"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_yaw_link"]
        # 只随机化下肢质量
        self.events.scale_link_mass.params["asset_cfg"].body_names = [
            "left_hip_.*_link", 
            "right_hip_.*_link",
            "left_knee_link", 
            "right_knee_link",
            "left_ankle_.*_link", 
            "right_ankle_.*_link"
        ]
        self.events.scale_link_mass.params["mass_distribution_params"] = (0.9, 1.1)

        # ========== 修改4b: 固定躯干/上肢关节驱动器目标 ==========
        # 这些关节不在动作空间中，若不定目标，PhysX 驱动器目标恒为 0，
        # 手臂会从默认姿态（肩 0.4/0.2、肘 -0.8）被猛拉到零位，把机器人掀翻。
        # 每次 reset 把目标钉回默认姿态（不会被动作管理器覆盖，整个 episode 有效）。
        self.events.pin_upper_joints_target = EventTerm(
            func=mdp.pin_joints_to_default_target,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LENS110_UPPER_JOINT_NAMES)},
        )

        # ========== 修改5: 奖励权重调整 ==========
        self.rewards.track_lin_vel_xy_exp.weight = 1.25
        self.rewards.track_ang_vel_z_exp.weight = 1.25
        self.rewards.alive.weight = 0.1
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height.params["target_height"] = LENS110_BASE_HEIGHT
        self.rewards.base_height.weight = -2.0
        self.rewards.joint_vel_l2.weight = -2e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_regularization.weight = -2e-3
        self.rewards.joint_deviation_hip.weight = -0.03
        # 上肢奖励设为0（或注释掉）
        # self.rewards.joint_deviation_arms.weight = -0.025  # 注释或设为0
        self.rewards.joint_deviation_torso.weight = -0.01
        # self.rewards.arm_symmetry.weight = -0.04        # 注释或设为0
        # self.rewards.arm_vel_symmetry.weight = -0.001    # 注释或设为0
        self.rewards.hip_yaw_symmetry.weight = -0.15
        self.rewards.feet_air_time.weight = 0.6
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.feet_slide.weight = -0.12
        self.rewards.sound_suppression.weight = -1e-4
        self.rewards.feet_distance.weight = 0.15
        self.rewards.knee_distance.weight = 0.10
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*ankle.*).*"],
        )

        # ========== 修改6: 命令范围 ==========
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)
        self.commands.base_velocity.ranges.zero_prob = (0.1, 0.1, 0.1)
        self.commands.base_velocity.ranges.heading = None

        self.curriculum.lin_vel_cmd_levels.params["lin_vel_x_limit"] = [-0.6, 2.0]
        self.curriculum.lin_vel_cmd_levels.params["lin_vel_y_limit"] = [-0.6, 0.6]

        # ========== 修改7: 终止条件 - 移除上肢接触检测 ==========
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis",
            "torso_yaw_link",
            ".*_hip_.*_link",
            ".*_knee_link",
            # 移除上肢检测
            # ".*_shoulder_.*_link",
            # ".*_elbow_link",
        ]
        self.terminations.base_height.params["minimum_height"] = 0.34

        if self.__class__.__name__ == "Lens110AmpEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Lens110AmpEnvCfg_PLAY(Lens110AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.zero_prob = (0.0, 0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.debug_vis = False

        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.scale_link_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.curriculum.lin_vel_cmd_levels = None