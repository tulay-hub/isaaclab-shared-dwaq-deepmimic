import torch
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.deepmimic.mdp as mdp
from legged_lab.tasks.locomotion.deepmimic.config.lens110.lens110_deepmimic_env_cfg import (
    ANIMATION_TERM_NAME,
    Lens110DeepMimicEnvCfg,
)


POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]

DATASET_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

DATASET_TO_POLICY = [DATASET_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES]


def root_ang_vel_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """根角速度投影到机身坐标系，替代 161 维训练契约中的世界系角速度。"""
    robot = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b


def ref_joint_pos_policy_order(
    env,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    """把动画数据集中的 USD 交叉顺序参考关节角重排为 URDF 策略顺序。"""
    animation_term = env.animation_manager.get_term(animation)
    ref_joint_pos = animation_term.get_dof_pos()
    ref_joint_pos = ref_joint_pos[:, :, DATASET_TO_POLICY]
    if flatten_steps_dim:
        return ref_joint_pos.reshape(env.num_envs, -1)
    return ref_joint_pos


@configclass
class Lens110DeepMimicEnvCfg159(Lens110DeepMimicEnvCfg):
    """159 维舞蹈训练契约。

    观测顺序: root_rot_tan_norm(6) + root_ang_vel_b(3) + joint_pos(21)
    + joint_vel(21) + ref_root_rot_tan_norm(24) + ref_joint_pos(84) = 159。
    动作与策略输入输出统一为 URDF 分组顺序; 动画数据集仍保持原 USD 交叉顺序。
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos = mdp.ReferenceJointPositionActionCfg(
            asset_name="robot",
            joint_names=POLICY_JOINT_NAMES,
            scale=0.25,
            use_default_offset=False,
            preserve_order=True,
            animation=ANIMATION_TERM_NAME,
        )

        joint_cfg = SceneEntityCfg(
            "robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True
        )
        self.observations.policy.root_ang_vel_w = ObsTerm(func=root_ang_vel_b)
        self.observations.policy.joint_pos.params = {"asset_cfg": joint_cfg}
        self.observations.policy.joint_vel.params = {"asset_cfg": joint_cfg}
        self.observations.policy.ref_joint_pos.params = {
            "animation": ANIMATION_TERM_NAME,
        }
        self.observations.policy.ref_joint_pos.func = ref_joint_pos_policy_order

        # 159 维契约删除接触标志；对应输入分布与 161 维旧 checkpoint 不同，因此重新训练。
        self.observations.policy.foot_contact = None

        obs_noise = {
            "root_rot_tan_norm": Unoise(n_min=-0.01, n_max=0.01),
            "root_ang_vel_w": Unoise(n_min=-0.1, n_max=0.1),
            "joint_pos": Unoise(n_min=-0.01, n_max=0.01),
            "joint_vel": Unoise(n_min=-0.2, n_max=0.2),
        }
        for name, noise_cfg in obs_noise.items():
            term = getattr(self.observations.policy, name, None)
            if term is not None:
                term.noise = noise_cfg


@configclass
class Lens110DeepMimicEnvCfg159_PLAY(Lens110DeepMimicEnvCfg159):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.animation.animation.random_initialize = False
