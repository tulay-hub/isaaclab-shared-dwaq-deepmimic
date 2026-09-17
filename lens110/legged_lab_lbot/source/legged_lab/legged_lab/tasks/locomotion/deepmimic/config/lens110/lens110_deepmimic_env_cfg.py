import os
import math
from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
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
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.deepmimic.mdp as mdp
from legged_lab.tasks.locomotion.deepmimic.deepmimic_env_cfg import DeepMimicEnvCfg, DeepMimicSceneCfg
from legged_lab import LEGGED_LAB_ROOT_DIR


def foot_contact_flags(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=[".*_ankle_roll_link"], preserve_order=True
    ),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact flags (0/1) for the feet, in left/right ankle roll link order.

    The final policy kept high action noise on the ankles (std ~0.28-0.46), i.e. it
    could not pin down foot control. Feeding the policy a direct "which foot is on the
    ground" signal gives it the foot-contact phase information it was missing.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 2, 3)
    flags = torch.norm(net_forces, dim=-1) > threshold  # (N, 2)
    return flags.float()


def action_rate_l2_scaled(
    env,
    scale: float = 0.25,
) -> torch.Tensor:
    """Action-rate L2 penalty computed in joint-target space (scale-invariant).

    The standard ``mdp.action_rate_l2`` penalizes the change of RAW policy actions.
    With ``actions.joint_pos.scale = 0.25``, the same joint motion requires 4x larger
    raw actions, so the raw penalty is amplified 16x and the policy is forced to slow
    down (falls over). Rescaling the action delta by ``scale`` before squaring makes
    the penalty proportional to the actual joint-target rate, independent of scale.
    """
    action_delta = (env.action_manager.action - env.action_manager.prev_action) * scale
    return torch.sum(torch.square(action_delta), dim=1)


def alive_reward(env) -> torch.Tensor:
    """轻量存活奖励: 每活一步 +1 (权重 0.2, 鼓励先站稳)。"""
    return torch.ones(env.num_envs, device=env.device)

# Robot configuration for dance (Lens110)
from legged_lab.data.Robots.model_humanoid_lens110.lens110_dance import LENS110_CFG

# The order must align with the key_body_names inside the motion data pkl:
#   data/MotionData/lens110_deepminic_lab/lens110_tangbohushuo_50hz_reorder.pkl
#   -> [left_ankle_roll_link, right_ankle_roll_link, left_elbow_link, right_elbow_link,
#       left_shoulder_roll_link, right_shoulder_roll_link]
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]
ANIMATION_TERM_NAME = "animation"

@configclass
class Lens110DeepMimicEnvCfg(DeepMimicEnvCfg):
    """Lens110 舞蹈任务（H 版：无关键点观测，161 维，真机部署用）。

    消融实验结论：删掉关键点观测（key_body_pos_b + ref_key_body_pos_b）后
    总奖励仅降 ~5%（24.2 → 23.0），但观测更简单、不依赖 FK 标定精度，
    适合真机部署。F 版（含关键点观测，251 维，训练效果最优）备份在
    ``lens110_deepmimic_env_cfg_keybody.py`` 中。
    """

    def __post_init__(self):
        super().__post_init__()

        # 500Hz 物理 / 100Hz 策略 (decimation=5)
        self.sim.dt = 1.0 / 500.0
        self.decimation = 5
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # 每集固定从第 0 帧开始（不随机起始帧），与真实部署时舞蹈从头播放一致
        self.animation.animation.random_initialize = False

        # 轻量存活奖励
        self.rewards.alive = RewTerm(func=alive_reward, weight=0.2)

        # 当前舞蹈 3452 帧 @ 100 Hz = 34.51 s; give a small margin.
        self.episode_length_s = 40.0

        self.scene.robot = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "lens110_deepminic_lab"
        )
        # 使用用户自己的舞蹈动作: tangbohushuoDJ_v2_100hz_flatfix
        # (100Hz, 34.51s, 脚底水平修正版, 关节已重排为 USD 顺序)
        self.motion_data.motion_dataset.motion_data_weights = {
            "tangbohushuoDJ_v2_100hz_flatfix": 1.0,
        }

        # -----------------------------------------------------
        # Actions
        # -----------------------------------------------------
        # 动作 clip 由 rsl_rl runner 的 clip_actions 执行；当前恢复 11-59-20 配置为 null。
        # 参考中心残差动作: q_des = 参考当前帧关节角 + 0.25 * clip(action)
        # scale=0.25 只表示残差修正范围, 完整舞蹈动作由参考直接提供
        # (与部署端 q_des = ref + scale*action 一致, 不会因为 ±0.25 丢失手臂动作)。
        self.actions.joint_pos = mdp.ReferenceJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.25,
            use_default_offset=False,
            animation=ANIMATION_TERM_NAME,
        )

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            )
        }
        self.observations.policy.ref_root_pos_error.params = {
            "animation": ANIMATION_TERM_NAME
        }
        self.observations.policy.ref_root_rot_tan_norm.params = {
            "animation": ANIMATION_TERM_NAME
        }
        self.observations.policy.ref_joint_pos.params = {
            "animation": ANIMATION_TERM_NAME
        }
        self.observations.policy.ref_key_body_pos_b.params = {
            "animation": ANIMATION_TERM_NAME
        }

        # H 版：删除关键点观测（key_body_pos_b + ref_key_body_pos_b），-90 维 → 161 维
        # （F 版备份见 lens110_deepmimic_env_cfg_keybody.py）
        self.observations.policy.key_body_pos_b = None
        self.observations.policy.ref_key_body_pos_b = None

        # Foot contact flags (2 dims: left/right foot on ground) to help ankle control.
        # NOTE: sensor_cfg 必须显式放进 params，否则 ObservationManager 不会 resolve 函数默认的
        # sensor_cfg（body_ids 保持 slice(None)），会把全部 22 个 body 的接触标志都输出。
        self.observations.policy.foot_contact = ObsTerm(
            func=foot_contact_flags,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[".*_ankle_roll_link"],
                    preserve_order=True,
                ),
                "threshold": 1.0,
            },
        )

        # 观测噪声：只给真实传感器可测的量加噪声，reference / contact 项不加
        self.observations.policy.enable_corruption = True
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

        # 真机部署（仅 IMU + 关节编码器）：移除依赖全局定位/状态估计、真机无法可靠获取的观测项
        self.observations.policy.root_vel_w = None          # 世界系线速度：IMU 无法直接测量，积分漂移
        self.observations.policy.root_height = None         # 机身绝对高度：IMU 测不出，无视觉/测距辅助
        self.observations.policy.ref_root_pos_error = None  # 参考位置误差 x,y：需当前根位置，漂移累积

        # -----------------------------------------------------
        # Events
        # -----------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = ".*"
        # 质量随机 ±0.1kg: 脚本身只有零点几 kg, ±1kg 会把脚质量拉得接近 0
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.1, 0.1)
        self.events.base_com.params["asset_cfg"].body_names = ".*"
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            # 0: 每集开始直接落在参考第 0 帧姿态（脚底贴地），不做额外抬升
            "height_offset": 0.0,
        }
        # 域随机：PD 增益 ±10%（scale 0.9~1.1）
        self.events.scale_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )
        # 域随机：摩擦系数 0.8~1.0（0.5 太滑, 会制造“出生就滑倒”的环境）
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.0)

        # -----------------------------------------------------
        # Rewards
        # -----------------------------------------------------
        # scale=0.25 下把 action_rate_l2 换成关节目标空间的 scale 不变版本
        # （原始动作惩罚会被放大 16 倍，导致策略动作变慢、摔倒率飙升）
        self.rewards.action_rate_l2.func = action_rate_l2_scaled
        self.rewards.action_rate_l2.params = {"scale": 0.25}
        # 恢复 2026-09-01_11-59-20: 原版 action_rate 惩罚强度。
        self.rewards.action_rate_l2.weight = -0.001

        self.rewards.ref_track_root_pos_w_error_exp.weight = 0.15
        self.rewards.ref_track_root_pos_w_error_exp.params = {
            "std": 0.5,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_quat_error_exp.weight = 0.15
        self.rewards.ref_track_quat_error_exp.params = {
            "std": 0.5,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_root_vel_w_error_exp.weight = 0.1
        self.rewards.ref_track_root_vel_w_error_exp.params = {
            "std": 1.0,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_root_ang_vel_w_error_exp.weight = 0.05
        self.rewards.ref_track_root_ang_vel_w_error_exp.params = {
            "std": 1.0,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_key_body_pos_b_error_exp.weight = 0.3
        self.rewards.ref_track_key_body_pos_b_error_exp.params = {
            "std": 0.3,
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            ),
        }
        self.rewards.ref_track_dof_pos_error_exp.weight = 0.8
        self.rewards.ref_track_dof_pos_error_exp.params = {
            "std": 2.0,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_dof_vel_error_exp.weight = 0.1
        self.rewards.ref_track_dof_vel_error_exp.params = {
            "std": 10.0,
            "animation": ANIMATION_TERM_NAME,
        }

        # -----------------------------------------------------
        # Terminations
        # -----------------------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis",
            ".*_shoulder_.*_link",
            ".*_elbow_link",
        ]
        self.terminations.base_height.params["minimum_height"] = 0.3
        # Keep a mild orientation limit for dance (matches earlier lens110 runs).
        self.terminations.bad_orientation.params["limit_angle"] = math.radians(40.0)
        # End the episode when the (finite) dance reference is exhausted.
        self.terminations.motion_data_finish = DoneTerm(func=mdp.motion_data_finish)
        # Dance travels around: use a looser root deviation threshold.
        self.terminations.deviation_root_pos_w.params = {
            "threshold": 1.2,
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.deviation_key_body_pos_w.params = {
            # scale=0.25 下“最佳可达”关键点偏差最大 0.32m (肘), 1.5 已留足余量;
            # 主要用途仍是捕获摔倒/整体漂移, 不会误杀正常站立的 episode。
            "threshold": 1.5,
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            ),
        }


# For debug only
@configclass
class Lens110DeepMimicEnvCfg_DEBUG(Lens110DeepMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 8
        self.scene.env_spacing = 3.0

        self.scene.robot_anim = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_anim")
        self.scene.robot_anim.spawn.rigid_props.disable_gravity = True  # type: ignore
        self.scene.robot_anim.spawn.articulation_props.enabled_self_collisions = False  # type: ignore
        self.scene.robot_anim.spawn.activate_contact_sensors = False  # type: ignore
        self.scene.robot_anim.spawn.collision_props = sim_utils.CollisionPropertiesCfg(  # type: ignore
            collision_enabled=False
        )

        self.animation.animation.enable_visualization = True
        self.animation.animation.vis_root_offset = [2.0, 0.0, 0.0]
        self.animation.animation.random_initialize = False


@configclass
class Lens110DeepMimicEnvCfg_PLAY(Lens110DeepMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0

        # All envs start from frame 0 and play the same dance in sync.
        self.animation.animation.random_initialize = False
