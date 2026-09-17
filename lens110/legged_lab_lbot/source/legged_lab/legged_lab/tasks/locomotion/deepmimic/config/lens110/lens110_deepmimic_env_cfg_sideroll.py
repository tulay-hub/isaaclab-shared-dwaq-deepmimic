import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.deepmimic.config.lens110.lens110_deepmimic_env_cfg import (
    ANIMATION_TERM_NAME,
)
from legged_lab.tasks.locomotion.deepmimic.config.lens110.lens110_deepmimic_env_cfg_159 import (
    Lens110DeepMimicEnvCfg159,
)


def standing_still_vel_penalty(
    env,
    ref_vel_threshold: float,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """参考静止时惩罚机器人根线速度, 压制站立段的蹭步。

    参考根速度低于阈值视为站立段, 此时对机器人速度平方求和作为惩罚;
    滚动段参考速度远高于阈值, 惩罚自动归零, 不影响翻滚学习。
    """
    robot = env.scene[asset_cfg.name]
    animation_term = env.animation_manager.get_term(animation)
    ref_vel = animation_term.get_root_vel_w()[:, 0, :]
    robot_vel = robot.data.root_lin_vel_w
    still_mask = (torch.norm(ref_vel, dim=-1) < ref_vel_threshold).to(robot_vel.dtype)
    return still_mask * torch.sum(torch.square(robot_vel), dim=-1)


@configclass
class Lens110DeepMimicSideRollEnvCfg(Lens110DeepMimicEnvCfg159):
    """侧滚动作训练配置（100Hz，动作5 重定向数据）。

    在 159 部署契约（观测/动作不变）基础上为滚地动作做两处调整：
    1) termination 调整：滚地时躯干横躺、触地都是合法状态，移除 base_contact 和
       bad_orientation，root 高度下限降到 0.02；但保留相对参考的偏差终止并收紧阈值
       (0.8m)，躺平后与参考站姿的关键点偏差远超该阈值，会被立刻终止，倒逼策略起立，
       同时不会误杀贴合参考的滚地中段；
    2) RSI 开启：random_initialize=True，每集从参考动作随机帧起步。
    """

    def __post_init__(self):
        super().__post_init__()

        # 侧滚 100Hz 数据（关节已重排为 USD 交叉顺序）
        self.motion_data.motion_dataset.motion_data_weights = {
            "side_roll_R_002__A415_M_100hz_lens110": 1.0,
        }

        # RSI：随机帧初始化
        self.animation.animation.random_initialize = True

        # termination 调整（滚地合法，躺平不可活）
        self.terminations.base_contact = None
        self.terminations.bad_orientation = None
        self.terminations.base_height.params["minimum_height"] = 0.02
        self.terminations.deviation_root_pos_w.params["threshold"] = 0.8
        self.terminations.deviation_key_body_pos_w.params["threshold"] = 0.8

        # 躺平不应享受存活低保
        self.rewards.alive.weight = 0.05

        # 鲁棒性扩展 (v6): 躯干滑动接触差异是 sim2sim 失败根源,
        # 摩擦范围加宽到 0.5~1.5 覆盖不同滑动条件;
        # 质量随机改到骨盆单点 ±0.5kg, 模拟真实负载/电池位置变化。
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.5)
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)

        # 站立段静止惩罚: 参考静止(根速度 < 0.1 m/s)时惩罚机器人根速度。
        # 站立段参考速度实测 ~0.04 m/s, 滚动段 ~1-2.4 m/s, 阈值 0.1 边界干净。
        # 权重 -2.0: 蹭步(0.1 m/s)一段站立累计代价超过其他奖励总和, 真站稳
        # (0.02 m/s)的代价趋近于零, 梯度方向明确指向站死。
        self.rewards.standing_still = RewTerm(
            func=standing_still_vel_penalty,
            weight=-2.0,
            params={
                "ref_vel_threshold": 0.1,
                "animation": ANIMATION_TERM_NAME,
            },
        )

        # root 位置保持原始权重 1.0: 滚动行程 2.6m, 降权会让滚动中段
        # 偏离参考轨迹、样本质量变差(2026-09-09 实测, 已回退)。


@configclass
class Lens110DeepMimicSideRollEnvCfg_PLAY(Lens110DeepMimicSideRollEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.animation.animation.random_initialize = False
