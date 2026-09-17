"""Lens110 H 版 (161 维观测) 跟踪任务配置。

使用 whole_body_tracking_engineai (BeyondMimic) 训练框架:
- 观测保持 H 版 161 维不变
- 动作 = 21 关节绝对位置 scale 0.25
- 奖励使用 T800 框架自带那套 (不覆盖 RewardsCfg)
- 终止按 H 版规范
- 物理 500Hz / 策略 100Hz
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from whole_body_tracking.robots.lens110 import LENS110_ACTION_SCALE, LENS110_CFG
from whole_body_tracking.tasks.tracking.config.lens110 import lens110_mdp
from whole_body_tracking.tasks.tracking.mdp import rewards as tracking_rewards
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

LENS110_BODY_NAMES = [
    "pelvis",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_link", "right_elbow_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
]

LENS110_MOTION_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..", "..", "..", "..", "data", "lens110_dance_100hz.npz",
    )
)


@configclass
class Lens110ObservationsCfg:
    """H 版观测: 161 维。"""

    @configclass
    class PolicyCfg(ObsGroup):
        root_rot_tan_norm = ObsTerm(func=lens110_mdp.root_rot_tan_norm)
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        ref_root_rot_tan_norm = ObsTerm(
            func=lens110_mdp.ref_root_rot_tan_norm,
            params={"command_name": "motion", "num_steps": 4},
        )
        ref_joint_pos = ObsTerm(
            func=lens110_mdp.ref_joint_pos,
            params={"command_name": "motion", "num_steps": 4},
        )
        foot_contact = ObsTerm(
            func=lens110_mdp.foot_contact_flags,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                ),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        root_rot_tan_norm = ObsTerm(func=lens110_mdp.root_rot_tan_norm)
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        ref_root_rot_tan_norm = ObsTerm(
            func=lens110_mdp.ref_root_rot_tan_norm,
            params={"command_name": "motion", "num_steps": 4},
        )
        ref_joint_pos = ObsTerm(
            func=lens110_mdp.ref_joint_pos,
            params={"command_name": "motion", "num_steps": 4},
        )
        foot_contact = ObsTerm(
            func=lens110_mdp.foot_contact_flags,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                ),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Lens110TerminationsCfg:
    """H 版终止条件。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_data_finish = DoneTerm(
        func=lens110_mdp.motion_data_finish,
        params={"command_name": "motion"},
        time_out=True,
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6981})
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.3}
    )
    deviation_root_pos_w = DoneTerm(
        func=lens110_mdp.deviation_root_pos_w,
        params={"command_name": "motion", "threshold": 1.2},
    )
    deviation_key_body_pos_w = DoneTerm(
        func=lens110_mdp.deviation_key_body_pos_w,
        params={
            "command_name": "motion",
            "threshold": 1.2,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=lens110_mdp.KEY_BODY_NAMES, preserve_order=True
            ),
        },
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis", ".*shoulder_.*_link", ".*elbow_link"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class Lens110FlatEnvCfg(TrackingEnvCfg):
    """Lens110 H 版跟踪环境。"""

    observations: Lens110ObservationsCfg = Lens110ObservationsCfg()
    terminations: Lens110TerminationsCfg = Lens110TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 机器人 + 动作 scale 0.25
        self.scene.robot = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = LENS110_ACTION_SCALE
        # 动作参考
        self.commands.motion.motion_file = LENS110_MOTION_FILE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = LENS110_BODY_NAMES
        self.commands.motion.motion_body_names = LENS110_BODY_NAMES
        self.commands.motion.joint_position_range = (-0.05, 0.05)
        self.commands.motion.min_traj_duration = 40.0
        self.commands.motion.bridge_frames = 20
        # 物理 500Hz / 策略 100Hz / 回合 40s
        self.sim.dt = 1.0 / 500.0
        self.decimation = 5
        self.episode_length_s = 40.0
        self.sim.render_interval = self.decimation
        # T800 奖励: 非脚接触排除名单按 Lens110 关节名
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
            r"(?!left_elbow_link$)(?!right_elbow_link$).+$"
        ]
        # 脚触地时长: 脚刚离地时, 若之前触地 <0.2s 则惩罚 (防止脚没站稳就抬)
        self.rewards.feet_contact_time = RewTerm(
            func=tracking_rewards.feet_contact_time,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                ),
                "threshold": 0.2,
            },
        )
        # 关节角度跟踪 (T800 基础上补充, 治 error_joint_pos 过大)
        self.rewards.motion_dof_pos = RewTerm(
            func=tracking_rewards.motion_dof_pos_error_exp,
            weight=3.0,
            params={"command_name": "motion", "std": 3.0},
        )
        # 框架默认事件里的躯干 body 名按 Lens110 关节名适配
        self.events.base_com.params["asset_cfg"].body_names = "torso_yaw_link"
