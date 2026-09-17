"""LENS110-specific DWAQ environment configuration.

盲走（DreamWAQ）：actor 不接收 base_lin_vel，速度由 β-VAE 从观测历史推断。

身体名映射（相对于 G1 的 DWAQ 配置）：
  G1 ``torso_link`` -> LENS110 ``torso_yaw_link``
  G1 ``waist_.*_joint`` -> LENS110 ``torso_yaw_joint``（LENS110 只有 1 个腰关节）
  G1 ``.*_wrist_.*_joint`` -> LENS110 没有腕关节（删除）
  G1 ``.*_ankle_roll_link`` -> LENS110 ``.*_ankle_roll_link``（一致）

步态参数（LENS110 腿短、步幅小，参考 G1 值做了缩放）：
  G1 ``period=0.8`` -> LENS110 ``period=0.6``
  G1 ``feet_too_near threshold=0.2`` -> LENS110 ``threshold=0.15``
"""

import math
from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legged_lab.data.Robots.model_humanoid_lens110.lens110_dance import LENS110_CFG
from legged_lab.terrains import LENS110_RANDOM_DWAQ_TERRAINS_CFG
from legged_lab.tasks.locomotion.dwaq.dwaq_env_cfg import LocomotionDWAQEnvCfg
import legged_lab.tasks.locomotion.dwaq.mdp as mdp
from legged_lab.tasks.locomotion.dwaq.mdp.commands import SegmentedVelocityCommand


@configclass
class Lens110ObservationsCfg:
    """LENS110 DWAQ observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations (blind — no base_lin_vel) + gait phase."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={"period": 0.6, "offset": [0.0, 0.5]})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations.

        All actor terms + privileged base_lin_vel, feet contact / pos / vel /
        force, root height.  Uses LENS110 body names.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={"period": 0.6, "offset": [0.0, 0.5]})
        # -- privileged --
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        feet_contact = ObsTerm(
            func=mdp.feet_contact_binary,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        feet_pos = ObsTerm(
            func=mdp.feet_pos_body_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
        )
        feet_vel = ObsTerm(
            func=mdp.feet_vel_body_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
        )
        feet_force = ObsTerm(
            func=mdp.feet_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        root_height = ObsTerm(func=mdp.base_pos_z)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ObsHistoryCfg(ObsGroup):
        """Observation history for β-VAE context encoder (5 frames).

        Same terms as PolicyCfg (including gait phase).
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={"period": 0.6, "offset": [0.0, 0.5]})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class VelocityCfg(ObsGroup):
        """Root body-frame linear velocity for VAE supervision."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    obs_history: ObsHistoryCfg = ObsHistoryCfg()
    velocity: VelocityCfg = VelocityCfg()


@configclass
class Lens110RewardsCfg:
    """LENS110 DWAQ reward terms.

    权重以 G1 的 DWAQ 配置为基准，针对 LENS110 的身材做了微调：
    - 步态周期 0.8 -> 0.6（LENS110 腿短，步频更快）
    - feet_too_near 阈值 0.2 -> 0.15（LENS110 腿窄，双脚间距更小）
    - 关节偏差惩罚去掉腕关节（LENS110 没有腕）
    - 腰关节惩罚从 waist_.*_joint 改为 torso_yaw_joint
    """

    # ---- velocity tracking ----
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.35},
    )

    # ---- base regularisation ----
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_yaw_link")},
    )

    # ---- joint regularisation ----
    energy = RewTerm(func=mdp.energy_norm, weight=-1e-3)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    # ---- contacts ----
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"),
            "threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 1.0,
        },
    )

    # ---- feet ----
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 500.0,
            "max_reward": 400.0,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold": 0.10,
        },
    )
    feet_heading_alignment = RewTerm(
        func=mdp.feet_heading_alignment,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "command_name": "base_velocity",
            "min_speed": 0.15,
            "max_yaw_rate": 0.2,
            "max_lateral_command": 0.1,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    # ---- posture (always penalised, not just when standing) ----
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_.*_joint")},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "torso_yaw_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )

    # ---- termination / survival ----
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    alive = RewTerm(func=mdp.alive, weight=0.15)

    # ---- DWAQ anti-lazy ----
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-2.0,
        params={"command_name": "base_velocity", "cmd_threshold": 0.2, "vel_threshold": 0.1},
    )

    # ---- leg-pitch reference trajectory (humanoid-gym style) ----
    leg_ref_joint_pos = RewTerm(
        func=mdp.leg_ref_joint_pos,
        weight=0.5,
        params={
            "left_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint"],
            ),
            "right_cfg": SceneEntityCfg(
                "robot",
                joint_names=["right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"],
            ),
            "period": 0.6,
            "scales": (-0.4, 0.4, -0.3),
            "double_support_threshold": 0.1,
            "command_name": "base_velocity",
            "cmd_threshold": 0.1,
        },
    )

    # ---- gait phase matching (bipedal walking) ----
    gait_phase_contact = RewTerm(
        func=mdp.feet_gait,
        weight=0.2,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold": 0.55,
            "command_name": "base_velocity",
        },
    )


@configclass
class Lens110DwaqEnvCfg(LocomotionDWAQEnvCfg):
    """LENS110 DWAQ flat-terrain environment configuration."""

    observations: Lens110ObservationsCfg = Lens110ObservationsCfg()
    rewards: Lens110RewardsCfg = Lens110RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # LENS110 training/deployment profile: 500 Hz physics, 100 Hz policy.
        self.sim.dt = 0.002
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.episode_length_s = 60.0

        self.commands.base_velocity.class_type = SegmentedVelocityCommand
        self.commands.base_velocity.resampling_time_range = (12.0, 16.0)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.30, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        # Headless training must not request the remote velocity-arrow USD.
        self.commands.base_velocity.debug_vis = False

        # -- Scene --
        self.scene.robot = LENS110_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Restore the original DWAQ training scene used by the first runs.
        self.scene.terrain.terrain_generator = LENS110_RANDOM_DWAQ_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 5
        # Disable terrain-level progression: every generated patch is sampled
        # independently instead of being advanced by walking performance.
        self.curriculum.terrain_levels = None

        # -- Events: base mass / COM / external force on torso --
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_yaw_link"
        self.events.base_com.params["asset_cfg"].body_names = "torso_yaw_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_yaw_link"

        # -- Terminations --
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_yaw_link"


class Lens110DwaqEnvCfg_PLAY(Lens110DwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.curriculum.terrain_levels = None

        self.observations.policy.enable_corruption = False
        self.observations.obs_history.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
