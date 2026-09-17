"""G1-specific DWAQ environment configuration.

Reward weights are adapted from the reference DreamWaQ G1 config.
"""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.dwaq.dwaq_env_cfg import LocomotionDWAQEnvCfg

import legged_lab.tasks.locomotion.dwaq.mdp as mdp


@configclass
class G1ObservationsCfg:
    """Observation groups tailored for G1 DWAQ.

    Matches the reference DreamWaQ G1 observation structure:
    - Actor (policy): proprioceptive + gait phase sin/cos (blind, no velocity)
    - Critic: all actor terms + velocity + feet contact + feet pos/vel + force + height
    - obs_history: same as actor, 5-frame stack for β-VAE encoder
    - velocity: base linear velocity for VAE supervision target
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations (blind — no base_lin_vel) + gait phase."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={"period": 0.8, "offset": [0.0, 0.5]})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations.

        Contains all proprioceptive terms (no noise) + gait phase + privileged:
        base_lin_vel, feet contact, feet pos/vel in body frame, contact force,
        root height.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={"period": 0.8, "offset": [0.0, 0.5]})
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
        gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={"period": 0.8, "offset": [0.0, 0.5]})

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


# ------------------------------------------------------------------ #
#  Rewards — ported from reference G1DwaqRewardCfg (DreamWaQ)         #
# ------------------------------------------------------------------ #


@configclass
class G1RewardsCfg:
    """Reward terms for G1 DWAQ blind walking.

    Weights are taken from the reference DreamWaQ G1 config.
    """

    # ---- velocity tracking (higher weight to discourage lazy standing) ----
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ---- base regularisation ----
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
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
            "threshold": 0.2,
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
                    "waist_.*_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    # joint_deviation_legs = RewTerm(
    #     func=mdp.joint_deviation_l1_always,
    #     weight=-0.02,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint"])},
    # )

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
            "period": 0.8,
            "scales": (-0.2, 0.4, -0.2),
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
            "period": 0.8,
            "offset": [0.0, 0.5],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold": 0.55,
            "command_name": "base_velocity",
        },
    )


@configclass
class G1DwaqEnvCfg(LocomotionDWAQEnvCfg):
    """G1 DWAQ flat-terrain environment configuration."""

    observations: G1ObservationsCfg = G1ObservationsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # -- Scene --
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # -- Events --
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"

        # -- Terminations --
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


class G1DwaqEnvCfg_PLAY(G1DwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.curriculum.terrain_levels = None

        self.observations.policy.enable_corruption = False
        self.observations.obs_history.enable_corruption = False

        self.events.base_external_force_torque = None
        self.events.push_robot = None
