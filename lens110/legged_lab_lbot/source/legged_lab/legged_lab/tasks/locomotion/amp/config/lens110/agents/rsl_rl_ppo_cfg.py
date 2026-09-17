from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg

from legged_lab.rsl_rl import RslRlAmpCfg, RslRlPpoAmpAlgorithmCfg
from legged_lab.tasks.locomotion.amp.mdp.symmetry import lens110


@configclass
class Lens110RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 16
    max_iterations = 200000
    save_interval = 100
    experiment_name = "lens110_amp"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )

    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=64,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[512, 256],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.4,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=lens110.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
        ),
    )


@configclass
class Lens110UpperLowerRslRlOnPolicyRunnerAmpCfg(Lens110RslRlOnPolicyRunnerAmpCfg):
    experiment_name = "lens110_amp_upper_lower"

    def __post_init__(self):
        self.algorithm.symmetry_cfg = None


@configclass
class Lens110MlpUpperLowerRslRlOnPolicyRunnerAmpCfg(Lens110UpperLowerRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "lens110_amp_mlp_upper_lower"


@configclass
class Lens110XmlTendonUpperLowerRslRlOnPolicyRunnerAmpCfg(Lens110UpperLowerRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "lens110_amp_xml_tendon_upper_lower"


@configclass
class Lens110MjcfUpperLowerRslRlOnPolicyRunnerAmpCfg(Lens110UpperLowerRslRlOnPolicyRunnerAmpCfg):
    experiment_name = "lens110_amp_mjcf_upper_lower"
    clip_actions = 1.0
