from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from legged_lab.rsl_rl import RslRlDwaqActorCriticCfg, RslRlDwaqAlgorithmCfg


@configclass
class G1DwaqRunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL runner configuration for G1 DWAQ training."""

    class_name = "DWAQRunner"
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "g1_dwaq"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "obs_history": ["obs_history"],
        "velocity": ["velocity"],
    }

    policy = RslRlDwaqActorCriticCfg(
        class_name="ActorCriticDWAQ",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        cenet_out_dim=19,
        velocity_dim=3,
        encoder_hidden_dims=[128],
        encoder_latent_dim=64,
        decoder_hidden_dims=[64, 128],
    )
    algorithm = RslRlDwaqAlgorithmCfg(
        class_name="DWAQPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        beta=1.0,
        vae_learning_rate=1.0e-3,
    )
