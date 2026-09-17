from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg
from .amp_cfg import RslRlAmpCfg

#########################
# Policy configurations #
#########################

@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with convolutional layers."""

    class_name: str = "ActorCriticConv2d"
    """The policy class name. Default is ActorCriticConv2d."""

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    """List of convolutional layer parameters for the convolutional network."""

    conv_linear_output_size: int = 16
    """Output size of the linear layer after the convolutional features are flattened."""

############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm."""
    
    class_name: str = "PPOAmp"
    """The algorithm class name. Default is PPOAmp."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""


@configclass
class RslRlDwaqActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the DWAQ actor-critic with β-VAE context encoder."""

    class_name: str = "ActorCriticDWAQ"
    """The policy class name. Default is ActorCriticDWAQ."""

    cenet_out_dim: int = 19
    """Dimension of the context encoder output (velocity_dim + latent_dim)."""

    velocity_dim: int = 3
    """Dimension of the velocity branch in the context encoder."""

    encoder_hidden_dims: list[int] = [128]
    """Hidden dimensions of the VAE encoder MLP."""

    encoder_latent_dim: int = 64
    """Dimension of the encoder intermediate latent space."""

    decoder_hidden_dims: list[int] = [64, 128]
    """Hidden dimensions of the VAE decoder MLP."""


@configclass
class RslRlDwaqAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the DWAQ algorithm (PPO + β-VAE)."""

    class_name: str = "DWAQPPO"
    """The algorithm class name. Default is DWAQPPO."""

    beta: float = 1.0
    """β coefficient for the KL divergence term in the VAE loss."""

    vae_learning_rate: float = 1e-3
    """Learning rate for the VAE optimizer (separate from the RL optimizer)."""


#########################
# Runner configurations #
#########################

    
