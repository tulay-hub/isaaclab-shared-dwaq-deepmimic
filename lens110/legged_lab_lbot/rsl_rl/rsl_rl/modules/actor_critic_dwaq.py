from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.context_vae import ContextVAE, ContextVAEOutput
from rsl_rl.networks import MLP


class ActorCriticDWAQ(ActorCritic):
    """Actor-Critic with DreamWAQ (DWAQ) context encoder.

    Inherits from :class:`ActorCritic` and adds a :class:`ContextVAE` (β-VAE)
    that infers velocity and a latent state from a flattened observation
    history.  The inferred latent code is concatenated with the current policy
    observations and fed to the actor MLP.  The critic operates on privileged
    observations only (asymmetric actor-critic).

    Required ``obs_groups`` keys (in addition to ``"policy"`` and ``"critic"``
    from the parent):
        - ``"obs_history"``: flattened observation history for the VAE encoder.
        - ``"velocity"``:    root linear velocity used as the VAE velocity
          supervision target.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # DWAQ-specific parameters
        cenet_out_dim: int = 19,
        velocity_dim: int = 3,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128,),
        encoder_latent_dim: int = 64,
        decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
        # ActorCritic parameters (forwarded to parent)
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        # Build critic, normalizers, action noise via parent.
        # The parent also builds an actor MLP — we replace it below with the
        # correct input dimension (code_dim + policy_obs_dim).
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        # -- DWAQ-specific state -------------------------------------------
        self.cenet_out_dim = cenet_out_dim
        self.velocity_dim = velocity_dim

        num_policy_obs = sum(obs[k].shape[-1] for k in obs_groups["policy"])
        num_obs_history = sum(obs[k].shape[-1] for k in obs_groups["obs_history"])
        self.obs_dim = num_policy_obs

        # -- Replace actor MLP: input = [VAE code | policy obs] ------------
        actor_input_dim = cenet_out_dim + num_policy_obs
        self.actor = MLP(actor_input_dim, num_actions, list(actor_hidden_dims), activation)
        print(f"DWAQ Actor MLP: {self.actor}")

        # -- Context encoder (β-VAE) --------------------------------------
        self.context_vae = ContextVAE(
            input_dim=num_obs_history,
            output_dim=num_policy_obs,
            code_dim=cenet_out_dim,
            velocity_dim=velocity_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            encoder_latent_dim=encoder_latent_dim,
            decoder_hidden_dims=decoder_hidden_dims,
            activation=activation,
        )
        print(
            f"DWAQ ContextVAE: code_dim={cenet_out_dim} "
            f"(vel={velocity_dim} + latent={cenet_out_dim - velocity_dim})"
        )

    # ------------------------------------------------------------------ #
    #  Observation helpers (DWAQ-specific)                                 #
    # ------------------------------------------------------------------ #

    def get_policy_obs(self, obs: TensorDict) -> torch.Tensor:
        """Alias for :meth:`get_actor_obs` (inherited from ActorCritic)."""
        return self.get_actor_obs(obs)

    def load_state_dict(self, state_dict, strict=True):
        state_dict = dict(state_dict)
        if self.noise_std_type == "log" and "std" in state_dict:
            std = state_dict.pop("std")
            if not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Cannot migrate invalid checkpoint action std")
            state_dict["log_std"] = std.clamp(0.05, 2.0).log()
        return super().load_state_dict(state_dict, strict=strict)

    def get_obs_history(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[k] for k in self.obs_groups["obs_history"]], dim=-1)

    def get_velocity(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[k] for k in self.obs_groups["velocity"]], dim=-1)

    # ------------------------------------------------------------------ #
    #  Context encoder delegation                                          #
    # ------------------------------------------------------------------ #

    def cenet_forward(self, obs_history: torch.Tensor) -> ContextVAEOutput:
        """Full forward pass through the context encoder (encode + decode).

        Returns a :class:`ContextVAEOutput` with code, velocity/latent samples,
        reconstruction, and posterior statistics.
        """
        return self.context_vae(obs_history)

    # ------------------------------------------------------------------ #
    #  Overridden core interface                                           #
    # ------------------------------------------------------------------ #

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        policy_obs = self.get_actor_obs(obs)
        policy_obs = self.actor_obs_normalizer(policy_obs)
        code = self.context_vae.encode(self.get_obs_history(obs))
        self._update_distribution(torch.cat([code, policy_obs], dim=-1))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        policy_obs = self.get_actor_obs(obs)
        policy_obs = self.actor_obs_normalizer(policy_obs)
        code = self.context_vae.encode(self.get_obs_history(obs))
        return self.actor(torch.cat([code, policy_obs], dim=-1))
