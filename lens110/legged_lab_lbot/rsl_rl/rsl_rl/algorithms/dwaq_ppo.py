from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_dwaq import ActorCriticDWAQ
from rsl_rl.modules.context_vae import ContextVAEOutput
from rsl_rl.storage import RolloutStorage


class DWAQPPO(PPO):
    """Proximal Policy Optimization with a β-VAE context encoder (DreamWAQ).
    """

    policy: ActorCriticDWAQ

    def __init__(
        self,
        policy: ActorCriticDWAQ,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        # DWAQ-specific parameters
        beta: float = 1.0,
        vae_learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(
            policy=policy,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.beta = beta

        # Following the official DreamWaQ: separate optimizers for RL and VAE.
        # Replace the single optimizer created by PPO.__init__ with two.
        rl_parameters = (
            list(self.policy.actor.parameters())
            + list(self.policy.critic.parameters())
            + [self.policy.std if self.policy.noise_std_type == "scalar" else self.policy.log_std]
        )
        self.optimizer = optim.Adam(rl_parameters, lr=learning_rate)
        self.vae_optimizer = optim.Adam(
            self.policy.context_vae.parameters(), lr=vae_learning_rate
        )

    def _step_if_finite(self, loss, optimizer):
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss).all():
            return False
        loss.backward()
        params = [p for group in optimizer.param_groups for p in group["params"]]
        norm = nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            return False
        optimizer.step()
        return True

    # ------------------------------------------------------------------ #
    #  Update (PPO + β-VAE autoencoder loss — two separate passes)        #
    # ------------------------------------------------------------------ #

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_autoenc_loss = 0.0
        skipped_rl = 0
        skipped_vae = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        # Pre-compute o_{t+1} for VAE reconstruction target (paper Eq. 7).
        T = self.storage.num_transitions_per_env
        N = self.storage.num_envs
        policy_keys = self.policy.obs_groups["policy"]
        policy_obs_seq = torch.cat(
            [self.storage.observations[k] for k in policy_keys], dim=-1
        )  # [T, N, D_policy]
        self.storage.observations["_next_policy_obs"] = torch.cat(
            [policy_obs_seq[1:], policy_obs_seq[-1:]], dim=0
        )

        # live_mask: 0 where episode ended (dones=1), used for vel/KL losses
        live_mask = 1.0 - self.storage.dones.float()  # [T, N, 1]
        self.storage.observations["_live_mask"] = live_mask

        # recon_mask: live_mask + last step also masked (no valid o_{t+1})
        recon_mask = live_mask.clone()
        recon_mask[-1] = 0.0
        self.storage.observations["_recon_mask"] = recon_mask

        # Mini-batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute policy outputs
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Adaptive learning-rate schedule (RL optimizer only)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ============================================================ #
            #  Pass 1: PPO loss → update RL parameters only                #
            # ============================================================ #
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            rl_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                symmetry_loss = nn.MSELoss()(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    rl_loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if not self._step_if_finite(rl_loss, self.optimizer):
                skipped_rl += 1
            if self.policy.noise_std_type == "log":
                with torch.no_grad():
                    self.policy.log_std.clamp_(-3.0, 0.6931471805599453)

            # RND (inherited from PPO, typically unused in DWAQ)
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = nn.MSELoss()(predicted_embedding, target_embedding)
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

            # ============================================================ #
            #  Pass 2: VAE loss → update encoder/decoder only              #
            # ============================================================ #
            obs_history = self.policy.get_obs_history(obs_batch)
            vae_out: ContextVAEOutput = self.policy.cenet_forward(obs_history)

            vel_target = self.policy.get_velocity(obs_batch).detach()
            decode_target = obs_batch["_next_policy_obs"].detach()
            live = obs_batch["_live_mask"].detach()
            recon_m = obs_batch["_recon_mask"].detach()

            # Velocity MSE — masked by live (skip terminated envs)
            vel_loss = nn.MSELoss()(vae_out.code_vel * live, vel_target * live)

            # Reconstruction MSE — masked by recon_mask (skip terminated + last step)
            recon_loss = nn.MSELoss()(vae_out.reconstruction * recon_m, decode_target * recon_m)

            # KL divergence — masked by live
            logvar = vae_out.logvar_latent.clamp(-10.0, 10.0)
            kl_divergence = -0.5 * torch.mean(
                torch.sum(1 + logvar - vae_out.mean_latent.pow(2) - logvar.exp(), dim=-1)
                * live.squeeze(-1)
            )

            autoenc_loss = vel_loss + recon_loss + self.beta * kl_divergence

            if not self._step_if_finite(autoenc_loss, self.vae_optimizer):
                skipped_vae += 1

            # ---- Accumulate metrics --------------------------------------
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_autoenc_loss += autoenc_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Average over all updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_autoenc_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Remove temporary keys
        del self.storage.observations["_next_policy_obs"]
        del self.storage.observations["_live_mask"]
        del self.storage.observations["_recon_mask"]
        self.storage.clear()

        loss_dict = {
            "skipped_rl_updates": skipped_rl,
            "skipped_vae_updates": skipped_vae,
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "autoencoder": mean_autoenc_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
