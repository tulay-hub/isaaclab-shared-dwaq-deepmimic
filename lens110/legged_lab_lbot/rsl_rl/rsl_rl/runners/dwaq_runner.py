from __future__ import annotations

import warnings

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.dwaq_ppo import DWAQPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.modules.actor_critic_dwaq import ActorCriticDWAQ
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.storage import RolloutStorage


class DWAQRunner(OnPolicyRunner):
    """On-policy runner for DreamWAQ (DWAQ) training.

    This runner extends :class:`OnPolicyRunner` with support for the DWAQ
    algorithm, which adds a β-VAE context encoder to PPO. The environment must
    provide the following observation groups (via the ``obs_groups`` config):

    - ``"policy"``:      proprioceptive actor observations.
    - ``"critic"``:      privileged critic observations.
    - ``"obs_history"``: flattened observation history for the VAE encoder.
    - ``"velocity"``:    root linear velocity for VAE velocity supervision.

    Example ``train_cfg`` snippet::

        train_cfg = {
            "policy": {
                "class_name": "ActorCriticDWAQ",
                "cenet_out_dim": 19,
                "activation": "elu",
                "init_noise_std": 1.0,
            },
            "algorithm": {
                "class_name": "DWAQPPO",
                "beta": 1.0,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 1e-3,
            },
            "obs_groups": {
                "policy": ["policy"],
                "critic": ["critic"],
                "obs_history": ["obs_history"],
                "velocity": ["velocity"],
            },
            "num_steps_per_env": 24,
            "save_interval": 50,
        }
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Ensure DWAQ-required algorithm keys exist so the Logger (which accesses
        # cfg["algorithm"]["rnd_cfg"]) does not raise a KeyError.
        train_cfg["algorithm"].setdefault("rnd_cfg", None)
        super().__init__(env, train_cfg, log_dir, device)

    # ------------------------------------------------------------------ #
    #  Overrides                                                           #
    # ------------------------------------------------------------------ #

    def _get_default_obs_sets(self) -> list[str]:
        """DWAQ requires critic, obs_history, and velocity observation sets."""
        return ["critic", "obs_history", "velocity"]

    def _construct_algorithm(self, obs: TensorDict) -> DWAQPPO:
        """Build the ActorCriticDWAQ policy, storage, and DWAQPPO algorithm."""
        # Resolve RND / symmetry configs (inherited from PPO pipeline)
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Build policy
        policy_class = eval(self.policy_cfg.pop("class_name", "ActorCriticDWAQ"))
        actor_critic: ActorCriticDWAQ = policy_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Build storage (reuses the standard RolloutStorage with TensorDict)
        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        # Build algorithm (DWAQPPO inherits PPO, so all PPO kwargs are accepted)
        alg_class = eval(self.alg_cfg.pop("class_name", "DWAQPPO"))
        alg: DWAQPPO = alg_class(
            actor_critic, storage, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **self.alg_cfg
        )

        return alg

    # ------------------------------------------------------------------ #
    #  Save / Load (handle VAE optimizer in addition to RL optimizer)      #
    # ------------------------------------------------------------------ #

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "vae_optimizer_state_dict": self.alg.vae_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg_cfg.get("rnd_cfg"):
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if self.alg_cfg.get("rnd_cfg"):
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "vae_optimizer_state_dict" in loaded_dict:
                self.alg.vae_optimizer.load_state_dict(loaded_dict["vae_optimizer_state_dict"])
            if self.alg_cfg.get("rnd_cfg"):
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
