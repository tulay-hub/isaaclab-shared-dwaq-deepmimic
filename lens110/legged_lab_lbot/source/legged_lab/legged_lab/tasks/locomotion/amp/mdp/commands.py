from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class UniformVelocityCommandwithZeroProb(UniformVelocityCommand):
    """Uniform velocity command with independent zero probability per velocity axis."""

    cfg: UniformVelocityCommandwithZeroProbCfg

    def __init__(self, cfg: UniformVelocityCommandwithZeroProbCfg, env):
        super().__init__(cfg, env)
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)

        r = torch.empty(len(env_ids), device=self.device)
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

    def _update_command(self):
        super()._update_command()

        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0


@configclass
class UniformVelocityCommandwithZeroProbCfg(UniformVelocityCommandCfg):
    """Uniform velocity command configuration with independent zero probability per axis."""

    class_type: type = UniformVelocityCommandwithZeroProb

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for x, y, and yaw commands."""

    ranges: Ranges = MISSING
