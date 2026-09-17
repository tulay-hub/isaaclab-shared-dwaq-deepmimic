"""Segmented velocity commands for LENS110 locomotion."""

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand


class SegmentedVelocityCommand(UniformVelocityCommand):
    """Resample a movement mode with stable speeds and straight-line heading lock."""

    FORWARD_SPEED_RANGE = (0.25, 0.35)
    BACKWARD_SPEED_RANGE = (-0.30, -0.20)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        n = len(env_ids)
        r = torch.rand(n, device=self.device)
        # 20% stand, 30% forward, 25% forward turn, 15% backward, 10% lateral.
        standing = r < 0.20
        forward = (r >= 0.20) & (r < 0.50)
        turning = (r >= 0.50) & (r < 0.75)
        backward = (r >= 0.75) & (r < 0.90)
        lateral = r >= 0.90
        commands = torch.zeros(n, 3, device=self.device)
        forward_speed = torch.empty(n, device=self.device).uniform_(*self.FORWARD_SPEED_RANGE)
        backward_speed = torch.empty(n, device=self.device).uniform_(*self.BACKWARD_SPEED_RANGE)
        commands[:, 0] = torch.where(forward | turning, forward_speed, commands[:, 0])
        commands[:, 0] = torch.where(backward, backward_speed, commands[:, 0])
        commands[:, 1] = torch.where(lateral, self.vel_command_b[env_ids, 1], commands[:, 1])
        commands[:, 2] = torch.where(turning, self.vel_command_b[env_ids, 2], commands[:, 2])
        self.vel_command_b[env_ids] = commands
        self.is_standing_env[env_ids] = standing

        # For straight forward/backward segments, hold the heading that was
        # present when the command was sampled.  Turning and lateral segments
        # keep the sampled yaw-rate command instead.
        straight = forward | backward
        straight_env_ids = env_ids[straight]
        self.heading_target[straight_env_ids] = self.robot.data.heading_w[straight_env_ids]
        self.is_heading_env[env_ids] = straight
