"""Success-gated terrain curriculum for LENS110."""

import torch


def terrain_levels_completed(env, env_ids):
    terrain = env.scene.terrain
    if not hasattr(env, "_dwaq_terrain_successes"):
        env._dwaq_terrain_successes = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    # Alive reward accumulates actual simulated time, unlike randomized
    # episode_length_buf at runner startup. This also excludes initial reset.
    alive_weight = env.reward_manager.get_term_cfg("alive").weight
    elapsed = env.reward_manager._episode_sums["alive"][env_ids] / alive_weight
    distance = torch.linalg.vector_norm(
        env.scene["robot"].data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=-1
    )
    failed = env.termination_manager.terminated[env_ids] & (elapsed > 0)
    completed = (
        env.termination_manager.time_outs[env_ids]
        & ~env.termination_manager.terminated[env_ids]
        & (elapsed >= env.max_episode_length_s - env.step_dt * 2)
        & (distance > terrain.cfg.terrain_generator.size[0] / 2)
    )
    previous = env._dwaq_terrain_successes[env_ids]
    streak = torch.where(completed, previous + 1, torch.zeros_like(previous))
    move_up = streak >= 2
    env._dwaq_terrain_successes[env_ids] = torch.where(move_up, torch.zeros_like(streak), streak)
    terrain.update_env_origins(env_ids, move_up, failed)
    return terrain.terrain_levels.float().mean()
