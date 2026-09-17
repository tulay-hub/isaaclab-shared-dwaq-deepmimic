from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Lens110PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    # 固定探索 std=init_noise_std (1.0), 不参与训练更新:
    # 否则 entropy 项会把 std 一路推大 (1 -> 10), 动作失控, 训练崩坏。
    fixed_action_std: bool = True
    # 恢复到 2026-09-01_11-59-20 那版 sim2sim 表现较好的动作约束配置。
    action_mean_l2_coef: float = 0.0


@configclass
class Lens110DeepMimicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 100
    experiment_name = "lens110_deepmimic"
    # 恢复 2026-09-01_11-59-20 配置：不硬裁动作输出，由策略自身学习动作幅值。
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # T800 使用 empirical_normalization=True (等价于下面两项)
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = Lens110PpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        # T800 原始值
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
