# 六个项目的奖励框架结构

本文档把“奖励项是什么、代码在哪里、如何影响策略”固定下来。权重来自当前配置源码；实验日志中的历史
权重只用于比较，不会因为目录整理被自动改写。

## 统一执行链

```text
environment state / reference motion / sensor
        |
        v
MDP function (mdp/rewards.py or config-local rewards.py)
        |
        v
RewardTermCfg(name, func, params, weight)
        |
        v
RewardManager: sum(weight_i * term_i)
        |
        +--> PPO value/surrogate update
        +--> AMP style discriminator objective
        +--> DWAQ PPO + beta-VAE objective
```

普通环境总奖励可写成：

```text
R_env(t) = sum_i weight_i * r_i(t)
```

参考跟踪项通常使用：

```text
r_track = exp(-||error||^2 / std^2)
```

`weight` 决定该目标在环境奖励中的相对梯度；终止条件不是奖励项，但会通过 episode 截断、termination penalty
或 AMP delay mask 改变训练信号。AMP 的 discriminator style reward、DWAQ 的 VAE loss 都属于算法层目标，
不能只看 `RewardTermCfg` 判断总优化目标。

## 1. 跳舞全身：DeepMimic 161-D H 版

代码：
`frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/lens110/lens110_deepmimic_env_cfg.py`

### 正向参考跟踪

| term | weight | 实现意图 |
|---|---:|---|
| `ref_track_root_pos_w_error_exp` | `+0.15` | 根位置跟踪 |
| `ref_track_quat_error_exp` | `+0.15` | 机身姿态跟踪 |
| `ref_track_root_vel_w_error_exp` | `+0.10` | 根线速度跟踪 |
| `ref_track_root_ang_vel_w_error_exp` | `+0.05` | 根角速度跟踪 |
| `ref_track_key_body_pos_b_error_exp` | `+0.30` | 踝/肘/肩关键点跟踪；H 版可保留奖励而删除对应观测 |
| `ref_track_dof_pos_error_exp` | `+0.80` | 21 个关节角跟踪，主跟踪项 |
| `ref_track_dof_vel_error_exp` | `+0.10` | 关节速度跟踪 |
| `alive` | `+0.20` | 轻量存活信号 |

### 正则与动作平滑

| term | weight | 说明 |
|---|---:|---|
| `dof_torques_l2` | `-1e-6` | 力矩平方 |
| `dof_acc_l2` | `-2.5e-8` | 关节加速度平方 |
| `action_rate_l2` | `-0.001` | 使用 `action_rate_l2_scaled`，先乘 `scale=0.25` 再平方 |

`action_rate_l2_scaled` 是必要的尺度修正：若直接惩罚 raw action，scale=0.25 时相同关节目标变化需要更大的
raw action，平方惩罚会被放大约 16 倍，策略会变慢并增加摔倒。

### 终止门控

`base_contact`、`base_height`、`bad_orientation`、`motion_data_finish`、根位置偏差和关键点偏差共同决定
回合是否结束。H 版当前关键阈值包括：`base_height=0.3 m`、姿态约 `40 deg`、根偏差 `1.2 m`、关键点偏差
`1.5 m`。这些是 termination，不要误列为正向奖励。

## 2. 跳舞半身：AMP + upper/lower IO

代码：
`projects/02_dance_half_body/framework/legged_lab_upper_lower/source/legged_lab/legged_lab/tasks/locomotion/amp/config/lens110/`

### 任务奖励层

| term | weight | 作用 |
|---|---:|---|
| `track_lin_vel_xy_exp` | `+1.25` | 根平面速度 |
| `track_ang_vel_z_exp` | `+1.25` | 根偏航速度 |
| `alive` | `+0.10` | 生存 |
| `ang_vel_xy_l2` | `-0.10` | 抑制横滚/俯仰角速度 |
| `flat_orientation_l2` | `-1.0` | 保持身体平整 |
| `base_height` | `-2.0` | 靠近 `0.66 m` 站高 |
| `joint_vel_l2` | `-2e-4` | 速度正则 |
| `joint_acc_l2` | `-2.5e-7` | 加速度正则 |
| `action_rate_l2` | `-0.01` | 动作平滑 |
| `joint_pos_limits` | `-1.0` | 关节限位 |
| `joint_torques_l2` | `-1e-5` | 力矩 |
| `joint_regularization` | `-2e-3` | 默认姿态偏差 |
| `joint_deviation_hip` / `arms` / `torso` | `-0.03/-0.025/-0.01` | 局部姿态约束 |
| `arm_symmetry` / `arm_vel_symmetry` | `-0.04/-0.001` | 左右臂对称性 |
| `hip_yaw_symmetry` | `-0.15` | 髋偏航对称性 |
| `feet_air_time` | `+0.60` | 步态节奏 |
| `feet_slide` | `-0.12` | 防滑步 |
| `sound_suppression` | `-1e-4` | 落脚冲击/声音代理 |
| `feet_distance` / `knee_distance` | `+0.15/+0.10` | 步宽与膝间距 |
| `undesired_contacts` | `-1.0` | 非脚接触 |
| `termination_penalty` | `-1.0` | 终止惩罚 |

### upper/lower 转换层

奖励比较的是 policy-facing upper/lower 状态或参考状态；执行器落地前再通过 polynomial、weighted MLP、XML
tendon solver 或直接 MJCF plant 转换。半身实验的核心不是新增一项“半身奖励”，而是让 ankle state/action
在训练、参考 motion、MuJoCo 和部署执行器之间处于同一坐标语义。

### AMP 算法层

`PPOAMP` 之外还有 AMP discriminator：`style_reward_scale=5.0`、`task_style_lerp=0.4`，并使用 LSGAN；
upper/lower policy 配置关闭了共享对称增强，因为映射后的左右执行器语义需要单独验证。

## 3. 行走：DWAQ 平地

代码：
`frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/tasks/locomotion/dwaq/config/lens110/lens110_dwaq_env_cfg.py`

| 分组 | term | weight | 作用 |
|---|---|---:|---|
| 速度 | `track_lin_vel_xy_exp` | `+2.5` | 平面速度跟踪 |
| 速度 | `track_ang_vel_z_exp` | `+3.0` | 偏航速度跟踪 |
| 身体 | `lin_vel_z_l2` / `ang_vel_xy_l2` | `-1.0/-0.05` | 抑制竖直/横摆晃动 |
| 身体 | `flat_orientation_l2` / `body_orientation_l2` | `-1.0/-2.0` | 保持躯干姿态 |
| 能耗 | `energy` | `-1e-3` | 力矩/速度能耗代理 |
| 平滑 | `joint_acc_l2` / `action_rate_l2` | `-2.5e-7/-0.01` | 加速度和动作变化 |
| 限位 | `dof_pos_limits` | `-2.0` | 关节限位 |
| 接触 | `undesired_contacts` / `fly` | `-1.0/-1.0` | 非期望接触、双脚离地 |
| 脚 | `feet_air_time` / `feet_slide` / `feet_force` | `+0.15/-0.25/-0.003` | 步态、滑动、冲击 |
| 脚 | `feet_too_near` / `feet_stumble` / `feet_heading_alignment` | `-2.0/-2.0/-2.0` | 防绊倒、脚距和脚朝向 |
| 姿态 | `joint_deviation_hip` / `ankle` / `arms` | `-0.3/-0.2/-0.2` | 防止异常关节姿态 |
| 生存 | `termination_penalty` / `alive` | `-200.0/+0.15` | 强终止惩罚与存活 |
| 防偷懒 | `idle_penalty` | `-2.0` | 有速度命令时不能原地不动 |
| 步态先验 | `leg_ref_joint_pos` / `gait_phase_contact` | `+0.5/+0.2` | 腿部周期和接触相位 |

DWAQ 还在算法层使用 `ActorCriticDWAQ`、`DWAQPPO` 和 beta-VAE。VAE 的 velocity/reconstruction/KL 项不是
环境 `RewardTermCfg`，而是在 `rsl_rl/rsl_rl/algorithms/dwaq_ppo.py` 的优化步骤中单独更新。

## 4. 上台阶：DWAQ + stair curriculum

楼梯项目复用行走的奖励表，新增的是地形与课程机制，不是把 `terrain_levels` 当成奖励百分比：

```text
LENS110_RANDOM_STAIRS_ROUGH_TERRAINS_CFG
  -> randomized connected path
  -> difficulty controls 5 cm .. 30 cm surface range
  -> start at terrain row 0
  -> timeout + enough travel + no failure = success
  -> two consecutive successes = promote
  -> fall/failure = demote
```

独立配置：
`frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/tasks/locomotion/dwaq/config/lens110/lens110_stairs_dwaq_env_cfg.py`。
它复制地形配置后再开启 `terrain_levels_completed`，避免修改平地 DWAQ 的 module-level terrain singleton。

## 5. 跌倒起身：AMP GetUp

代码：
`projects/04_fall_to_stand/framework/amp_mjlab/AMP_mjlab/src/tasks/amp_loco/config/lens110/env_cfgs.py`

### 任务层奖励

| term | weight | 作用 |
|---|---:|---|
| `track_anchor_linear_velocity` | `+1.0` | 零速度站立目标；恢复 delay 期间可 mask |
| `track_anchor_angular_velocity` | `+1.0` | 零角速度目标 |
| `track_root_height` | `+5.0` | 骨盆回到站高，整项放大但保持单调 |
| `torso_upright` | `+2.0` | 防止只抬骨盆而躯干仍趴着 |
| `body_ang_vel_xy_l2` | `+0.5` 基础项 | 身体角速度约束，按当前配置的符号/实现解释 |
| `is_terminated` | `-200.0` | 终止惩罚 |
| `joint_acc_l2` | `-2.5e-7` | 加速度 |
| `joint_pos_limits` | `-10.0` | 限位 |
| `action_rate_l2` | `-0.01` | 平滑 |
| `foot_slip_stand` | `-2.0` | 零速度命令下仍惩罚接触脚滑动 |
| `over_height_air` | `-20.0` | 防止蹬地跳高骗站高分 |
| `feet_sole_flat` | `+0.6` | 仅在高度、水平速度、双脚接触三重门控同时满足时生效 |
| `self_collisions` | `-0.1` | 自碰撞 |

起身配置把 `foot_slip` 替换成 `foot_slip_stand`，因为原始 feet slip 会被零速度 command gate 直接关掉；
同时把脚掌平整奖励放到站定阶段，避免恢复途中锁死踝关节。

### AMP 层

`lens110_amp_ppo_runner_cfg()` 采用 AMP actor/critic 和 discriminator；`amp_reward_coef=0.1`、
`amp_task_reward_lerp=0.85`，tracked bodies 是骨盆、髋/膝/踝和肩/肘关键刚体。站立动作与 Recovery 动作
分别从 `Stand/` 和 `Recovery/` 目录加载。

## 6. 翻滚：SideRoll

代码：
`frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/lens110/lens110_deepmimic_env_cfg_sideroll.py`

SideRoll 继承 159-D DeepMimic 的跟踪奖励，针对滚地阶段做以下结构性调整：

- `base_contact=None`、`bad_orientation=None`：横躺和躯干接触是动作的一部分；
- `base_height.minimum_height=0.02`：允许贴地滚动；
- 根/关键点偏差阈值收紧为 `0.8 m`，防止躺平后无限活着；
- `alive` 降为 `0.05`；
- 摩擦随机范围 `0.5..1.5`，骨盆质量随机 `+-0.5 kg`；
- 新增 `standing_still`，参考速度低于 `0.1 m/s` 时以 `-2.0` 惩罚根部滑动，滚动段自动关闭。

因此 SideRoll 不是“把普通舞蹈动作改名”，而是同一个 159-D IO 下的专用终止/正则化框架。

## 修改奖励后的最低验证

```bash
python -m py_compile <changed_reward_or_config.py>
git diff --check
```

然后在对应项目 README 的环境中做配置加载、1 iteration smoke test、固定动作回放；对真机导出还要做 ONNX
输入/输出和部署 YAML 的逐项比对。
