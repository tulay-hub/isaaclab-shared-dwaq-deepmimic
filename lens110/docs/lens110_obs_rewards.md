# Lens110 DeepMimic 当前观测项与奖励项（H 版，161 维）

> 任务：`LeggedLab-Isaac--Deepmimic-Lens110-v0`（唐伯虎点秋香舞蹈模仿）
> 配置来源：`source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/lens110/lens110_deepmimic_env_cfg.py`（H 版，真机部署）
> 日期：2026-08-27

---

## 一、观测项（Policy 观测组，共 161 维）

| # | 观测项 | 函数 | 维度 | 内容 | 真机来源 |
|---|---|---|---|---|---|
| 1 | `root_rot_tan_norm` | 自定义（deepmimic mdp） | 6 | 机身旋转矩阵 tan/norm 列（姿态表示） | IMU 姿态解算 |
| 2 | `root_ang_vel_w` | isaaclab `root_ang_vel_w` | 3 | 根部角速度（世界系） | IMU 陀螺仪 |
| 3 | `joint_pos` | isaaclab `joint_pos` | 21 | 全部关节位置 | 关节编码器 |
| 4 | `joint_vel` | isaaclab `joint_vel` | 21 | 全部关节速度 | 关节编码器 |
| 5 | `ref_root_rot_tan_norm` | 自定义 | 24 | 参考运动朝向（4 步 × 6） | 离线舞蹈 pkl |
| 6 | `ref_joint_pos` | 自定义 | 84 | 参考运动关节位置（4 步 × 21） | 离线舞蹈 pkl |
| 7 | `foot_contact` | 自定义 `foot_contact_flags` | 2 | 左右脚触地标志（sensor_cfg 已修复，只取踝部） | 接触检测/电机电流估计 |

**H 版已删除的观测**（与 F 版差异）：`key_body_pos_b`（18 维，当前状态关键点）、`ref_key_body_pos_b`（72 维，参考关键点）

**更早删除的观测**：`root_vel_w`（3，世界系线速度）、`root_height`（1，机身绝对高度）、`ref_root_pos_error`（12，参考位置误差）

### 维度核算

```
161 = 6 + 3 + 21 + 21 + 24 + 84 + 2
```

> 参考类观测（`ref_*`）数据来自离线舞蹈片段 `lens110_tangbohushuo_50hz_reorder.pkl`（1812 帧 @50Hz），每控制步（0.02s）推进 1 帧。

### F 版（备份，251 维）

F 版 = H 版 + `key_body_pos_b`（18）+ `ref_key_body_pos_b`（72）= 251 维。配置文件：`lens110_deepmimic_env_cfg_keybody.py`，任务 ID：`LeggedLab-Isaac--Deepmimic-Lens110-KeyBody-v0`。

---

## 二、奖励项（10 个）

### 正向跟踪奖励（模仿舞蹈）

| # | 奖励项 | 函数 | 权重 | std | 内容 |
|---|---|---|---|---|---|
| 1 | `ref_track_root_pos_w_error_exp` | mdp 自定义 | +0.15 | 0.5 | 根位置跟踪（世界系） |
| 2 | `ref_track_quat_error_exp` | mdp 自定义 | +0.15 | 0.5 | 机身姿态四元数跟踪 |
| 3 | `ref_track_root_vel_w_error_exp` | mdp 自定义 | +0.1 | 1.0 | 根线速度跟踪 |
| 4 | `ref_track_root_ang_vel_w_error_exp` | mdp 自定义 | +0.05 | 1.0 | 根角速度跟踪 |
| 5 | `ref_track_key_body_pos_b_error_exp` | mdp 自定义 | +0.3 | 0.3 | 关键点位置跟踪（**H 版仍保留此奖励**，仅删了对应观测） |
| 6 | `ref_track_dof_pos_error_exp` | mdp 自定义 | +0.8 | 2.0 | 关节位置跟踪（权重最大） |
| 7 | `ref_track_dof_vel_error_exp` | mdp 自定义 | +0.1 | 10.0 | 关节速度跟踪 |

### 惩罚项

| # | 奖励项 | 函数 | 权重 | 内容 |
|---|---|---|---|---|
| 8 | `dof_torques_l2` | isaaclab `joint_torques_l2` | -1e-6 | 关节力矩平方惩罚 |
| 9 | `dof_acc_l2` | isaaclab `joint_acc_l2` | -2.5e-8 | 关节加速度平方惩罚 |
| 10 | `action_rate_l2` | **自定义 `action_rate_l2_scaled`** | -0.001 | 动作变化率惩罚（在关节目标空间计算，params `{"scale": 0.25}`） |

> `action_rate_l2` 说明：标准 `mdp.action_rate_l2` 计算在原始动作上，scale=0.25 时惩罚被放大 16 倍导致策略动作变慢、摔倒率飙升；`action_rate_l2_scaled` 把动作差值乘回 scale 再平方（惩罚与 scale 无关）。

---

## 三、观测与奖励配置要点

- **观测无噪声**：`enable_corruption = False`（所有版本均未加观测噪声）
- **无历史**：`history_length = 0`，每步独立观测
- **动作**：21 关节位置目标，`actions.joint_pos.scale = 0.25`
- **PPO 超参**：`init_noise_std = 2.0`（等效关节目标噪声 0.5 rad）、`actor/critic_hidden_dims=[512,256,128]`、`learning_rate=1e-3`、`desired_kl=0.05`、`max_iterations=10000`

### 终止条件（非奖励，供参考）

| 终止项 | 条件 |
|---|---|
| `motion_data_finish` | 舞蹈参考片段耗尽（1812 帧） |
| `bad_orientation` | 机身姿态偏离 > 40°（摔倒） |
| `base_height` | 机身高度 < 0.3 m |
| `deviation_root_pos_w` | 根位置偏离参考 > 1.2 m |
| `deviation_key_body_pos_w` | 关键点偏离参考 > 1.2 m |
| `base_contact` | 骨盆/肩/肘非预期接触 |
| `time_out` | 回合超时（`episode_length_s = 40.0`） |

---

## 四、版本对应关系（参考 `lens110_train_log.md`）

| 版本 | 观测维度 | 说明 | checkpoint 目录 |
|---|---|---|---|
| F | 251 维 | 训练最优（含关键点观测） | `logs/rsl_rl/lens110_deepmimic/2026-08-27_14-32-04` |
| **H（当前）** | **161 维** | **真机部署（删关键点观测）** | `logs/rsl_rl/lens110_deepmimic/2026-08-27_15-49-10` |

验证方式：checkpoint 中 `actor.0.weight` 形状 H 版应为 `[512, 161]`，F 版应为 `[512, 251]`。
