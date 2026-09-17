# 观测、动作、数据与关节顺序契约

不同项目共享同一台 Lens110，但不共享同一个 policy interface。任何导出、回放或部署前都要先确认项目
对应的契约，禁止按文件名或 checkpoint 编号猜维度。

## 1. 坐标与四元数

| 边界 | 约定 |
|---|---|
| GMR pkl/CSV | 根四元数 `xyzw` |
| 部署 CSV | `[root_pos(3), root_rot_xyzw(4), 21 dof]`，每行保留尾逗号 |
| MuJoCo `qpos` | 根四元数 `wxyz`；只在进入 MuJoCo 时转换 |
| Isaac Lab / DeepMimic motion pkl | 以对应配置和保存的 joint/body 顺序为准，不能因 shape 相同而假定顺序相同 |

## 2. 全身舞蹈 H 版

任务：`LeggedLab-Isaac--Deepmimic-Lens110-v0`。

```text
policy observation =
  root_rot_tan_norm  6
+ root_ang_vel_w     3
+ joint_pos         21
+ joint_vel         21
+ ref_root_rot      24   # 未来 4 帧，每帧 6
+ ref_joint_pos     84   # 未来 4 帧，每帧 21
+ foot_contact       2   # 左/右踝
= 161
action = 21
q_des = reference_joint_pos_current + 0.25 * clipped_policy_action
```

H 版删除当前关键点观测、根高度和世界系根线速度，适合 IMU + 关节编码器的部署权衡。F/KeyBody 版仍是
`251 -> 21`，任务 ID 为 `LeggedLab-Isaac--Deepmimic-Lens110-KeyBody-v0`，不得与 H checkpoint 混用。

## 3. 159-D URDF 顺序（侧滚/兼容舞蹈）

```text
root_rot_tan_norm  6
+ root_ang_vel_b   3
+ joint_pos       21
+ joint_vel       21
+ ref_root_rot    24
+ ref_joint_pos   84
= 159
```

策略 joint order 为：

```text
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
right_ankle_pitch, right_ankle_roll,
torso_yaw,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
```

数据集原本使用 USD 交叉顺序时，必须通过配置中的 `DATASET_TO_POLICY` 显式重排；`preserve_order=True` 是这个
变体的关键设置。侧滚任务使用同一个 159-D policy IO，但修改了终止条件和站立静止奖励。

## 4. 半身 upper/lower

半身项目的物理训练模型仍可使用 `ankle_pitch/ankle_roll`，policy-facing IO 将四个踝槽位暴露为：

```text
left_ankle_upper_joint, left_ankle_lower_joint,
right_ankle_upper_joint, right_ankle_lower_joint
```

动作链路为：

```text
policy upper/lower target
  -> polynomial / weighted MLP / XML tendon solver
  -> physical pitch/roll target or torque
  -> Lens110 ankle plant
```

配置变体和任务 ID 在 `projects/02_dance_half_body/framework/legged_lab_upper_lower/source/legged_lab/legged_lab/tasks/locomotion/amp/config/lens110/__init__.py` 中注册。`MJCF-UpperLower` 是直接在 upper/lower plant 上训练，不能拿来替换 pitch/roll plant 的 checkpoint。

## 5. DWAQ 行走与楼梯

DWAQ actor 是 blind proprioceptive policy：actor 使用 velocity command、相对关节位置/速度、上一动作和
历史；不直接接收 base linear velocity。critic 使用 privileged observations，VAE 使用历史观测推断速度/latent。

```text
policy history -> ActorCriticDWAQ(actor + beta-VAE context)
               -> 21-DOF action
critic privileged obs -> value
velocity target -> VAE supervision
```

当前配置为 `sim.dt=0.002`、`decimation=5`，策略 100 Hz，`episode_length_s=60.0`。平地任务使用
`LeggedLab-Isaac--DWAQ-Lens110-v0`；楼梯任务使用本次新增的
`LeggedLab-Isaac--DWAQ-Lens110-Stairs-v0`，仅地形/课程入口不同，不能将楼梯的 checkpoint 宣称为平地收敛。

## 6. 跌倒起身 AMP

起身 actor 使用 4 个历史帧：每帧 72 个特征，输入 `288`，输出 `21` 个关节目标。

```text
IMU/root + joint state history (4 x 72 = 288)
  -> AMP actor
  -> 21 joint-position action
  -> q_des = default_joint_pos + action_scale * action
```

起身物理频率为 500 Hz，policy 为 100 Hz。`joint_names_mjcf`、`default_joint_pos_rad`、`action_scale`、PD 和
effort 必须以 `projects/04_fall_to_stand/exports/versions/*/config/deploy_config.yaml` 为部署真值；不要把
DeepMimic 159-D 或 DWAQ action vector 直接送给 GetUp。

## 7. 入口自检

```bash
# 查看新 DWAQ 平地/楼梯注册代码
rg -n 'DWAQ-Lens110-(Stairs|PLAY)|DWAQ-Lens110-v0' \
  frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/tasks/locomotion/dwaq/config/lens110

# 查看全身/侧滚注册代码
rg -n 'Deepmimic-Lens110|SideRoll' \
  frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/lens110
```
