# Lens110 Isaac Lab → MuJoCo Sim2Sim 对齐与诊断报告

日期: 2026-08-19 | 训练策略: Lens110 21-DOF 官方 dance (13 维动作)

## 0. Baseline 备份

`sim2sim_baseline/` 保存了修改前的稳定版本:

- `mujoco_sim2sim_official.py` (sha256 `5c0d77e9...`)
- `mujoco_sim2sim.py`
- `test_zero_residual_mujoco.py`
- `mujoco_play_npz_full.py`
- `lens110_21dof_sim_flatfoot.xml`

项目非 git 仓库, 以文件哈希和时间戳记录版本。

## 1. 本体参数对比 (USD vs MuJoCo XML)

### 关节

21 个 joint, 名称/轴/顺序 USD 与 MuJoCo **完全一致** (hip p/r/y, knee,
ankle p/r, torso_yaw, shoulder p/r/y, elbow; 轴 Y/X/Z 逐项相同)。
范围除 knee 外全部一致。

| 差异项 | USD | MuJoCo (修改前) | 处理 |
|---|---|---|---|
| knee 限位 | [-0.087, 2.443] | [-0.087, **0.9**] | **已修正为 2.443** (2026-08-19) |
| 多数 body 惯性主轴 xx/yy | URDF/USD 顺序 | 主轴分解后 xx/yy 交换 | 已记录, 见 §9 |

### 质量 (全部一致, 含 pelvis)

所有 21 body 质量 USD == MuJoCo (差异 <1e-5)。pelvis 已统一为 **2.3981 kg**
(2026-08-18 从 legged_lab 训练简化值 2.1315 改回实机/Isaac 值)。
单因素实验确认 pelvis 2.3981 vs 2.1315 对零残差结果无影响。

### 碰撞体

脚底 box (pos 0.0353/-0.001/-0.015, size 0.0975/0.0455/0.0175) 与
legged_lab 成功训练包/ThreeAction 播放器一致。Isaac USD 使用 STL convex hull;
MuJoCo 用 box (单因素: box 换 mesh 凸包更差, 维持 box)。

## 2. Joint Mapping (Isaac 0..20 ↔ MuJoCo 0..20)

完全一一对应, 无 sign inversion:

| # | Isaac USD | MuJoCo | 轴 | 范围 (rad) |
|---|---|---|---|---|
| 0 | left_hip_pitch | 同名 | Y | [-2.181, 2.181] |
| 1 | right_hip_pitch | 同名 | Y | [-2.181, 2.181] |
| 2 | torso_yaw | 同名 | Z | [-1.832, 1.832] |
| 3 | left_hip_roll | 同名 | X | [-0.523, 1.570] |
| 4 | right_hip_roll | 同名 | X | [-1.570, 0.523] |
| 5-10 | shoulder/elbow 左右 | 同名 | Y/X/Z | 一致 |
| 11 | left_knee | 同名 | Y | [-0.087, 2.443] (已对齐) |
| 12 | right_knee | 同名 | Y | 同上 |
| 13-20 | 手臂 8 关节 | 同名 | 一致 | 一致 |

动作映射: 训练动作 CSV 顺序 ↔ USD 顺序的 `usd_to_csv`/`csv_to_usd`
此前与 Isaac dump 逐维验证误差 5e-7。

## 3. 观测: 126 维 (确认)

从 checkpoint 实测: `actor_mlp.0.weight` shape = **(256, 126)**;
`mu.weight` = (13, 64)。MLP = [256, 128, 64]。

```
126 = 45 (state) + 81 (future)
state:
  [0:13]  12腿 q + 腰 q          (CSV 顺序)
  [13:26] 12腿 dq + 腰 dq
  [26:39] 12腿 prev_action + 腰 prev_action
  [39:42] base 角速度 (body 系, root_ang_vel_b)
  [42:45] gravity_b (world gravity -> body, 含 roll/pitch)
future:
  3 帧 (k, k+10, k+20) x (21 关节 CSV + 6 root euler+pos) = 81
```

MuJoCo headless 测试断言 `obs.shape == (126,)`, 已通过。

## 4. Observation Normalization

- 训练: `normalize_input=True`, RL-Games running_mean_std。
- checkpoint 内含 `model.running_mean_std.{running_mean,running_var,count}`。
- ONNX 导出 (export_onnx.py) 已把归一化**烘焙进网络** (第一层做
  `(obs-mean)/sqrt(var+eps)`), 因此 MuJoCo 侧使用 ONNX 时**不需要外部归一化**。
- play.py (PT 播放) 由 RL-Games BasePlayer 恢复 running_mean_std。

## 5. Action 接口

训练 (OfficialDanceJointAction, actions.py) 实际语义:

```
clipped_action = clip(policy_output, -1, 1)   # env.clip_actions=1.0
q_des[腿 12] = default_joint_pos + 0.25 * clipped_action   # default=参考帧0腿角!
q_des[腰 1]  = 0.25 * clipped_action
q_des[臂 8]  = motion reference (开环)
```

注意: **不是** `motion_reference + scale*action`。腿的 baseline 是
**default (帧0腿角)**, 不是逐帧 motion reference。MuJoCo 播放器已按此实现。
13→21 映射: 前 12 CSV 腿 + 第 13 腰, 臂 8 开环参考。无重复 offset/scale。

## 6. 控制时间轴

```
Isaac:   physics 500Hz (dt=0.002), decimation=5, policy 100Hz, action hold=5
MuJoCo:  physics 1000Hz (dt=0.001, sim_flatfoot), decimation=10, policy 100Hz, action hold=10
Motion:  fps=100, 整数帧推进 (time_steps += 1)
```

两边策略更新时刻与 motion reference 更新时刻一致 (策略步末推进一帧)。
训练本身用整数帧 (非插值), MuJoCo 播放器保持一致; 插值版可作为可选改进,
但与训练观测不一致, 默认不启用。

## 7. PD / Actuator (从代码读取)

训练 (assets/lens110.py, ImplicitActuatorCfg):

| 组 | stiffness | damping | effort |
|---|---|---|---|
| legs (髋/膝) | 40 | 5 | 80 (髋p/r+膝), 36 (髋yaw/腰) |
| feet (踝 p/r) | 20 | 20 | 36 |
| arms | 100 | 5 | 36 |
| waist (在 legs) | 100 | 5 | 36 |

MuJoCo 播放器: `motor + gain FIXED + bias AFFINE`
(gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd), 与原生 position 等价
(力公式已验证)。力矩限位一致。

单关节响应 (MuJoCo, 左膝阶跃 +0.1 rad, 悬浮, kp=40/kd=5):

```
time   q      dq      torque
0.001  0.000  0.128   4.000
0.101  0.055  0.377  -0.072
0.500  0.098  0.014   0.003
```

0.5s 收敛, 无振荡, 稳态误差 2%。Isaac 侧单关节响应待训练结束后补测。

## 8. 接触 Baseline (保持稳定, 未改)

```
friction = 0.7 (floor), solref = [0.02, 1], condim = 3,
armature = 0.01, damping = 0.01, frictionloss = 0.01
```

已知: condim=4 + solref=0.001 → root 飞 10m + segfault (已回退);
armature=0 → 关节速度爆炸。不再强行对齐 Isaac solver。

## 9. 实验结果 (MuJoCo, 2026-08-19)

### Test A: policy=0, frame0 固定 (zero residual)

```
0.5s root_z=0.6459  1.0s root_z=0.6455  2.0s root_z=0.5852  5.0s 倒
```

### Test B: policy=0, 完整 motion

与 A 几乎一致 (前段为静止站姿): 2s 0.5855, 5s 倒。

### Test C: policy ON, frame0 固定

**15s 不倒**, root_z 0.645→0.613。

### Test D: policy ON, 完整 motion (30000 轮弱 PD checkpoint)

**36s 完整舞蹈不倒**, root_z 0.644→0.533, max joint vel 1.84 rad/s。

### Pitch 平衡扫描 (policy=0, frame0, 每姿态 5s)

| pitch | 2s root_z | 结果 |
|---|---|---|
| -8°~-5° | <0.10 | 2s 内前倾倒 |
| -4° | 0.585 | 5s 内倒 |
| -3° | 0.644 | 5s 内倒 (最稳) |
| -2° | 0.640 | 5s 内倒 |
| -1° | 0.617 | 5s 内倒 |
| 0° | 0.552 | 5s 内倒 |
| +1°~+2° | <0.35 | 2s 内后仰倒 |

2 秒稳定区间约 **-4°~0°**, 5 秒无一存活。Isaac 侧扫描待训练后补。

### 单因素 (零残差, 均已跑)

| 变量 | 结果 |
|---|---|
| dt 500/1000Hz | 相同 |
| armature 0 | 速度爆炸 (57-80 rad/s) |
| armature 0.01 | 稳定 (保持) |
| solref 0.02/0.01/0.005 | 相同 |
| condim 3/4 | 相同 |
| friction 0.5/0.7/1.0 | 完全相同 |
| 骨盆惯量 2.3981/2.1315 | 相同 |
| 腰 0/ref0/refmax | 相同 |
| 踝 Kp 20/30/100 | 改善前 1s, 仍倒 |
| 关节阻尼 5 | 轻微改善, 仍倒 |

## 10. 发现的关键 Bug (本次修复)

**MuJoCo 播放器观测角速度坐标系错误** (mujoco_sim2sim_official.py):

```python
# 旧 (错误): body 系角速度又被旋转成 world 系
base_ang_vel = quat_rotate_wxyz(quat_inv_wxyz(robot_q_wxyz), data.qvel[3:6])
# 新 (修复): MuJoCo qvel[3:6] 即 body 系, 与训练 root_ang_vel_b 一致
base_ang_vel = data.qvel[3:6].copy()
```

这是"播放乱动/站不住"的主要原因: 策略观测的 [39:42] 与训练不一致。
修复后 30000 轮 checkpoint 在 MuJoCo 完整跳 36s 不倒。

### 10b. 播放器 geom margin=0.04 导致 policy ON 失稳 (2026-08-20)

播放器此前"对齐 Isaac contact_offset=0.04"把地板/脚底 geom margin 设为 0.04。
headless 对照实验 (43000 轮策略):

| 配置 | 首次倒地 |
|---|---|
| XML 默认 (margin 0.001, friction 1.8, 自碰撞开) | 36s 不倒 |
| friction 0.7 单独 | 36s 不倒 |
| 关闭 mesh 自碰撞 单独 | 36s 不倒 |
| **margin 0.04 单独** | **1.25s 倒** |

margin 增大使脚底离地数厘米即进入接触检测, 配合 MuJoCo 求解产生数值扰动,
策略直接失稳。已移除 margin 修改, 播放器与 headless 行为一致。

同时修复: 播放器策略推理频率 bug (policy_step % DECIMATION 导致 10Hz 推理,
应为每策略步 100Hz); 播放器 default 关节角硬编码旧帧0 (应取 motion 帧0)。

## 11. 其他已知差异 (待处理, 不影响当前结论)

- **惯性主轴 xx/yy 交换**: MuJoCo XML 多数 body 的 diaginertia 与
  URDF/USD 相比 xx/yy 交换 (如 ankle_pitch: MJC 5.5e-05/3.2e-05 vs
  USD 3.2e-05/5.5e-05)。mass 全部一致。修正需重算主轴分解, 对当前
  稳定性结论无影响, 留作后续严格对齐项。
- **播放器 physics 1000Hz vs Isaac 500Hz**: 单因素无差异, 保留 1000Hz
  (sim_flatfoot 默认, 成功包验证)。

## 12. 最终归类

主要问题: **E. observation mismatch (播放器角速度坐标系)**, 叠加
**J. 训练早期策略未收敛时的保命行为** (已被后续训练缓解)。

排除项 (有实验数据): B (归一化已在 ONNX 烘焙), C (action 一致),
D (mapping 一致), F (单关节响应合理), G (时间轴一致),
H (friction/solref/condim 单因素无差异), I (box 与训练包一致)。
A (126 维确认), 但播放器实现曾有坐标系错误 → 归为 E。

## 13. 推荐 MuJoCo 参数 (当前)

```
physics dt = 0.001 (1000Hz), decimation = 10, policy = 100Hz
PD: 髋/膝 40/5, 踝 20/20, 腰 100/5, 臂 100/5
effort: 髋p/r+膝 80, 其余 36
floor friction = [0.7, 0.005, 0.0001]
solref = [0.02, 1], condim = 3
armature = 0.01, damping = 0.01, frictionloss = 0.01
knee limit = [-0.087, 2.443] (已对齐)
pelvis = 2.3981 kg (已对齐)
```

## 14. 修改文件清单 (2026-08-18/19)

1. `assets/mjcf/lens110_21dof_sim_flatfoot.xml`: pelvis 2.3981; knee limit 2.443
2. `lens110_lab/scripts/bvh_tools/mujoco_sim2sim_official.py`: PD 20/20;
   分段 stand PD 开关 (默认关); **角速度 obs 修复**; **策略推理频率 10Hz→100Hz**;
   **default 取 motion 帧0**; **移除 geom margin=0.04** (均 2026-08-19/20)
3. `lens110_lab/scripts/bvh_tools/test_zero_residual_mujoco.py`: 单因素选项
   (pelvis/clearance/mesh_feet/pitch/joint_damping/friction)
4. `lens110_lab/scripts/bvh_tools/sim2sim_headless_test.py`: 新增 headless
   policy 测试 (Test C/D)
5. `lens110_lab/scripts/bvh_tools/make_start_transition.py`: 新增过渡 npz 生成
6. `lens110_lab/scripts/bvh_tools/monitor_training.py`: 训练监控
7. `lens110_lab/scripts/rl_games/play.py`: 播放禁用退火 + --pd_preset
8. `lens110_lab/scripts/rl_games/train.py`: --no_pd_anneal
9. `lens110_lab/source/.../lens110.py`: init_state 过渡起点; 踝 PD 20/20
10. `lens110_lab/source/.../lens110_env.py`: PD 退火课程 + 分段 stand PD (默认关)
11. `lens110_lab/source/.../lens110_lab_env_cfg.py`: 退火配置 + 奖励权重
    (leg 8→15, feet_still 1→3)
