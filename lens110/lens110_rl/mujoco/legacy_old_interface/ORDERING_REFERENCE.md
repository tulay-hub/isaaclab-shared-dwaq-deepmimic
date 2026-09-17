# Lens110 Dance Sim2Real 顺序对照表

所有顺序以训练/播放器实际使用为准, 供实机 SDK 逐维对照。

## 1. 关节顺序

### 1.1 USD 顺序 (21) —— 训练资产/URDF/npz joint_pos

```text
 0 left_hip_pitch_joint      11 left_knee_joint
 1 right_hip_pitch_joint     12 right_knee_joint
 2 torso_yaw_joint           13 left_shoulder_yaw_joint
 3 left_hip_roll_joint       14 right_shoulder_yaw_joint
 4 right_hip_roll_joint      15 left_ankle_pitch_joint
 5 left_shoulder_pitch_joint 16 right_ankle_pitch_joint
 6 right_shoulder_pitch_joint17 left_elbow_joint
 7 left_hip_yaw_joint        18 right_elbow_joint
 8 right_hip_yaw_joint       19 left_ankle_roll_joint
 9 left_shoulder_roll_joint  20 right_ankle_roll_joint
10 right_shoulder_roll_joint
```

### 1.2 CSV 顺序 (21) —— MuJoCo qpos / 播放器 / 观测 future

```text
 0 left_hip_pitch_joint      11 right_ankle_pitch_joint
 1 left_hip_roll_joint       12 torso_yaw_joint
 2 left_hip_yaw_joint        13 left_shoulder_pitch_joint
 3 left_knee_joint           14 left_shoulder_roll_joint
 4 left_ankle_pitch_joint    15 left_shoulder_yaw_joint
 5 left_ankle_roll_joint     16 left_elbow_joint
 6 right_hip_pitch_joint     17 right_shoulder_pitch_joint
 7 right_hip_roll_joint      18 right_shoulder_roll_joint
 8 right_hip_yaw_joint       19 right_shoulder_yaw_joint
 9 right_knee_joint          20 right_elbow_joint
10 right_ankle_pitch_joint
```

映射: `USD_TO_CSV[i] = CSV.index(USD[i])`, `CSV_TO_USD[j] = USD.index(CSV[j])`。

## 2. 动作 13 维顺序 (策略输出) + 逐关节 scale

顺序 = CSV 前 13 (12腿 + 腰), scale 为 2026-08-20 [0.25, 0.60] 版:

```text
 0 left_hip_pitch_joint    scale 0.42
 1 left_hip_roll_joint     scale 0.25
 2 left_hip_yaw_joint      scale 0.60
 3 left_knee_joint         scale 0.47
 4 left_ankle_pitch_joint  scale 0.30
 5 left_ankle_roll_joint   scale 0.25
 6 right_hip_pitch_joint   scale 0.38
 7 right_hip_roll_joint    scale 0.25
 8 right_hip_yaw_joint     scale 0.45
 9 right_knee_joint        scale 0.60
10 right_ankle_pitch_joint scale 0.36
11 right_ankle_roll_joint  scale 0.25
12 torso_yaw_joint         scale 0.52
```

执行语义:

```text
q_des[腿 i] = default_joint_pos[腿 i] + scale[i] * clip(action[i], -1, 1)
q_des[腰]   = scale[12] * clip(action[12], -1, 1)
q_des[臂 8] = motion reference (开环)
```

default_joint_pos = 动作第 0 帧关节角 (USD 顺序)。

## 3. 观测 126 维索引

### state (45) —— 与 `official_state_obs` 一致

```text
[0:13]    12腿 q + 腰 q                (CSV 前 13)
[13:26]   12腿 dq + 腰 dq              (CSV 前 13)
[26:39]   12腿 prev_action + 腰 prev   (原始 action, 未缩放)
[39:42]   base 角速度 (body 系)
[42:45]   gravity_b = R^T @ [0,0,-1]   (world->body)
```

### future (81) —— 与 `official_future_ref` 一致

```text
3 帧: k, k+10, k+20 (当前帧 + 0.1s/0.2s 前瞻)
每帧 27 维:
  [0:21]   21 关节 (CSV 顺序)
  [21:27]  root (roll, pitch, yaw, x, y, z)
合计 3 x 27 = 81
```

## 4. root 状态

- npz `body_pos_w` / `body_quat_w`: (N, 7, 3) / (N, 7, 4), 7 个 body 顺序:

```text
0 pelvis
1 left_ankle_roll_link
2 right_ankle_roll_link
3 left_elbow_link
4 right_elbow_link
5 left_shoulder_roll_link
6 right_shoulder_roll_link
```

- root = index 0; 四元数 wxyz; future 的 root 用 XYZ 欧拉 (roll, pitch, yaw)。

## 5. PD (播放端, MuJoCo)

USD 顺序 21:

```text
Kp: 髋 p/r/y = 100, 腰 = 100, 肩/肘 = 100,
     膝 = 40, 踝 p/r = 20
Kd: 髋 p/r/y = 8, 腰 = 5, 肩/肘 = 5,
     膝 = 5, 踝 p/r = 20
effort: 髋p/r + 膝 = 80, 其余 = 36
```

训练资产 (Isaac) 髋 = 40/5, 播放端加固为 100/8 (2026-08-20)。

## 6. 时间轴

```text
Isaac:  physics 500Hz, decimation 5, policy 100Hz, action hold 5
MuJoCo: physics 1000Hz, decimation 10, policy 100Hz, action hold 10
Motion: fps 100, 整数帧推进 (帧号 = 策略步)
```

## 7. 其他

- npz 字段: `fps, joint_pos(USD 21), joint_vel(USD 21), body_pos_w(N,7,3),
  body_quat_w(N,7,4), body_lin_vel_w(N,7,3), body_ang_vel_w(N,7,3)`
- 观测裁剪: clip_observations = 50; 动作裁剪: clip_action = 1.0
- 归一化: running_mean_std 已烘焙进 ONNX, 输入原始 126 维即可
