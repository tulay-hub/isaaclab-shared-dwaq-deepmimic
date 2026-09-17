# lens110_pr_xml_tendon.py 代码说明

文件路径：

```text
projects/02_dance_half_body/framework/legged_lab_upper_lower/scripts/sim2sim_pr_up/lens110_pr_xml_tendon.py
```

相关 XML tendon 求解器：

```text
projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py
```

## 1. 这份 sim2sim 在做什么

这份脚本的目标是：

```text
lens110_amp 策略仍然按照训练时的 pitch/roll 脚踝语义工作，
但是 MuJoCo 里实际控制真实 upper/lower 脚踝电机。
```

也就是：

```text
policy 输入 obs: pitch/roll 脚踝观测
policy 输出 action: pitch/roll 脚踝目标
        ↓
用 XML spatial tendon 几何把 pitch/roll 目标解成 upper/lower 目标
        ↓
对 upper/lower actuator 做 torque PD 控制
```

所以这份代码不是纯 PR 控制，也不是 UL policy。

更准确地说：

```text
策略 IO = PR
物理 plant = UL
PR -> UL = XML tendon 几何求解
最终 data.ctrl = upper/lower 电机力矩
```

## 2. 关键关节

策略语义里的 pitch/roll 脚踝关节：

```python
ANKLE_PR = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
```

真实被控制的 upper/lower 电机关节：

```python
UPPER_LOWER = (
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
)
```

脚踝 PR 关节用于：

- 给 policy 提供观测。
- 表达 policy 输出的目标语义。
- 在 MuJoCo 里作为被动机构状态参与运动。

脚踝 UL 关节用于：

- 作为真实 actuator 控制对象。
- 接收 PR->UL 求解后的目标角。
- 最终通过 PD 算 torque 写入 `data.ctrl`。

## 3. 配置文件

默认配置：

```text
scripts/sim2sim_pr_up/pr_up.json
```

里面最重要的是三个 joint list。

### 3.1 mujoco_joint_names

这个是 policy 观测空间对应的 MuJoCo 关节列表。

里面脚踝是 PR：

```text
left_ankle_pitch_joint
left_ankle_roll_joint
right_ankle_pitch_joint
right_ankle_roll_joint
```

它对应训练时的 `lens110_amp` 策略输入/输出语义。

### 3.2 control_joint_names

这个是真实控制空间。

非脚踝关节基本和 `mujoco_joint_names` 一样，但是脚踝换成了 UL：

```text
left_ankle_upper_joint
left_ankle_lower_joint
right_ankle_upper_joint
right_ankle_lower_joint
```

所以最终 PD 控制的是这四个脚踝电机。

### 3.3 policy_joint_names

这个是 policy action/obs 中的关节顺序。

代码会建立映射：

```python
policy_to_mujoco = make_mapping(cfg["robot"]["policy_joint_names"], cfg["robot"]["mujoco_joint_names"])
```

这样 policy 输出顺序和 MuJoCo joint 顺序可以不一样。

## 4. 启动流程

入口是：

```python
def main() -> None:
```

主要流程如下：

```python
args = parse_args()
cfg = load_config(cfg_path)
policy_path = resolve_path(args.policy or cfg["policy"]["onnx"])
xml_path = resolve_path(args.xml or cfg["model"]["path"])
```

默认 policy 是：

```text
logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.onnx
```

默认 XML 是配置里的：

```text
source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml
```

也可以用参数换成 AMP_mjlab 那份：

```bash
--xml frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml
```

## 5. MuJoCo model 加载

代码用：

```python
model = Lens110XmlTendonSolver._load_model(xml_path)
model.opt.timestep = float(cfg["simulation"]["dt"])
data = mujoco.MjData(model)
```

这里不用裸的：

```python
mujoco.MjModel.from_xml_path(...)
```

原因是 AMP_mjlab 那份 XML 里：

```xml
<compiler meshdir="./meshes/"/>
```

但实际 meshes 在：

```text
../meshes
```

`Lens110XmlTendonSolver._load_model()` 内部会自动修正这种 meshdir 路径问题。

## 6. Joints 类

`Joints` 类负责把 joint name 转成 MuJoCo 里的索引。

核心内容：

```python
self.joint_ids = ...
self.qpos_ids = model.jnt_qposadr[self.joint_ids]
self.qvel_ids = model.jnt_dofadr[self.joint_ids]
self.actuator_ids = self._find_actuators(model)
```

它提供两个读取函数：

```python
def qpos(self, data):
    return data.qpos[self.qpos_ids].copy()

def qvel(self, data):
    return data.qvel[self.qvel_ids].copy()
```

脚本创建两套 joints：

```python
observe_joints = Joints(model, cfg["robot"]["mujoco_joint_names"], require_actuators=False)
control_joints = Joints(model, cfg["robot"]["control_joint_names"], require_actuators=True)
```

含义是：

```text
observe_joints: 用 PR 脚踝，给 policy 做观测。
control_joints: 用 UL 脚踝，最后用于 actuator torque 控制。
```

## 7. contact 和 actuator patch

函数：

```python
patch_contacts_and_actuators(model, control_joints, cfg)
```

做几件事：

1. 把脚底碰撞几何改成训练时类似的 box。
2. 只打开地面和脚底相关 contact。
3. 设置地面摩擦。
4. 设置 actuator 为 torque motor。
5. 设置 actuator force range。

其中 actuator force range 来自：

```json
"tau_limit": [...]
```

也就是说这份 sim 里不是 MuJoCo 自带 position actuator，而是显式算 torque 后写 `data.ctrl`。

## 8. 默认姿态处理

训练默认姿态在配置里：

```json
"default_pos": [...]
"stand_default_pos": [...]
```

读取后会经过：

```python
default_pr = training_defaults_to_mujoco(default_pr_training, cfg["robot"]["mujoco_joint_names"])
stand_pr = training_defaults_to_mujoco(stand_pr_training, cfg["robot"]["mujoco_joint_names"])
```

目前主要处理右肘轴方向：

```python
TRAINING_TO_MUJOCO_SIGN = {
    "right_elbow_joint": -1.0,
}
```

然后根据 PR 默认姿态求出对应的 UL 默认姿态：

```python
default_ankle_pr = np.array([default_pr[observe_index[name]] for name in ANKLE_PR])
mapper = XmlTendonPrToUpperLower(xml_path, default_ankle_pr)
```

## 9. PR -> UL 映射器

类：

```python
class XmlTendonPrToUpperLower:
```

它替代了旧版 `lens110_pr_up.py` 里的 pkl 模型。

初始化：

```python
self.solver = Lens110XmlTendonSolver(xml_path)
self.last_ul = None
self.default_ul = self(default_pr)
self.last_ul = self.default_ul.copy()
```

调用时：

```python
def __call__(self, pr):
    ul = self.solver.solve_pr_to_ul(
        pr[0], pr[1], pr[2], pr[3],
        initial=self.last_ul,
    )
    self.last_ul = ul.copy()
    return ul
```

这里的 `initial=self.last_ul` 很重要。

因为四连杆/tendon 几何可能有多解，如果每次都全局找，可能跳到另一支解。用上一帧的 UL 作为初值，可以让解连续。

## 10. 为什么 target_with_default_offset 要这样写

代码：

```python
def target_with_default_offset(self, target_pr, default_control_ul):
    return default_control_ul + (self(target_pr) - self.default_ul)
```

它不是直接：

```python
return self(target_pr)
```

原因是：训练默认姿态和仿真站立默认姿态不一定完全一样。

这个式子的含义是：

```text
UL_target = 当前仿真的默认 UL + (目标 PR 对应 UL - 训练默认 PR 对应 UL)
```

也就是保留 policy 输出的“相对默认姿态变化量”，但落到当前物理站立默认 UL 上。

这样比裸映射更稳。

## 11. XML tendon 求解器怎么工作

文件：

```text
pitchRoll2UpperLower/solve/xml_tendon_pr_to_ul.py
```

核心类：

```python
class Lens110XmlTendonSolver:
```

初始化时读 XML：

```python
self.model = self._load_model(self.xml_path)
self.data = mujoco.MjData(self.model)
self.qpos_addr = ...
self.joint_range = ...
self.tendon_targets = np.asarray(self.model.tendon_range[:4, 0])
```

它从 XML 里读到四条 tendon 目标长度：

```text
left upper:  0.192
left lower:  0.132
right upper: 0.192
right lower: 0.132
```

对应 XML：

```xml
<spatial range="0.192 0.192001">
  <site site="lleg_Link4_1"/>
  <site site="lleg_Link6_a"/>
</spatial>

<spatial range="0.132 0.132001">
  <site site="lleg_Link4_2"/>
  <site site="lleg_Link6_b"/>
</spatial>
```

数学上就是：

```text
|site_a(q) - site_b(q)| = target_length
```

对左上来说：

```text
f(u; p, r) =
    |lleg_Link4_1(u) - lleg_Link6_a(p, r)| - 0.192
```

给定 `pitch=p`、`roll=r`，求一个 `upper=u` 让：

```text
f(u; p, r) = 0
```

## 12. tendon 误差函数

代码：

```python
def _channel_error(self, channel, pitch, roll, motor_angle):
    self.reset_qpos()
    self.data.qpos[self.qpos_addr[channel.pitch_joint]] = pitch
    self.data.qpos[self.qpos_addr[channel.roll_joint]] = roll
    self.data.qpos[self.qpos_addr[channel.solve_joint]] = motor_angle
    mujoco.mj_forward(self.model, self.data)
    return data.ten_length[channel.tendon_id] - tendon_target
```

关键是：

```python
mujoco.mj_forward(self.model, self.data)
```

MuJoCo 会根据 XML 的 body、joint、site、tendon 自动算：

```text
site 世界坐标
tendon 当前长度
```

所以代码没有手写三角公式，也没有用拟合模型。

它是直接用 XML 几何计算。

## 13. tendon 求根

函数：

```python
solve_channel(...)
```

它对单个通道求解：

```text
left_upper
left_lower
right_upper
right_lower
```

流程：

1. 取该 UL joint 的 XML 限位。
2. 如果有上一帧解，就先在上一帧附近找根。
3. 如果附近找不到，就在整个 joint range 扫描。
4. 找到误差变号区间后，用二分法求根。
5. 如果找不到满足 tendon 长度的角度，就报错。

局部搜索：

```python
for radius in (0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20):
    a = max(lo, center - radius)
    b = min(hi, center + radius)
    if error 变号:
        return bisection(...)
```

全局搜索：

```python
grid = np.linspace(lo, hi, self.grid_samples)
values = [error(x) for x in grid]
```

这保证：

```text
求解只在 XML joint range 内进行。
找不到真实几何解时会报错，不会强行 clip 伪造。
```

## 14. policy observation 怎么构造

函数：

```python
build_observation(...)
```

内容：

```python
quat = imu.quat(data)
rot = quat_to_rotmat_wxyz(quat)
omega = imu.gyro_body(data)
gravity = rot.T @ np.array([0.0, 0.0, -1.0])
q_obs = (observe_joints.qpos(data) - default_pos)[policy_to_mujoco] * policy_action_sign
dq_obs = observe_joints.qvel(data)[policy_to_mujoco] * policy_action_sign
obs = np.concatenate([omega, gravity, command, q_obs, dq_obs, last_action])
```

观测顺序：

```text
base angular velocity
projected gravity
command velocity
joint position error
joint velocity
last action
```

这跟训练里的 AMP policy 观测形式对齐。

注意：

```text
observe_joints 使用的是 mujoco_joint_names
其中脚踝是 PR
```

所以 policy 看到的是 PR，而不是 UL。

## 15. policy action 怎么变成 PR 目标

主循环里：

```python
raw_action = policy(obs).astype(np.float32)
```

然后：

```python
target_delta_policy = raw_action * action_scale_policy * policy_action_sign
target_pr = default_pr.copy()
target_pr[policy_to_mujoco] += target_delta_policy
np.clip(target_pr, observe_lower, observe_upper, out=target_pr)
```

含义：

```text
target_pr = default_pr + action * action_scale
```

这里得到的仍然是 PR 目标。

## 16. PR 目标怎么变成 control_target

函数：

```python
target_pr_to_control(...)
```

非脚踝：

```python
control[control_index[name]] = target_pr_full[observe_index[name]]
```

脚踝：

```python
ankle_pr = [left_pitch, left_roll, right_pitch, right_roll]
ankle_ul = mapper.target_with_default_offset(ankle_pr, default_ul)
control[upper_lower_ids] = ankle_ul
```

所以 control target 里：

```text
hip/knee/torso/arm: 普通目标角
ankle: upper/lower 目标角
```

## 17. ankle_action_scale_mult 是什么

配置里有：

```json
"pr_to_ul_action_scale_mult": 0.25
```

代码里：

```python
if default_pr_full is not None and ankle_pr_scale != 1.0:
    default_ankle_pr = ...
    ankle_pr = default_ankle_pr + (ankle_pr - default_ankle_pr) * ankle_pr_scale
```

含义是只缩放脚踝 PR 目标的相对动作：

```text
ankle_pr_for_mapping = default_ankle_pr + (target_ankle_pr - default_ankle_pr) * scale
```

默认是 `0.25`，说明先把策略脚踝动作缩小到 25%，再做 PR->UL。

这是为了防止真实 UL plant 上脚踝动作过猛。

如果你想放大脚踝动作：

```bash
--ankle_action_scale_mult 0.5
```

如果你想完全不缩：

```bash
--ankle_action_scale_mult 1.0
```

## 18. ankle_pr_feedback 是什么

代码里有：

```python
ankle_pr_map, ankle_pr_err = compensated_ankle_pr_for_mapping(...)
```

它可以根据当前实际 PR 和目标 PR 的误差，修正用于映射的 PR。

公式近似是：

```text
mapped_pr = target_pr + gain * (target_pr - actual_pr)
```

默认配置里：

```json
"ankle_pr_feedback": {
  "gain": 0.0,
  "clip": 0.25
}
```

所以默认不启用反馈补偿。

如果你想让 UL 目标更激进地追 PR，可以试：

```bash
--ankle_pr_track_gain 0.2
```

但这个可能会让系统更容易震。

## 19. upper/lower command safety

函数：

```python
limit_upper_lower_command(...)
```

它支持两个限制：

```text
rel_clip: 限制 UL target 相对默认 UL 的最大偏移
rate_limit: 限制 UL target 每秒变化率
```

配置里默认：

```json
"ankle_command_safety": {
  "rel_clip": null,
  "rate_limit": null
}
```

也就是默认不限制。

运行时可以加：

```bash
--ankle_command_rel_clip 0.4
--ankle_command_rate_limit 4.0
```

这样可以防止 UL target 突然跳很大。

## 20. 最终 torque 怎么算

函数：

```python
pd_torque(...)
```

公式：

```python
tau = kp * (target - q) - kd * dq
tau = np.clip(tau, -tau_limit, tau_limit)
```

主循环每个 MuJoCo step 都执行：

```python
torque = pd_torque(control_joints.qpos(data), control_joints.qvel(data), control_target, cfg)
data.ctrl[control_joints.actuator_ids] = torque
mujoco.mj_step(model, data)
```

注意：

```text
control_joints 是 control_joint_names
脚踝在 control_joint_names 中是 upper/lower
```

所以实际写入 `data.ctrl` 的脚踝 torque 是：

```text
left_ankle_upper_joint actuator torque
left_ankle_lower_joint actuator torque
right_ankle_upper_joint actuator torque
right_ankle_lower_joint actuator torque
```

不是 PR torque。

## 21. 主循环完整数据流

每个控制周期：

```text
1. 读取 MuJoCo 状态
2. 从 PR 关节和 IMU 构造 policy obs
3. policy 输出 21 维 action
4. action 变成 target_pr
5. target_pr 的脚踝部分送进 XML tendon solver
6. solver 输出 target_ul
7. 组合 control_target：
   - 非脚踝 = target_pr 对应普通关节
   - 脚踝 = target_ul
8. 每个 MuJoCo step 用 PD 算 torque
9. data.ctrl 写入真实 actuator
10. mj_step
```

流程图：

```text
MuJoCo state
  ↓
PR qpos/qvel + IMU
  ↓
policy obs
  ↓
lens110_amp policy
  ↓
PR action
  ↓
target_pr = default_pr + action * scale
  ↓
XML tendon solve
  ↓
target_ul
  ↓
PD torque on control_joints
  ↓
UL actuator ctrl
  ↓
MuJoCo step
```

## 22. 打印日志怎么看

运行时会打印类似：

```text
ankle_pr_tgt=(...)
ankle_pr_act=(...)
ankle_pr_map=(...)
ankle_pr_err=(...)
ankle_ul_rel=(...)
```

含义：

```text
ankle_pr_tgt: policy 输出转换后的目标 PR
ankle_pr_act: MuJoCo 当前实际 PR
ankle_pr_map: 送进 XML tendon solver 的 PR
ankle_pr_err: 目标 PR 和实际 PR 的误差
ankle_ul_rel: 最终 UL target 相对站立默认 UL 的偏移
```

如果 `ankle_pr_tgt` 很大，`ankle_ul_rel` 也很大，可能脚踝动作过激。

可以先尝试：

```bash
--ankle_action_scale_mult 0.1
--ankle_command_rel_clip 0.3
--ankle_command_rate_limit 3.0
```

## 23. 常用运行命令

默认运行，有 viewer：

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower
python scripts/sim2sim_pr_up/lens110_pr_xml_tendon.py
```

指定 AMP_mjlab XML：

```bash
python scripts/sim2sim_pr_up/lens110_pr_xml_tendon.py \
  --xml frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml
```

无渲染快速跑日志：

```bash
python scripts/sim2sim_pr_up/lens110_pr_xml_tendon.py \
  --no_render --sim_duration 5
```

先站住不跑 policy：

```bash
python scripts/sim2sim_pr_up/lens110_pr_xml_tendon.py \
  --no_render --hold_default --sim_duration 5
```

降低脚踝动作：

```bash
python scripts/sim2sim_pr_up/lens110_pr_xml_tendon.py \
  --ankle_action_scale_mult 0.1 \
  --ankle_command_rel_clip 0.3 \
  --ankle_command_rate_limit 3.0
```

## 24. 和旧 lens110_pr_up.py 的区别

旧脚本：

```text
PR policy -> pkl 模型 -> UL target -> UL motor PD
```

新脚本：

```text
PR policy -> XML tendon 几何求解 -> UL target -> UL motor PD
```

旧脚本依赖：

```text
pos_pr2ul_left.pkl
pos_pr2ul_right.pkl
```

新脚本不依赖这些模型。

新脚本使用：

```python
from xml_tendon_pr_to_ul import Lens110XmlTendonSolver
```

## 25. 这份代码的本质

它不是重新训练策略。

它是在 sim2sim 里做一个控制接口转换：

```text
训练策略认为自己控制 pitch/roll 脚踝。
真实 XML 机器人有 upper/lower 脚踝电机。
所以每一步把 pitch/roll 目标解成 upper/lower 目标。
```

因此最终真实控制对象是：

```text
upper/lower physical motors
```

而策略输入输出仍然是：

```text
pitch/roll policy semantics
```

