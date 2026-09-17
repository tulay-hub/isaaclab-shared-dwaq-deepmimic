# Sim2Sim Commands And Files

默认从仓库根目录运行：

```bash
cd frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot
```

如果当前 shell 没有进入对应 Python 环境，把下面命令里的 `python` 换成你的环境命令，例如：

```bash
python
```

## 1. 老版通用 Atom01 Sim2Sim

用途：早期/通用 Atom01 部署脚本，23 action，配置直接写在脚本内部。

运行：

```bash
python scripts/locomotion_sim_to_sim.py
python scripts/locomotion_sim_to_sim.py --headless
```

调用/使用文件：

```text
入口:
  scripts/locomotion_sim_to_sim.py

默认 policy:
  logs/rsl_rl/atom01_amp/2025-12-26_16-46-08/exported/policy.pt

默认 MuJoCo XML:
  source/legged_lab/legged_lab/data/Robots/atom01/mjcf/atom01.xml

输出:
  simulation.mp4                 # --headless 时
  joint_positions.png
  base_velocities.png
```

## 2. Lens110 PR 主版本

用途：Lens110 pitch/roll 踝关节 policy 的主 sim2sim，默认使用 `torchscript`。

运行：

```bash
python scripts/sim2sim_pr/lens110_pr.py --cmd_vel 0.8 0.0 0.0 --sim_duration 40
python scripts/sim2sim_pr/lens110_pr.py --headless --cmd_vel 0.8 0.0 0.0 --sim_duration 10
```

切 ONNX：

```bash
python scripts/sim2sim_pr/lens110_pr.py --policy_backend onnx --cmd_vel 0.8 0.0 0.0
```

调用/使用文件：

```text
入口:
  scripts/sim2sim_pr/lens110_pr.py

配置:
  scripts/sim2sim_pr/pr.json

policy:
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.onnx

MuJoCo XML:
  source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml

配置引用的训练参数目录:
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/params
```

注意：当前本地该 `params` 目录下没有 `env.yaml/agent.yaml`。`lens110_pr.py` 已处理为默认回退到 `pr.json` 内置参数；如果手动传 `--params_dir`，该目录必须包含 `env.yaml` 和 `agent.yaml`。

常用调试参数：

```bash
python scripts/sim2sim_pr/lens110_pr.py --debug_actions --debug_contacts --cmd_vel 0.8 0.0 0.0
```

注意：`--hold_default` 不是该脚本参数；如果只想等非零命令再启动，用：

```bash
python scripts/sim2sim_pr/lens110_pr.py --wait_for_command
```

## 3. Lens110 PR 精简版

用途：`lens110_pr.py` 的 compact 版本，少一些参数同步和调试逻辑。

运行：

```bash
python scripts/sim2sim_pr/lens110_pr_simple.py --cmd_vel 0.8 0.0 0.0 --sim_duration 40
python scripts/sim2sim_pr/lens110_pr_simple.py --headless --output simulation_simple.mp4 --sim_duration 10
```

调用/使用文件：

```text
入口:
  scripts/sim2sim_pr/lens110_pr_simple.py

配置:
  scripts/sim2sim_pr/pr.json

policy:
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.onnx

MuJoCo XML:
  source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml

输出:
  simulation_simple.mp4
```

## 4. Lens110 PR Motor XML 版本

用途：仍是 PR policy 语义，但 MuJoCo plant 使用 motor actuator XML。

运行：

```bash
python scripts/sim2sim_pr/lens110_pr_motor.py --cmd_vel 0.8 0.0 0.0 --sim_duration 40
python scripts/sim2sim_pr/lens110_pr_motor.py --headless --output simulation_motor.mp4 --sim_duration 10
```

调用/使用文件：

```text
入口:
  scripts/sim2sim_pr/lens110_pr_motor.py

配置:
  scripts/sim2sim_pr/pr_motor.json

policy:
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.onnx

MuJoCo XML:
  source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof_motor.xml

配置引用的训练参数目录:
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/params

输出:
  simulation_motor.mp4
```

## 5. Lens110 PR + Real Upper/Lower Ankle Command

用途：PR policy 跑动时，额外输出真实 upper/lower 踝命令，主要用于桥接和诊断。

运行：

```bash
python scripts/sim2sim_pr/lens110_pr_real_ankle.py --cmd_vel 0.8 0.0 0.0 --sim_duration 40
python scripts/sim2sim_pr/lens110_pr_real_ankle.py --headless --output simulation_motor_real_ankle_cmd.mp4 --sim_duration 10
```

调用/使用文件：

```text
入口:
  scripts/sim2sim_pr/lens110_pr_real_ankle.py

配置:
  scripts/sim2sim_pr/pr_real_ankle.json

policy:
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.onnx

主 MuJoCo XML:
  source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof_motor.xml

真实踝映射 XML:
  source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_motor_sensor.xml

配置引用的训练参数目录:
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/params

输出:
  simulation_motor_real_ankle_cmd.mp4
```

可关闭 real ankle 打印：

```bash
python scripts/sim2sim_pr/lens110_pr_real_ankle.py --no_print_real_ankle
```

## 6. PR Policy 部署到 Upper/Lower 真实踝 Plant

用途：旧 PR policy + 新真实 upper/lower 踝结构；通过 `pos_pr2ul_left/right.pkl` 做 PR 到 UL 目标转换。

运行：

```bash
python scripts/sim2sim_pr_up/lens110_pr_up.py
python scripts/sim2sim_pr_up/lens110_pr_up.py --headless --output outputs/lens110_pr_up.mp4
```

调用/使用文件：

```text
入口:
  scripts/sim2sim_pr_up/lens110_pr_up.py

配置:
  scripts/sim2sim_pr_up/pr_up.json

policy:
  logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.onnx

MuJoCo XML:
  source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml

PR -> upper/lower 拟合模型:
  pitchRoll2UpperLower/ankle_model/models/pos_pr2ul_left.pkl
  pitchRoll2UpperLower/ankle_model/models/pos_pr2ul_right.pkl

输出:
  outputs/lens110_pr_up.mp4
```

常用调参：

```bash
python scripts/sim2sim_pr_up/lens110_pr_up.py --ankle_action_scale_mult 0.25
python scripts/sim2sim_pr_up/lens110_pr_up.py --ankle_command_rel_clip 0.3 --ankle_command_rate_limit 3.0
python scripts/sim2sim_pr_up/lens110_pr_up.py --hold_default
```

## 7. Lens110 Upper/Lower 主推荐入口

用途：真实 upper/lower 踝结构的主 sim2sim。当前默认参考 `sim2sim_pr_up` 的路线，使用 `upper_lower_action_mode=through_pitch_roll`：policy-facing action 是 upper/lower，但先转换回训练时的 pitch/roll target，再映射到真实 upper/lower 控制端。

运行：

```bash
python scripts/sim2sim_ul/lens110_ul.py --hold_default
python scripts/sim2sim_ul/lens110_ul.py --load_model <本地21DoF-upper-lower-policy.pt> --cmd_vel 0.8 0.0 0.0 --sim_duration 40
python scripts/sim2sim_ul/lens110_ul.py --load_model <本地21DoF-upper-lower-policy.pt> --headless --output lens110_ul.mp4 --sim_duration 10
```

注意：当前整理目录没有找到与 `policy.io=upper_lower` 匹配的专用策略文件，
`ul.json` 因此不设置虚假的默认 policy。运行 policy 时必须显式提供本地
`--load_model <本地21DoF-upper-lower-policy.pt>` 或 `--load_onnx <本地21DoF-upper-lower-policy.onnx>`。

调用/使用文件：

```text
入口:
  scripts/sim2sim_ul/lens110_ul.py

核心实现:
  scripts/sim2sim_ul/ul_full.py

配置:
  scripts/sim2sim_ul/ul.json

policy:
  当前未找到本地专用 upper/lower policy；必须通过命令行显式传入本地文件

训练参数:
  当前未找到本地 upper/lower params；使用 `ul.json` 内置参数

MuJoCo XML:
  source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml

踝关节转换模型目录:
  pitchRoll2UpperLower/ankle_model/models/
    ankle_poly_models.npz
    pos_pr2ul.pkl
    pos_pr2ul_left.pkl
    pos_pr2ul_right.pkl
    pos_ul2pr.pkl
    pos_ul2pr_left.pkl
    pos_ul2pr_right.pkl
    vel_pr2ul.pkl
    vel_ul2pr.pkl

输出:
  lens110_ul.mp4
```

常用变体：

```bash
# 保持默认站姿，不跑 policy，用于查 plant/初始姿态
python scripts/sim2sim_ul/lens110_ul.py --hold_default

# 手臂 action 置零
python scripts/sim2sim_ul/lens110_ul.py --arm_action_scale 0

# 踝 action 缩小
python scripts/sim2sim_ul/lens110_ul.py --ankle_action_scale 0.5

# 走训练时 through_pitch_roll 路线，当前 ul.json 已默认如此
python scripts/sim2sim_ul/lens110_ul.py --training_action

# 强制 direct upper/lower 路线；只建议用于真正 MJCF direct upper/lower policy
python scripts/sim2sim_ul/lens110_ul.py --direct_ul

# 冻结 base，用于隔离关节/踝控制问题
python scripts/sim2sim_ul/lens110_ul.py --freeze_base
```

## 8. `ul_full.py` 完整/高级入口

用途：`lens110_ul.py` 是瘦入口；`ul_full.py` 是完整高级入口，支持更多 policy IO、target clip、tendon、ankle control 等参数。

运行：

```bash
python scripts/sim2sim_ul/ul_full.py --config scripts/sim2sim_ul/ul.json --hold_default
python scripts/sim2sim_ul/ul_full.py --config scripts/sim2sim_ul/ul.json --load_model <本地21DoF-upper-lower-policy.pt> --headless --output simulation_motor_upper_lower.mp4
```

调用/使用文件：

```text
入口:
  scripts/sim2sim_ul/ul_full.py

默认配置:
  scripts/sim2sim_ul/ul.json

备用配置：无。当前目录只保留可核对的 `scripts/sim2sim_ul/ul.json`。

`ul.json` 的 `policy.torchscript`、`policy.onnx` 和 `source.params_dir` 均为
`null`，因为本地没有找到对应的专用 upper/lower policy；运行 policy 时必须
显式传入本地文件：

```bash
python scripts/sim2sim_ul/ul_full.py \
  --config scripts/sim2sim_ul/ul.json \
  --load_model <本地21DoF-upper-lower-policy.pt>
```

本地已确认的 plant 和踝转换依赖：

```text
MuJoCo XML:
  source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml

踝关节转换模型目录:
  pitchRoll2UpperLower/ankle_model/models/
  pitchRoll2UpperLower/t_p_v/models/weighted_mlp/
```

## 快速选择

日常验证真实 upper/lower 踝：

```bash
python scripts/sim2sim_ul/lens110_ul.py --cmd_vel 0.8 0.0 0.0
```

复现旧 PR policy：

```bash
python scripts/sim2sim_pr/lens110_pr.py --cmd_vel 0.8 0.0 0.0
```

检查 PR policy 到真实 upper/lower 踝映射：

```bash
python scripts/sim2sim_pr_up/lens110_pr_up.py
```

隔离 plant/默认姿态问题：

```bash
python scripts/sim2sim_ul/lens110_ul.py --hold_default
```
