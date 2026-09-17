# Lens110 RL 项目（2026-08-28 整理版）

本目录是 Lens110 21-DOF 的唯一训练/仿真工程。旧 rl_games 框架已迁移归档，
当前主线为 `lens110_rl`（whole_body_tracking_engineai / rsl_rl 框架）。

## 目录结构

```text
frameworks/shared/lens110_isaaclab/lens110/
├── assets/        # 机器人本体资产: URDF / MJCF / USD / meshes (唯一来源)
├── configs/       # 机器人/PD/部署配置: robot_humanoid_lens110_config.yaml 等
├── docs/          # 观测/奖励规范、训练记录、维度说明
├── lens110_rl/    # 新训练框架 (rsl_rl):
│   ├── scripts/   #   train.py / play.py / 转换脚本
│   ├── source/    #   whole_body_tracking 扩展包 (pip 可编辑安装)
│   ├── data/      #   训练动作 npz (lens110_dance_100hz.npz)
│   ├── logs/      #   训练日志/checkpoint (rsl_rl/lens110_flat/<run>)
│   └── mujoco/    #   MuJoCo 播放器与 XML (legacy_old_interface 为旧接口版本)
├── logs/
│   └── archive/   # 旧 rl_games 训练日志 (仅存档, 可删除)
└── _archive_20260828/   # 旧框架代码/脚本/副本备份 (工作区根目录)
```

## 常用命令

训练（新框架）:

```bash
cd frameworks/shared/lens110_isaaclab/lens110/lens110_rl
python scripts/rsl_rl/train.py --task Tracking-Flat-Lens110-v0 \
  --num_envs 1024 --max_iterations 30000 --headless
```

续训:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-Lens110-v0 \
  --num_envs 1024 --max_iterations 30000 --headless \
  --resume true --load_run <run目录> --checkpoint <模型文件>
```

Isaac 播放:

```bash
python scripts/rsl_rl/play.py --task Tracking-Flat-Lens110-v0 \
  --load_run <run目录> --checkpoint <模型文件> --num_envs 1
```

## 关键配置

- 任务: `Tracking-Flat-Lens110-v0`
- 观测: H 版 161 维（root 姿态6 + 角速度3 + 关节位置21 + 速度21 + 前瞻24 + 前瞻关节84 + 触地2）
- 动作: 21 关节绝对位置, scale 0.25
- 奖励: T800 框架自带 9 项（见 `lens110_rl/source/.../tracking_env_cfg.py`）
- 物理: 500 Hz, 策略 100 Hz（decimation=5）
- PD: 髋/膝 120/4, 踝 55/2, 腰 45/1.5, 臂 35/1.2

## 归档说明

- 旧 rl_games 框架代码: `_archive_20260828/lens110_lab_old_framework_20260828.tar.gz`
- 旧训练日志: `logs/archive/rl_games`
- 旧 MuJoCo 播放器: `lens110_rl/mujoco/legacy_old_interface`（13 维残差旧接口, 新接口适配中）
- 上游/副本: `_archive_20260828/` 下 `source_whole_body_tracking_old` 与两份 example 副本

## English

This directory is the shared Isaac Lab base for the DWAQ, DeepMimic, and related humanoid tasks. It contains common task registration, terrain generation, MuJoCo models, reward components, replay tools, and training wrappers used by the standalone dance, walking, side-roll, and stairs projects.

The shared layer does not define one universal checkpoint. Project-specific policies and upper/lower ankle mappings must be supplied by the consuming project. Use repository-relative paths, keep policy dimensions and joint order in the interface contract, and record the exact framework commit with every export.

Typical reproduction is: install Isaac Lab and the local Python dependencies, run the consuming project wrapper, perform MuJoCo replay, and compare the recorded observation/action contract. Shared simulation code is not a real-robot safety certification.
