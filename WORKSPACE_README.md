# Lens110 Motion and Reinforcement-Learning Collection

这是 Lens110 人形机器人动作重定向、模仿学习、盲行走、跌倒恢复和 sim2real 资料的统一工作区。
原来按 `动作1`、`动作2`、`lens110RL`、`资料` 平铺的目录已经按任务重新命名；每个任务都有自己的
`framework/`、`data/`、`exports/`、`experiments/` 和 `docs/`。

## 六个独立项目

| 项目 | 中文名称 | 训练方法 | 当前任务入口 | 说明 |
|---|---|---|---|---|
| [`01_dance_whole_body`](projects/01_dance_whole_body/README.md) | 跳舞全身 | DeepMimic / reference residual | `LeggedLab-Isaac--Deepmimic-Lens110-v0` | 当前 H 版 `161 -> 21` |
| [`02_dance_half_body`](projects/02_dance_half_body/README.md) | 跳舞半身（upper/lower） | AMP + upper/lower ankle IO | `LeggedLab-Isaac-AMP-Lens110-UpperLower-v0` | 含 polynomial、MLP、XML tendon、MJCF 变体 |
| [`03_walk`](projects/03_walk/README.md) | 行走 | DWAQ（PPO + beta-VAE） | `LeggedLab-Isaac--DWAQ-Lens110-v0` | 500 Hz 物理、100 Hz 策略、60 s 回合 |
| [`04_fall_to_stand`](projects/04_fall_to_stand/README.md) | 跌倒起身 | AMP | `Lens110-AMP-GetUp` | `288 -> 21`，站立/恢复动作分开采样 |
| [`05_side_roll`](projects/05_side_roll/README.md) | 翻滚（侧滚） | DeepMimic / 159-D URDF order | `LeggedLab-Isaac--Deepmimic-Lens110-SideRoll-v0` | 滚地期间放宽姿态和接触终止 |
| [`06_stairs`](projects/06_stairs/README.md) | 上台阶 | DWAQ + stair curriculum | `LeggedLab-Isaac--DWAQ-Lens110-Stairs-v0` | 本次补齐独立训练/PLAY 注册入口 |

每个项目的训练数据和导出包只在自己的 `data/`、`exports/` 下管理。共享算法源码不复制六份，统一放在
[`frameworks/shared/lens110_isaaclab`](frameworks/shared/lens110_isaaclab)，项目目录中的
`framework/isaaclab_shared` 是可见入口软链接。

## 总体目录

```text
.
├── projects/
│   ├── 01_dance_whole_body/     # 跳舞全身：DeepMimic 161-D 主线
│   ├── 02_dance_half_body/      # 跳舞半身：upper/lower 踝执行器映射
│   ├── 03_walk/                 # 行走：DWAQ 平地/命令行走
│   ├── 04_fall_to_stand/        # 跌倒起身：AMP GetUp
│   ├── 05_side_roll/             # 翻滚：SideRoll 159-D
│   └── 06_stairs/               # 上台阶：DWAQ 楼梯课程
├── frameworks/shared/            # 只维护一份共享 Isaac Lab/DWAQ/DeepMimic 基座
├── tools/retargeting/            # 两套重定向工具
├── datasets/dance/raw_bvh/       # 原始 BVH 舞蹈源数据
├── deployment/                   # 推理、硬件资产和参考部署资料
├── docs/                         # 项目地图、接口、奖励和 GitHub 发布规则
└── archive/                      # 历史框架、旧包和原始说明，不作为当前入口
```

## 先读这几份说明

- [项目地图与迁移关系](docs/PROJECT_MAP.md)
- [观测、动作、关节顺序和四元数契约](docs/INTERFACE_CONTRACTS.md)
- [六个项目的奖励框架结构](docs/REWARD_FRAMEWORKS.md)
- [两套重定向工具说明](tools/retargeting/README.md)
- [GitHub 发布前检查清单](docs/GITHUB_RELEASE_CHECKLIST.md)

## 环境边界

- Isaac Lab 训练使用 `frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot`；AMP GetUp 使用
  `projects/04_fall_to_stand/framework/amp_mjlab/AMP_mjlab`。
- GMR 重定向使用 `gmr` 环境；已验证链路是 `BVH(nokov) -> unitree_g1 -> lens110_21dof`，不要再使用
  已废弃的 BVH 直连 Lens110 配置作为默认流程。
- 训练、回放、导出和真机控制是不同操作。当前整理不会启动训练、不会连接/控制真机，也不会执行
  `git push` 或删除旧数据。
- `deployment/reference_docs/unverified_robot_bundle/` 明确标记为不可信/未审计资料，默认不应直接部署。

## 当前验证状态

- 全身舞蹈 H 版、侧滚 159-D、DWAQ 走行代码和起身 AMP 都有既有运行记录与导出包。
- 楼梯训练入口和 PLAY 入口已注册到共享 DWAQ 工程，但重组后的新入口仍需在 Isaac Lab 环境中重新做一次
  配置加载和短时 smoke test；不能仅凭 Python 编译声称楼梯已经收敛。
- 当前运行中的 DWAQ MuJoCo 回放和 TensorBoard 只读保留，不在整理过程中停止。

## 许可

根目录保留原项目的 [LICENSE](LICENSE)。第三方代码、机器人模型、动作数据和发布包应在公开 GitHub 前逐项
确认其许可证和分发权限。
