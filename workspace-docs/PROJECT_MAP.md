# 项目地图与迁移关系

本文件是当前目录的唯一总资产地图。中文显示名用于人读，ASCII 目录名用于脚本和 GitHub。

## 当前项目归属

| 新目录 | 中文项目 | 训练源码/共享底座 | 训练数据 | 导出/实验 |
|---|---|---|---|---|
| `projects/01_dance_whole_body` | 跳舞全身 | `frameworks/shared/lens110_isaaclab` 的 DeepMimic | `data/processed/retargeted_actions`、`data/training/deepmimic_motion` | `exports/versions`、`exports/packages`、`experiments/deepmimic_runs` |
| `projects/02_dance_half_body` | 跳舞半身（upper/lower） | `framework/legged_lab_upper_lower` | `data/models/pitchRoll2UpperLower`、`data/training/motion_data` | `exports/training_exports`、`experiments/runs` |
| `projects/03_walk` | 行走 | 共享 DWAQ | 命令行走，不依赖 reference motion；地形入口在 `data/terrain/source` | `exports/packages`、`exports/legacy`、`experiments/dwaq_runs` |
| `projects/04_fall_to_stand` | 跌倒起身 | `framework/amp_mjlab/AMP_mjlab` | `data/motions/walk0821`、`data/training/amp_lens110_motions` | `exports/versions`、`experiments/runs` |
| `projects/05_side_roll` | 翻滚/侧滚 | 共享 DeepMimic 159-D 配置 | `data/motions/side_roll_retargeted`、`data/training/deepmimic_motion` | `exports/versions`、`experiments/deepmimic_runs` |
| `projects/06_stairs` | 上台阶 | 共享 DWAQ + 独立 `Stairs` task | `data/terrain/source` | `exports/packages`、`experiments/dwaq_runs` |

## 旧目录到新目录

| 原路径 | 新路径 | 处理方式 |
|---|---|---|
| `lens110RL` | `frameworks/shared/lens110_isaaclab` | 实体目录已迁移；根保留兼容软链接 |
| `lens110-amp-mjlab` | `projects/04_fall_to_stand/framework/amp_mjlab` | 实体目录已迁移；根保留兼容软链接 |
| `资料/legged_lab_gitee` | `projects/02_dance_half_body/framework/legged_lab_upper_lower` | 实体目录已迁移；`资料/` 下保留兼容软链接 |
| `GMR-master` | `tools/retargeting/gmr_lens110` | 当前 GMR 工具入口；根保留兼容软链接 |
| `robot_retargeter` | `tools/retargeting/robot_retargeter` | 当前第二套重定向工具；根保留兼容软链接 |
| `资料/gmr_lbot` | `archive/reference/gmr_lbot_validated` | 已验证参考副本归档，不作为第三套工具 |
| `动作1` | `projects/01_dance_whole_body/data/processed/retargeted_actions/tangbohushuo_dance` | 保留内部文件名 |
| `动作2` | `projects/01_dance_whole_body/data/processed/retargeted_actions/fast_cars_super_stars` | 保留内部文件名 |
| `动作3` | `projects/01_dance_whole_body/data/processed/retargeted_actions/bangbangbang` | 保留内部文件名 |
| `动作` | `projects/01_dance_whole_body/data/processed/retargeted_actions/legacy_k1_and_dance` | 混合历史动作，暂不强行细分 |
| `动作4` | `projects/02_dance_half_body/data/processed/retargeted_actions/mixed_k1_mj_motion` | 当前归入半身/upper-lower 相关实验，状态保留为 mixed |
| `动作5` | `projects/05_side_roll/data/motions/side_roll_retargeted` | 侧滚重定向动作 |
| `舞蹈BVH` | `datasets/dance/raw_bvh/original_bvh` | 共享原始舞蹈源；项目内 `data/raw/bvh` 为入口链接 |
| `walk0821` | `projects/04_fall_to_stand/data/motions/walk0821` | 含跌倒/恢复质量修复和 100 Hz 动作 |
| `Lens110_Dance_*` | `projects/01_dance_whole_body/exports/versions` | 各版本舞蹈 sim2real 包集中管理 |
| `Lens110_GetUp_*`、`lens110-amp-mjlab_getup_*` | `projects/04_fall_to_stand/exports/versions` | 起身导出包和复现包集中管理 |
| `Lens110_SideRoll_*` | `projects/05_side_roll/exports/versions` | 侧滚导出包集中管理 |
| `资料/Lens110_ThreeAction_Walk_Marktime11999_Minimal` | `projects/03_walk/exports/legacy` | 旧行走发布包 |
| 根 `base/`、`data/`、`scripts/`、`source/`、`pyproject.toml` | `archive/legacy/t800_flat_tracking` | 旧 T800 whole-body tracking 工程，不与 Lens110 主线混用 |
| `_archive_20260828` | `archive/legacy/archive_20260828` | 历史框架归档 |
| `infer_zero` | `deployment/infer_zero` | 硬件推理/ROS 资产 |
| `lens110RL_hardware` | `deployment/lens110_hardware` | 硬件模型和说明 |

## 软链接与历史原则

兼容软链接只为让已存在的绝对路径、回放进程和旧脚本过渡使用；新文档、新脚本和 GitHub 入口一律使用
新路径。`experiments/` 中的旧 `env.yaml` 记录了当时的绝对路径，属于复现实验证据，不批量改写。

根目录旧说明已原文保存于 `archive/legacy/docs/`。大型压缩包集中在
`archive/release_packages/`，不再与当前源码平铺。
