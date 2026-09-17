# 整理与验证状态

更新时间：2026-09-17（本地链路复核与 GitHub 发布前整理）。

## 已完成

- 六个任务项目已建立：全身舞蹈、半身 upper/lower、行走、跌倒起身、侧滚、上台阶；
- 共享 Isaac Lab/DWAQ/DeepMimic 基座已迁移到 `frameworks/shared/lens110_isaaclab`；
- 两套重定向工具已迁移到 `tools/retargeting/`；
- 原始 BVH 进入 `datasets/dance/raw_bvh/original_bvh`；动作、导出包和旧框架已按项目/归档分层；
- 根 Git 已初始化为 `main`，当前没有 commit，也没有执行 push；
- 两个嵌套 `.git` 已保留为 `.git-legacy`，避免形成 embedded repository/gitlink；
- 新增根 README、项目地图、接口契约、奖励结构、GitHub 发布清单和六份项目奖励说明；
- 新增楼梯 DWAQ 训练/PLAY 配置和注册：`LeggedLab-Isaac--DWAQ-Lens110-Stairs-v0`、
  `LeggedLab-Isaac--DWAQ-Lens110-Stairs-PLAY-v0`；
- 楼梯导出目录已与行走项目分离；当前没有未经楼梯验证的楼梯压缩包，不再把 `03_walk` 包作为楼梯包入口；
- 旧行走 12-DoF MuJoCo 入口已改为本地导出包/XML，短时无渲染仿真 `30` 个控制步通过；
- Python AST 解析 `696/696`、Shell 语法 `31/31`、JSON 解析 `138/138` 通过；活动代码/配置无旧机器绝对路径；
- 独立 MuJoCo 根模型 `117/117` 编译通过；X2 `4/4` XML 编译通过；H1 场景重复纹理已重命名修复；
- URDF 网格引用 `3843` 个，其中本地引用 `2909` 个全部存在，ROS `package://` 引用 `934` 个保持为显式外部包语义；
- canonical 动作数据 `29` 个 PKL、起身 `9` 个 NPZ 的字段、21 DoF、帧数和有限值检查通过；两个本地 12-DoF TorchScript 均为 `45 -> 12`；
- 活动 Markdown 本地链接 `54` 个全部有效；全项目软链接扫描断链为 `0`；
- Python `py_compile`、shell `bash -n`、GMR 默认 MJCF/训练目录解析、六个训练 wrapper dry-run 和关键入口 `--help` 已通过；
- `deployment/infer_zero/infer_zero/latest` 已改为本地 `log/latest`，不再指向另一台机器；
- 当前活动 MuJoCo DWAQ 回放 PID `677422`、TensorBoard PID `3955808` 未停止。

## 尚未完成/不能宣称

- 在本机当前终端中，直接 `isaaclab` Python 导入因缺少 `pxr` 失败；
- 使用 `/home/tulay/IsaacLab/isaaclab.sh` 的真实启动尝试因缺少
  `/home/tulay/IsaacLab/_isaac_sim/python.sh` 停止；
- 因此楼梯新任务目前是“代码编译/注册通过，Isaac Lab 配置加载和短时 smoke test 待补”，不是“楼梯已收敛”；
- 本次没有启动训练、没有重新导出模型、没有进行真机动作、没有修改历史日志和 checkpoint；
- 共享 Isaac Lab 仓库中的 PR/PR-UP/UL 某些 sim2sim 配置仍属于项目 02 的具体策略/踝映射 profile；拆成独立 GitHub 仓库时应通过明确的项目依赖或子模块提供，不应把项目 02 权重伪装成共享默认；
- 当前根 Git 仍无 commit、无 remote、未执行 `git push`；GitHub CLI 已登录 `tulay-hub`，新仓库按最新决定使用 Public；
- `git-lfs` 当前未安装，`.pt/.onnx/.pkl/.npz` 和大包只保留在本地并按 `docs/GITHUB_RELEASE_CHECKLIST.md` 等待 LFS/Release 方案；
- GitHub 远程仓库尚未创建或上传；仓库名和 Public 可见性已确定后即可执行创建和 push。

## 下一次验证顺序

1. 准备与当前 `Isaac Sim` 版本匹配的 `_isaac_sim/python.sh` 或正确 AppLauncher 环境；
2. 在 `frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot` 加载并打印平地 DWAQ、全身、侧滚、楼梯任务；
3. 对楼梯做地形可见性、相邻 riser 高度、课程 row 0/升降级和 1 iteration smoke test；
4. 再按项目 README 分别做 playback、ONNX checker 和导出包检查；
5. 最后才由用户决定 GitHub 远程、LFS/Release 内容和是否 commit/push。
