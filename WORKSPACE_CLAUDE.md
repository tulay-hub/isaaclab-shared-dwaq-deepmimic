# Lens110 工作区开发与验证约定

本文件是整理后的根级开发约定。历史 T800 whole-body tracking 说明保存在
`archive/legacy/docs/CLAUDE_legacy_t800.md`，不再作为当前 Lens110 项目的入口。

## 项目边界

- 六个任务项目位于 `projects/01_*` 至 `projects/06_*`，每个项目拥有自己的 `data/`、`exports/`、
  `experiments/` 和 `docs/`。
- Isaac Lab/DWAQ/DeepMimic 共享源码只维护在
  `frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot`。
- 半身 upper/lower AMP 源码位于
  `projects/02_dance_half_body/framework/legged_lab_upper_lower`。
- AMP GetUp 源码位于 `projects/04_fall_to_stand/framework/amp_mjlab/AMP_mjlab`。
- 两个重定向工具位于 `tools/retargeting/gmr_lens110` 和 `tools/retargeting/robot_retargeter`。

## 不变量

- Lens110 是 21-DOF；不要把 DeepMimic 的 USD 顺序、159-D URDF 顺序、GetUp 的 MJCF 顺序和 SDK 顺序
  直接按索引混用。
- GMR pkl/CSV 根四元数使用 `xyzw`；只有在进入 MuJoCo `qpos` 时转换到 `wxyz`。
- 21-DOF 部署 CSV 为 `[root_pos(3), root_rot_xyzw(4), 21 dof]`，并保留文件末尾逗号。
- 全身舞蹈当前 H 版是 `161` 维观测、`21` 维动作；159-D 变体删除双脚接触项并使用 URDF 分组顺序。
- GetUp actor 输入为 `4 * 72 = 288`，输出为 `21`；物理 500 Hz、策略 100 Hz，踝 PD 需与导出配置一致。
- DWAQ actor 不接收直接 base linear velocity；速度由历史观测和 beta-VAE context 处理，critic 可以使用
  privileged velocity。

## 修改与验证

修改训练、观测、动作、奖励、终止、地形或导出代码后：

1. 对修改的 Python 执行 `python -m py_compile`。
2. 对 shell 执行 `bash -n`，对文本执行 `git diff --check`。
3. 先做配置加载/任务注册检查，再做 1 iteration foreground smoke test。
4. 涉及 motion 时，检查 schema、fps、joint order、root/quaternion 约定，并优先实际回放。
5. 涉及 ONNX 时，用 `onnx.checker` 检查模型，并记录输入输出维度。
6. 涉及楼梯时，验证相邻台阶高度语义、地形可见性、课程升降级和实际最大 riser；
   `terrain_levels` 是行索引/平均难度，不是百分比。

不要因为整理目录而重新训练、覆盖 checkpoint、删除历史产物或改变奖励数值。训练健康监控保持手动查询，
不要默认启动持久守护。

## 真机安全

任何会启用、保持或切换 EtherCAT/机器人控制状态的动作，都必须先做 endpoint、SSH、service、listener 和
launcher 的只读核查，并获得明确批准。目录整理、静态检查和 MuJoCo/Isaac playback 不等于真机授权。
