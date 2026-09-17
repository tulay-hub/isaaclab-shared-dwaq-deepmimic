# GitHub 发布前检查清单

当前工作区约 80G，包含 checkpoint、TensorBoard event、SMPL-X 模型、AMASS/机器人数据集、压缩包和硬件资料。
“整理好”不等于把全部本机缓存直接提交到 GitHub；公开仓库要让代码可读、入口可复现，同时避免超出 GitHub
单文件限制、泄露本机路径或发布未经审计的硬件包。

## 建议的发布分层

### 默认 Git 提交

- 根 README、`docs/`、六个项目 README 和奖励/接口说明；
- Python、shell、YAML、JSON、URDF/MJCF 等可审查的小型配置；
- 小型示例 motion 和必要的模型资产；
- 每个 `data/`、`exports/` 下的目录说明与 manifest；
- `LICENSE`、第三方许可证清单。

### Git LFS 或外部 Release

- `.pt`、`.pth`、`.onnx`、`.npz`、`.pkl`、TensorBoard event；
- 大型 STL/DAE/USD、SMPL-X body model、AMASS 数据集；
- `projects/*/exports/versions` 中的整包 zip/tar；
- `tools/retargeting/robot_retargeter/dataset` 的私有/大型数据。

### 默认不发布

- `deployment/reference_docs/unverified_robot_bundle/`；
- 机器专属的 `/home/orangepi`、`/home/lbot`、`/home/tulay` 绝对链接或日志；
- 实机网络、service、SDK 凭据和未审计的控制配置；
- `__pycache__`、`.egg-info`、build/install/log/output/video 缓存。

## 建议的独立 GitHub 仓库

当前 `tulay-hub` 下没有占用以下名称的仓库。按当前决定，以下仓库创建为 Public；训练权重、原始动作数据、body model、硬件 SDK 和未审计部署包仍按下面的规则排除或单独发布，不能因为仓库公开就直接提交。

| 仓库 | 内容 | 本地依赖边界 |
|---|---|---|
| `dance-whole-body-deepmimic` | 跳舞全身项目 01 | 共享 Isaac Lab 基座、舞蹈数据仓库 |
| `dance-half-body` | 跳舞半身项目 02 | 自带 upper/lower 框架、PR/PR-UP/UL 映射和本地策略 |
| `walk-dwaq-ppo-beta-vae` | 行走项目 03 | 共享 Isaac Lab/DWAQ 基座 |
| `fall-to-stand-amp-getup` | 跌倒起身项目 04 | 自带 AMP GetUp 框架和起身数据 |
| `side-roll-deepmimic` | 翻滚项目 05 | 共享 DeepMimic 基座、舞蹈数据仓库 |
| `stairs-dwaq-ppo-beta-vae` | 上台阶项目 06 | 共享 DWAQ 基座；不复用行走导出包 |
| `isaaclab-shared-dwaq-deepmimic` | 共享 Isaac Lab/DWAQ/DeepMimic 基座 | 被项目 01/03/05/06 作为显式本地子模块使用 |
| `gmr-retargeting` | GMR/Lens110 重定向工具 | 可选使用舞蹈数据仓库和机器人重定向资产 |
| `robot-retargeter-smplx` | robot_retargeter 重定向工具 | 独立维护 body model/机器人资产边界 |
| `dance-dataset` | 原始 BVH 舞蹈数据和可发布清单 | 仅作为数据子模块/Release，不把私有模型混入代码仓库 |

独立仓库不能依赖当前工作区的根目录软链接。上传时应将 `framework/isaaclab_shared`、`data/raw/bvh` 等跨仓库入口转换为明确的 Git submodule 或 Release 下载说明；项目 02 的 PR/PR-UP/UL profile 留在项目 02，不把其 checkpoint 和踝映射伪装成共享默认。

推荐的克隆方式是 `git clone --recurse-submodules`，并在每个仓库 README 中写明对应 commit、策略输入输出、关节顺序和动作数据 checksum。

## 提交前命令

在根目录执行：

```bash
# 1. 确认没有把大型文件误放进待提交区
find . -type f -not -path './.git/*' -size +95M -printf '%s\t%p\n' | sort -nr | head -100

# 2. 确认生成物和机器专属路径命中忽略规则
git check-ignore -v \
  frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/logs \
  projects/04_fall_to_stand/framework/amp_mjlab/AMP_mjlab/logs \
  deployment/infer_zero/infer_zero/latest

# 3. 查看 staged 清单，确认没有嵌套仓库 gitlink、硬件包和本机路径
git status --short
git diff --check
git ls-files | rg '(^|/)(\.git|logs|outputs|videos|__pycache__|unverified_robot_bundle)(/|$)' || true

# 4. 只读检查绝对路径和旧目录引用
rg -n '/home/|动作[0-9]?/|舞蹈BVH|GMR-master|lens110RL|lens110-amp-mjlab' \
  README.md docs projects tools frameworks deployment \
  --glob '*.md' --glob '*.py' --glob '*.sh' --glob '*.yaml' --glob '*.yml' --glob '*.json'
```

## 版本与发布原则

- 不执行 `git push`，除非用户明确要求并确认目标远程/分支。
- 不删除旧包来降低仓库大小；使用 `.gitignore`、Git LFS 或 GitHub Release，并保留校验和/manifest。
- 发布 checkpoint 时同时发布 policy interface、joint order、训练配置 commit、motion checksum 和回放结果。
- 发布实机包时把“仿真验证”“导出验证”“真机验证”分开写，不把 sim2sim 结果写成真机已验证。
