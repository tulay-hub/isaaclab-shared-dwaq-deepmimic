# 发布范围 / Publish Scope

## 中文

本仓库只保留共享训练/MuJoCo 框架和当前可复现所需的最新资产：

- Isaac Lab、DWAQ、DeepMimic、terrain、MuJoCo 源码、任务配置和机器人 XML/网格；
- 当前 DWAQ 回放引用的 `model_36600.pt`；
- 同一训练 run 的最新 `model_36700.pt`、`policy_dwaq_urdf.onnx`、必要参数和 canonical MotionData；
- 当前最新行走发布包。

旧 checkpoint、历史 TensorBoard/log、archive、deployment 和重复训练产物没有进入本次 Public commit；它们仍在本机的完整 staging backup 中。共享仓库不绑定某个项目的策略，其他项目策略必须显式提供。

## English

This repository keeps the shared training/MuJoCo framework and only the newest assets needed for current reproduction:

- Isaac Lab, DWAQ, DeepMimic, terrain, MuJoCo source, task configuration, and robot XML/mesh assets;
- `model_36600.pt`, which is referenced by the current DWAQ MuJoCo replay;
- `model_36700.pt`, the newest checkpoint in the same run, plus `policy_dwaq_urdf.onnx`, required params, and canonical MotionData;
- the latest walking release package.

Old checkpoints, historical TensorBoard/log data, archive, deployment, and duplicate training outputs are not part of this Public commit; they remain in the complete local staging backup. The shared repository does not bind to one project's policy; consuming projects must provide policies explicitly.
