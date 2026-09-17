from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv


class ReferenceJointPositionAction(JointPositionAction):
    """参考中心残差动作: q_des = 参考当前帧关节角 + scale * clip(action)。

    与固定 offset 的 JointPositionAction 不同, 动作基准是动画逐帧参考而不是
    第 0 帧。这样 scale=0.25 只表示残差修正范围 (±0.25 rad), 完整舞蹈动作
    由参考直接提供, 与部署端 ``q_des = ref + scale * action`` 语义一致。
    """

    cfg: ReferenceJointPositionActionCfg

    def apply_actions(self) -> None:
        # 当前帧参考关节角 (N, num_dofs)
        animation_term = self._env.animation_manager.get_term(self.cfg.animation)
        ref_joint_pos = animation_term.get_dof_pos()[:, 0, :]
        # 目标 = 参考 + 残差 (offset 已置 0, processed = scale * action)
        target = self.processed_actions + ref_joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class ReferenceJointPositionActionCfg(JointPositionActionCfg):
    """参考中心残差关节位置动作配置。"""

    class_type: type[ActionTerm] = ReferenceJointPositionAction
    animation: str = "animation"
    """动画管理器中的参考项名称 (用于取当前帧关节角)。"""
