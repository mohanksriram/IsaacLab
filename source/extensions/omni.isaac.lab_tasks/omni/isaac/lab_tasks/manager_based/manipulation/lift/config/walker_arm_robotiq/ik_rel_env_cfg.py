# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.walker_arm_robotiq import WALKER_ARM_ROBOTIQ_CFG  # isort: skip


@configclass
class WalkerArmRobotiqCubeLiftEnvCfg(joint_pos_env_cfg.WalkerArmRobotiqCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = WALKER_ARM_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint_.*"],
            body_name="EndEffector_Link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.17], rot=[0.48, 0.51, -0.52, 0.47]),
        )


@configclass
class WalkerArmRobotiqCubeLiftEnvCfg_PLAY(WalkerArmRobotiqCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
