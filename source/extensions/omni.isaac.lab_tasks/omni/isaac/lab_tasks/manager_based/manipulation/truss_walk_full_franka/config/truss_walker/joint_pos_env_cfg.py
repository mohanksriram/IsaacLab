# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.truss_walk_full_franka import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.truss_walk_full_franka.truss_walk_full_franka_env_cfg import TrussWalkerEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from omni.isaac.lab_assets.space_walker_franka_hand import SPACE_WALKER_CFG  # isort: skip
from omni.isaac.lab_assets.space_walker_franka_hand import SPACE_WALKER_CFG  # isort: skip


@configclass
class TrussWalkerTrussEnvCfg(TrussWalkerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = SPACE_WALKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # import pdb; pdb.set_trace()
        # Set actions for the specific robot type (franka)
        self.actions.left_body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda1_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.left_finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda1_finger_joint1", "panda1_finger_joint2"],
            open_command_expr={"panda1_finger_joint1": 0.04, "panda1_finger_joint2": 0.04},
            close_command_expr={"panda1_finger_joint1": 0.0, "panda1_finger_joint2": 0.0},
        )
        self.actions.right_body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda2_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.right_finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda2_finger_joint1", "panda2_finger_joint2"],
            open_command_expr={"panda2_finger_joint1": 0.04, "panda2_finger_joint2": 0.04},
            close_command_expr={"panda2_finger_joint1": 0.0, "panda2_finger_joint2": 0.0},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda2_hand"

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.cube_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Truss/got_full_truss/truss_line_1/truss_main_2/Tritruss2",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Truss/got_full_truss/truss_line_1/truss_main_1/Tritruss1",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(-0.1582, -0.09067, -0.038756),
                ),
                )
            ],
        )        

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/leg2/franka2/panda2_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg.copy(),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/leg2/franka2/panda2_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
            ],
        )


@configclass
class TrussWalkerTrussEnvCfg_PLAY(TrussWalkerTrussEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
