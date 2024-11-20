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

from omni.isaac.lab_tasks.manager_based.manipulation.walk_on_truss import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.walk_on_truss.walk_on_truss_env_cfg import SpaceWalkerEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from omni.isaac.lab_assets.space_walker_franka_hand import SPACE_WALKER_CFG  # isort: skip
from omni.isaac.lab_assets.space_walker_bi_franka_hand import SPACE_WALKER_CFG  # isort: skip


@configclass
class SpaceWalkerTrussEnvCfg(SpaceWalkerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = SPACE_WALKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # import pdb; pdb.set_trace()
        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_.*_2"], scale=0.5, use_default_offset=True
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            # joint_names=["finger_joint_2", "right_outer_knuckle_joint_2"],
            joint_names=["panda_finger_2_joint1", "panda_finger_2_joint2"],
            # open_command_expr={"finger_joint_2": 0.0, "right_outer_knuckle_joint_2": 0.0},
            # open_command_expr={"finger_joint_2": 50.0, "right_outer_knuckle_joint_2": -50.0},
            # close_command_expr={"left_outer_knuckle_joint_1": 0.0, "right_outer_knuckle_joint_1": 0.0},
            open_command_expr={"panda_finger_2_joint1": 0.04, "panda_finger_2_joint2": 0.04},
            # open_command_expr={"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
            # open_command_expr={"finger_joint_2": 45.0,},
            # open_command_expr={"finger_joint_2": 45.0,},
            close_command_expr={"panda_finger_2_joint1": 0.0, "panda_finger_2_joint2": 0.0},
            # close_command_expr={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
            # close_command_expr={"finger_joint_2": 45.0,},
            # close_command_expr={"finger_joint_2": 0.0,},
            # close_command_expr={"finger_joint_2": 50.0, "right_outer_knuckle_joint_2": -50.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand_2"

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.cube_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Truss/got_full_truss/truss_line_1/truss_main_2/Tritruss2",
            # prim_path="{ENV_REGEX_NS}/Truss/got_full_truss/truss_line_1/truss_main_1/Tritruss/Mushrooms/mushroom_cube_06/inner",
            # prim_path="{ENV_REGEX_NS}/Robot/arm1/actuator/robot_base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Truss/got_full_truss/truss_line_1/truss_main_1/Tritruss1",
                    # prim_path="{ENV_REGEX_NS}/Truss/got_full_truss/truss_line_1/truss_main_2/Tritruss/Mushrooms/mushroom_cube_01/inner",
                    name="end_effector",
                    offset=OffsetCfg(
                        # pos=(-15.82, -9.067, -3.8756),
                        pos=(-0.1582, -0.09067, -0.038756),
                        # pos=(-0.1582, -0.09067, -0.048756),
                        # pos=(-0.1582, -0.09067, -0.118756),
                        # pos=(-0.1582, -0.59067, -0.118756), # current working location
                        # pos=(-0.1582, -0.09067, -0.138756),
                        # pos=(-0.1582, -0.09067, -0.178756),
                        # pos=(-0.1582, -0.09067, -0.188756),
                        # pos=(-0.1582, -0.09067, -0.148756),
                        # pos=(-0.1582, -0.09067, -0.168756),
                        # pos=(-0.1582, -0.09067, -0.198756),
                        # pos=(-0.1582, -0.09067, -0.208756),
                        # pos=(-0.1582, -0.09067, -0.228756),
                        # pos=(-0.1582, -0.19067, -0.258756),
                        # pos=(-0.1582, -0.09067, -0.250756),
                        # pos=(-0.1582, -0.49067, -0.14118756),
                        # pos=(-0.1582, -0.09067, 0.182),
                        # rot=(0.0, 0.0, 0.0, 1.0),
                        # rot=(0.707,0,0.707,0),
                        # rot=(0.707,0.707,0, 0),
                        # rot=(0.707,0,0.707,0),
                        # rot=(0.707,0.707,0.,0.),
                        # rot=(0., 0.707,0.,-0.707),
                        # rot=(0., 0.707,-0.707,0.),
                        # rot=(0.5, 0.5, -0.5, 0.5),
                        # rot=(0., 0.707, 0.707, 0.),
                        # rot=(0,-0.707,0.707,0),
                    ),
                    # offset=OffsetCfg(
                    #     # pos=[0.0, 0.0, 0.1034],
                    #     # pos=[0.0, 0.0, 0.166],
                    #     # pos=(0.0, 0.0, 0.0),
                    #     # pos=(0.0, 0.0, 0.225),
                    #     # rot=(0.48, 0.51, -0.52, 0.47)
                    #     # pos=[0.0, 0.0, 0.08],
                    # ),
                ),
            ],
        )        

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/leg2/actuator2/RobotBase_Link_2",
            # prim_path="{ENV_REGEX_NS}/Robot/leg2/actuator/robot_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg.copy(),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # prim_path="{ENV_REGEX_NS}/Robot/leg2/panda_gripper_2/panda_hand/geometry/tool_center",
                    prim_path="{ENV_REGEX_NS}/Robot/leg2/panda_gripper_2/panda_hand_2",
                    # prim_path="{ENV_REGEX_NS}/Robot/leg2/gripper2/RightInnerFinger_2",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.08],
                        # pos=[0.0, 0.0, 0.1034],
                        # pos=[0.0, 0.0, 0.166],
                        # pos=(0.0, 0.0, 0.0),
                        # pos=(0.0, 0.0, 0.225),
                        # pos=(0.0, 0.0, 0.225),
                        # rot=(0.48, 0.51, -0.52, 0.47)
                        # pos=[0.0, 0.0, 0.08],
                    ),
                ),
            ],
        )


@configclass
class SpaceWalkerTrussEnvCfg_PLAY(SpaceWalkerTrussEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
