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

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.walker_arm_robotiq import WALKER_ARM_ROBOTIQ_CFG  # isort: skip


@configclass
class WalkerArmRobotiqCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = WALKER_ARM_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_.*"], scale=0.5, use_default_offset=True
        )
        self.actions.finger_joint_pos = mdp.BinaryJointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
            open_command_expr={"left_outer_knuckle_joint": 0.0, "right_outer_knuckle_joint": 0.0},
            # close_command_expr={"finger_joint": -80.0, "right_outer_knuckle_joint": -80.0},
            # open_command_expr={"finger_joint": -80.0, "right_outer_knuckle_joint": -80.0},
            # open_command_expr={"finger_joint": 80.0, "right_outer_knuckle_joint": 80.0},
            close_command_expr={"left_outer_knuckle_joint": 50.0, "right_outer_knuckle_joint": -50.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "robotiq_arg2f_base_link"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.37, 0, 0], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                # scale=(1.0, 0.8, 0.8),
                # scale=(1.0, 1.0, 1.0),
                # scale=(0.8, 0.8, 0.8),
                scale=(0.5, 0.5, 0.5),
                # scale=(0.9, 0.9, 0.9),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=10.0,
                    max_linear_velocity=10.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/robot_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/robotiq_arg2f_base_link",
                    # prim_path="{ENV_REGEX_NS}/Robot/EndEffector_Link",
                    name="end_effector",
                    offset=OffsetCfg(
                        # pos=[0.0, 0.0, 0.1034],
                        # pos=[0.0, 0.0, 0.166],
                        # pos=(0.0, 0.0, 0.0),
                        pos=(0.0, 0.0, 0.225),
                        # rot=(0.48, 0.51, -0.52, 0.47)
                        # pos=[0.0, 0.0, 0.08],
                    ),
                ),
            ],
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
