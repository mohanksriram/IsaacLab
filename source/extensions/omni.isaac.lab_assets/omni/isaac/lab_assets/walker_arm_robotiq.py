# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/I2A-space-model-dump/model-dump/Collected_Walker_rev2/Library/robots/arms/kg3-lite/derived/usd/kinova_arm_panda_hand.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/I2A-space-model-dump/model-dump/Collected_Walker_rev2/Library/robots/arms/kg3-lite/derived/usd/kinova_arm_panda_hand.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/I2A-space-model-dump/model-dump/Collected_Walker_rev2/Library/robots/arms/kg3-lite/derived_with_gripper_and_camera_custom/kinova_plus_robotiq_flat.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_arm_best_v2.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_arm_robotiq_145.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_robotiq_hierarchy_145.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_robotiq_hierarchy_145.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_final.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_final_two_arm.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_all_arms_final.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_all_arms_final_scaled.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_fours.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_threes.usd"
# WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_twos.usd"
WALKER_ARM_USD_PATH = "/home/mohan/Downloads/2F-85/walker_twos_no_gravity.usd"

##
# Configuration
##
WALKER_ARM_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WALKER_ARM_USD_PATH}",
        activate_contact_sensors=False,
        # scale=(100.0, 100.0, 100.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(320.0, 650.0, 100.0),
        # rot=(0.707, 0., 0., -0.707),
        joint_pos={
            "joint_1_1": 0.0,
            "joint_2_1": 0.175,
            "joint_3_1": 0.0,
            "joint_4_1": 1.89,
            "joint_5_1": 0.0,
            "joint_6_1": 0.6,
            "joint_7_1": -1.57,
            ".*_finger_joint_1": 0.0,
            ".*_knuckle_joint_1": 0.0
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]_1"],
            velocity_limit=100.0,
            effort_limit={
                "joint_[1-4]_1": 39.0,
                "joint_[5-7]_1": 9.0,
            },
            stiffness={
                "joint_[1-4]_1": 40.0,
                "joint_[5-7]_1": 15.0,
            },
            damping={
                "joint_[1-4]_1": 1.0,
                "joint_[5-7]_1": 0.5,
            },
        ),
        "robotiq_hand": ImplicitActuatorCfg(
            joint_names_expr=["left_outer_knuckle_joint_1", "right_outer_knuckle_joint_1"],
            effort_limit=200.0,
            velocity_limit=300.0,
            # stiffness=0.0,
            # damping=5e3,
            stiffness=200.0,
            damping=20.0,
            # stiffness=400.0,
            # damping=80.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Walker arm with robotiq gripper robot."""


"""Configuration of Kinova arm robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
