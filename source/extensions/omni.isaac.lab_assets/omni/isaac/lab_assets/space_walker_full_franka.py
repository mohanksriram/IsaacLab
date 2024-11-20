# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Space Walker robot.

The following configurations are available:

* :obj:`SPACE_WALKER_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`SPACE_WALKER_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

SPACE_WALKER_USD_PATH = "/home/mohan/Downloads/2F-85/space_walker_full_franka.usd"
# SPACE_WALKER_USD_PATH = "/mnt/Downloads/2F-85/space_walker_jason_fix.usd"

##
# Configuration
##
SPACE_WALKER_CFG = ArticulationCfg(
    
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{SPACE_WALKER_USD_PATH}",
        activate_contact_sensors=True,
        # scale=(100.0, 100.0, 100.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
 init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda._joint1": 0.0,
            "panda._joint2": 0.08727,
            "panda._joint3": 0.0,
            "panda._joint4": -1.74533,
            "panda._joint5": 0.0,
            "panda._joint6": 3.037,
            "panda._joint7": 0.741,
            "panda._finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda1_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda1_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda1_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda1_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda1_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda1_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "panda2_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda2_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda2_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda2_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda2_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda2_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of space walker robot."""


"""Configuration of Kinova arm robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
