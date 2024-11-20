# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul


gripper_angle_thres = 0.001
cube_holding_dist_thres = 0.01

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def full_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    cube_pos_w = env.scene['cube_frame'].data.target_pos_w[..., 0, :]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    cube_rot_w = env.scene['cube_frame'].data.target_quat_w[..., 0, :]
    ee_rot_w = ee_frame.data.target_quat_w[..., 0, :]
    
    # Get gripper angle (0.04 is fully open, 0 is fully closed)
    asset = env.scene['robot']
    asset_cfg = SceneEntityCfg("robot")
    angles = asset.data.joint_pos[:, asset_cfg.joint_ids]
    gripper_angle = angles[:, 21]
    
    # Normalize gripper state for reward calculations (1 is closed, 0 is open)
    normalized_gripper_state = 1.0 - (gripper_angle / 0.04)
     
    # Calculate rewards
    reaching_reward = calculate_reaching_reward(ee_pos_w, cube_pos_w)
    orientation_reward = calculate_orientation_reward(ee_rot_w, cube_rot_w)
    grasping_reward = calculate_grasping_reward(normalized_gripper_state, ee_pos_w, cube_pos_w)
    closing_reward = calculate_closing_reward(normalized_gripper_state, ee_pos_w, cube_pos_w)
    stability_reward = calculate_stability_reward(ee_pos_w, cube_pos_w, normalized_gripper_state)
    
    # Combine rewards with appropriate weights - increased emphasis on grasping and closing
    total_reward = (
        1.0 * reaching_reward + 
        0.5 * orientation_reward + 
        1.5 * grasping_reward + 
        2.0 * closing_reward +  # New reward component
        2.0 * stability_reward  # Increased weight for stability
    )
    
    return total_reward

def calculate_reaching_reward(ee_pos, cube_pos):
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    reaching_reward = torch.exp(-5.0 * distance)
    return reaching_reward

def calculate_orientation_reward(ee_rot, cube_rot):
    quat_diff = quaternion_difference(ee_rot, cube_rot)
    angle_diff = 2 * torch.acos(torch.clamp(quat_diff[..., 0], -1.0, 1.0))
    orientation_reward = torch.exp(-2.0 * angle_diff)
    return orientation_reward

def calculate_grasping_reward(normalized_gripper_state, ee_pos, cube_pos):
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    close_enough = distance < cube_holding_dist_thres
    
    # Enhanced reward that scales with both proximity and closure
    distance_factor = torch.exp(-10.0 * distance)
    grasping_reward = torch.where(
        close_enough,
        distance_factor * normalized_gripper_state**2,  # Quadratic scaling for gripper closure
        torch.zeros_like(normalized_gripper_state)
    )
    return grasping_reward

def calculate_closing_reward(normalized_gripper_state, ee_pos, cube_pos):
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    very_close = distance < cube_holding_dist_thres
    
    # Strong incentive to close the gripper when very close
    closing_reward = torch.where(
        very_close,
        3.0 * normalized_gripper_state**3,  # Cubic scaling for even stronger emphasis on closing
        torch.zeros_like(normalized_gripper_state)
    )
    return closing_reward

def calculate_stability_reward(ee_pos, cube_pos, normalized_gripper_state):
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    
    # More nuanced stability assessment
    perfect_grasp = (distance < cube_holding_dist_thres) & (normalized_gripper_state > 0.95)
    good_grasp = (distance < cube_holding_dist_thres * 1.5) & (normalized_gripper_state > 0.8)
    
    # Graduated rewards for different quality grasps
    stability_reward = torch.where(
        perfect_grasp,
        2.0 * torch.ones_like(distance),  # Higher reward for perfect grasp
        torch.where(
            good_grasp,
            torch.ones_like(distance),    # Standard reward for good grasp
            torch.zeros_like(distance)    # No reward otherwise
        )
    )
    return stability_reward

def quaternion_difference(q1, q2):
    q1_inv = torch.cat([-q1[..., 1:], q1[..., :1]], dim=-1)
    return quaternion_multiply(q1_inv, q2)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)