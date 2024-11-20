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

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# Constants
GRASPING_DISTANCE_THRESHOLD = 0.008  # Threshold for considering grasping complete
CONTACT_FORCE_THRESHOLD = 0.1  # Threshold for contact force in stability
# EXISTENCE_PENALTY = -0.01  # Penalty for each time step
EXISTENCE_PENALTY = 0.0  # Penalty for each time step

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
    left_gripper_angle = angles[:, 21]
    right_gripper_angle = angles[:, 22]
    gripper_angle = (left_gripper_angle + right_gripper_angle) / 2
    
    # Normalize gripper state for reward calculations (1 is closed, 0 is open)
    normalized_gripper_state = (gripper_angle / 0.04)
    # exp_normalized_gripper_state = torch.exp(-5.0 * normalized_gripper_state)


    # Get contact forces
    contact_forces = env.scene['contact_forces'].data.net_forces_w
    left_gripper_contact = contact_forces[:, 0]
    right_gripper_contact = contact_forces[:, 1]
    
    # Calculate individual reward components
    reaching_reward = calculate_reaching_reward(ee_pos_w, cube_pos_w)
    orientation_reward = calculate_orientation_reward(ee_rot_w, cube_rot_w)
    grasping_reward = calculate_grasping_reward(normalized_gripper_state, ee_pos_w, cube_pos_w, left_gripper_contact, right_gripper_contact)
    
    # Combine rewards
    combined_reward = 2.0 * reaching_reward + 2.0 * orientation_reward + 3.0 * grasping_reward
    print_progress(reaching_reward, orientation_reward, grasping_reward, 0.0, torch.norm(ee_pos_w - cube_pos_w, dim=-1))
    # Apply existence penalty
    final_reward = combined_reward + EXISTENCE_PENALTY
    
    return final_reward

def calculate_reaching_reward(ee_pos, cube_pos):
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    return torch.exp(-5.0 * distance)

def calculate_orientation_reward(ee_rot, cube_rot):
    quat_diff = quat_error_magnitude(ee_rot, cube_rot)
    return torch.exp(-2.0 * quat_diff)

def calculate_grasping_reward(normalized_gripper_state, ee_pos, cube_pos, left_contact, right_contact):
    distance = torch.norm(ee_pos - cube_pos, dim=-1)
    total_contact_magnitude = torch.norm(left_contact, dim=1) + torch.norm(right_contact, dim=1)
    
    # Encourage opening when far from object
    # grasp_reward = (2*(distance < GRASPING_DISTANCE_THRESHOLD).float() - 1) * normalized_gripper_state

    grasp_reward = (2*((distance < GRASPING_DISTANCE_THRESHOLD).float()) - 1) * normalized_gripper_state

    return grasp_reward

def print_progress(reaching_reward, orientation_reward, grasping_reward, stability_reward, distance):
    progress = {
        "Reaching": reaching_reward.mean().item(),
        "Orientation": orientation_reward.mean().item(),
        "Grasping": grasping_reward.mean().item(),
        "Stability": stability_reward,
        "Distance": distance.mean().item()
    }
    print("Progress:", {k: f"{v:.4f}" for k, v in progress.items()})