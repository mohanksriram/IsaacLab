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
GRASPING_DISTANCE_THRESHOLD = 0.02  # Threshold for considering grasping complete
GRASPING_QUAT_THRESHOLD = 0.05  # Threshold for considering grasping complete
CONTACT_QUAT_THRESHOLD = 0.5  # Threshold for considering grasping complete

CONTACT_FORCE_THRESHOLD = 0.1  # Threshold for contact force in stability
# EXISTENCE_PENALTY = -0.01  # Penalty for each time step
EXISTENCE_PENALTY = 0.0  # Penalty for each time step
debug = False

def calculate_reaching_reward(env, ee_frame_cfg, object_cfg):
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    cube_pos_w = env.scene[object_cfg.name].data.target_pos_w[..., 0, :]
    distance = torch.norm(ee_pos_w - cube_pos_w, dim=-1)
    if debug:
        print(f"mean distance: {distance.mean().item()}")

    # penalize if that the cube z position is below the ee z position
    # import pdb; pdb.set_trace()



    reaching_reward = torch.where(distance < GRASPING_DISTANCE_THRESHOLD, 2.0 * torch.exp(-5.0 * distance), torch.exp(-5.0 * distance))
    if debug:
        print(f"mean reaching reward: {reaching_reward.mean().item()}")
    return reaching_reward

def neg_z_reward(env, ee_frame_cfg, object_cfg):
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    cube_pos_w = env.scene[object_cfg.name].data.target_pos_w[..., 0, :]
    
    # penalize if that the cube z position is below the ee z position
    print(f"ee_pos_w: {ee_pos_w[0]}")
    print(f"cube_pos_w: {cube_pos_w[0]}")
    
    neg_z_reward = torch.where(cube_pos_w[..., 2] < ee_pos_w[..., 2], -1.0, 0.0)

    return neg_z_reward

def calculate_orientation_reward(env, ee_frame_cfg, object_cfg):
    ee_rot_w = env.scene[ee_frame_cfg.name].data.target_quat_w[..., 0, :]
    cube_rot_w = env.scene[object_cfg.name].data.target_quat_w[..., 0, :]
    quat_diff = quat_error_magnitude(ee_rot_w, cube_rot_w)
    if debug:
        print(f"mean quat diff: {quat_diff.mean().item()}")
    
    return torch.where(quat_diff < GRASPING_QUAT_THRESHOLD, 2.0 * torch.exp(-2.0 * quat_diff), torch.exp(-2.0 * quat_diff))

def calculate_grasping_reward(env, ee_frame_cfg, object_cfg, robot_cfg, grasping_distance_threshold):
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    cube_pos_w = env.scene[object_cfg.name].data.target_pos_w[..., 0, :]
    distance = torch.norm(ee_pos_w - cube_pos_w, dim=-1)


    ee_rot_w = env.scene[ee_frame_cfg.name].data.target_quat_w[..., 0, :]
    cube_rot_w = env.scene[object_cfg.name].data.target_quat_w[..., 0, :]
    quat_diff = quat_error_magnitude(ee_rot_w, cube_rot_w)
    # print(f"mean quat diff: {quat_diff.mean().item()}")
    # print(f"mean distance: {distance.mean().item()}")

    is_close = (distance <= GRASPING_DISTANCE_THRESHOLD).float() * (quat_diff <= GRASPING_QUAT_THRESHOLD).float()

    # print(f"mean distance: {distance.mean().item()}")
    # import pdb; pdb.set_trace()
    asset = env.scene[robot_cfg.name]
    angles = asset.data.joint_pos[:, robot_cfg.joint_ids]
    left_gripper_angle = angles[:, 20]
    right_gripper_angle = angles[:, 21]
    # print the angle names
    # print(asset.data.joint_names[robot_cfg.joint_ids])
    # import pdb; pdb.set_trace()

    # raise error if the gripper angles are less than -0.01
    if (left_gripper_angle < -0.04).any() or (right_gripper_angle < -0.04).any():
        raise ValueError(f"Gripper angles are less than -0.01:  left: {left_gripper_angle.min().item()}, right: {right_gripper_angle.min().item()}")

    # make sure the gripper angles are positive
    left_gripper_angle = torch.abs(left_gripper_angle)
    right_gripper_angle = torch.abs(right_gripper_angle)


    gripper_angle = (left_gripper_angle + right_gripper_angle) / 2
    # gripper_angle = torch.maximum(left_gripper_angle, right_gripper_angle)

    # compute gripper_angle as the difference between the left and right gripper angles
    

    # import pdb; pdb.set_trace()
    # # normalized_gripper_state = 1-(gripper_angle / 0.04)
    normalized_gripper_state = (gripper_angle / 0.04)
    # # print(f"mean normalized gripper state: {normalized_gripper_state.mean().item()}")
    # # exp_gripper_state = torch.exp(-7 * normalized_gripper_state)
    # # exp_gripper_state = torch.exp(-7 * normalized_gripper_state)
    exp_gripper_state = torch.exp(-2 * normalized_gripper_state)

    # compute reward such that it's high only when the gripper is close to the object and normalized_gripper_state is high
    

    # grasp_reward = (2*(is_close) - 1) * exp_gripper_state

    # provide reward only when the gripper is closed
    # grasp_reward = is_close * torch.sum(0.04 - gripper_angle, dim=-1)
    grasp_reward = is_close * exp_gripper_state


    # print(f"mean grasp reward: {grasp_reward.mean().item()}")
    # print(f"weight: {(2*((distance < GRASPING_DISTANCE_THRESHOLD).float()) - 1).mean().item()}")
    # print(f"distance: {distance.mean().item()}")
    # print(f"gripper state: {normalized_gripper_state.mean().item()}")
    # print(f"mean grasp reward: {grasp_reward.mean().item()}")
    # breakpoint()
    if debug:
        print(f"mean grasp reward: {grasp_reward.mean().item()}")
    # if grasp_reward.sum() > 0:
    #     print(f"mean grasp reward: {grasp_reward.mean().item()}")
    return grasp_reward

def is_grasped(env, ee_frame_cfg, object_cfg):
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    cube_pos_w = env.scene[object_cfg.name].data.target_pos_w[..., 0, :]
    distance = torch.norm(ee_pos_w - cube_pos_w, dim=-1)
    ee_rot_w = env.scene[ee_frame_cfg.name].data.target_quat_w[..., 0, :]
    cube_rot_w = env.scene[object_cfg.name].data.target_quat_w[..., 0, :]
    quat_diff = quat_error_magnitude(ee_rot_w, cube_rot_w)

    is_close = (distance < GRASPING_DISTANCE_THRESHOLD).float() * (quat_diff < CONTACT_QUAT_THRESHOLD).float()

    # # ensure that the gripper state is closed
    # asset = env.scene['robot']
    # angles = asset.data.joint_pos[:, [20, 21]]
    # left_gripper_angle = angles[:, 0]
    # right_gripper_angle = angles[:, 1]
    # left_gripper_angle = torch.abs(left_gripper_angle)
    # right_gripper_angle = torch.abs(right_gripper_angle)
    # gripper_angle = torch.maximum(left_gripper_angle, right_gripper_angle)
    # normalized_gripper_state = 1-(gripper_angle / 0.04)

    # # check if the gripper state is close to zero
    # # print(f"mean normalized gripper state: {normalized_gripper_state.mean().item()}")
    # is_gripper_closed = (normalized_gripper_state > 0.8).float()


    return is_close #* is_gripper_closed

def stable_grasp_bonus(env, ee_frame_cfg, object_cfg, robot_cfg, contact_force_threshold):
    grasped = is_grasped(env, ee_frame_cfg, object_cfg).float()
    # import pdb; pdb.set_trace()
    # print(f"mean is_grasped: {grasped.mean().item()}")
    contact_forces = env.scene['contact_forces'].data.net_forces_w
    left_gripper_contact = contact_forces[:, 0]
    right_gripper_contact = contact_forces[:, 1]

    # use min contact magnitude to ensure that both the fingers are in contact
    min_contact_magnitude = torch.min(torch.norm(left_gripper_contact, dim=1), torch.norm(right_gripper_contact, dim=1))


    # print(f"mean min contact magnitude: {min_contact_magnitude.mean().item()}")

    # import pdb; pdb.set_trace()
    # print(f"mean min contact magnitude: {min_contact_magnitude.mean().item()}")


    # total_contact_magnitude = torch.norm(left_gripper_contact, dim=1) + torch.norm(right_gripper_contact, dim=1)
    
    is_stable = (min_contact_magnitude > contact_force_threshold).float()

    if is_stable.mean() > 0:
        # import pdb; pdb.set_trace()
        print(f"mean is_stable: {is_stable.mean().item()}")
    
    # print(f"left_gripper_contact: {torch.norm(left_gripper_contact).mean().item()}")
    # print(f"right_gripper_contact: {torch.norm(right_gripper_contact).mean().item()}")

    # grasp_bonus = (is_grasped + 1.0) * is_stable
    grasp_bonus = (grasped) * is_stable

    # print(f"mean grasp bonus: {grasp_bonus.mean().item()}")
    # if grasp_bonus.sum() > 0:
    #     print(f"mean grasp bonus: {grasp_bonus.mean().item()}")
    # print(f"mean grasp bonus: {grasp_bonus.mean().item()}")
    return grasp_bonus


def multi_stage_stable_grasping(env, ee_frame_cfg, object_cfg, robot_cfg, contact_force_threshold):
    is_grasped = calculate_grasping_reward(env, ee_frame_cfg, object_cfg, robot_cfg, 0.008).float()
    
    contact_forces = env.scene['contact_forces'].data.net_forces_w
    left_gripper_contact = contact_forces[:, 0]
    right_gripper_contact = contact_forces[:, 1]
    # total_contact_magnitude = torch.norm(left_gripper_contact, dim=1) + torch.norm(right_gripper_contact, dim=1)
    min_contact_magnitude = torch.min(torch.norm(left_gripper_contact, dim=1), torch.norm(right_gripper_contact, dim=1))
    # stage_easy = (total_contact_magnitude > 0.5 * contact_force_threshold) * 0.5
    stage_medium = (min_contact_magnitude > contact_force_threshold) * is_grasped
    stage_hard = (min_contact_magnitude > 2 * contact_force_threshold) * is_grasped

    # return stage_easy + stage_medium + stage_hard
    return stage_medium + stage_hard

def constant_reward(env, value):
    return torch.full((env.num_envs,), value, device=env.device)

def print_progress(reaching_reward, orientation_reward, grasping_reward, stability_reward, distance):
    progress = {
        "Reaching": reaching_reward.mean().item(),
        "Orientation": orientation_reward.mean().item(),
        "Grasping": grasping_reward.mean().item(),
        "Stability": stability_reward,
        "Distance": distance.mean().item()
    }
    print("Progress:", {k: f"{v:.4f}" for k, v in progress.items()})