# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from unimanip.utils.env_model import *



@configclass
class G1RobotEnvCfg(UniManipEnvCfg):
    """
    Configuration for the G1Robot Environment based on UniManip Environment, defined with the direct workflow.
    
    G1RobotEnvCfg defines: sim, robot, camera.
    UniManipEnvCfg defines: scene, object, room, ground, stage, terrain.
    """
    
    # space
    action_space = 6 + 1  # hand pose + hand joint
    action_frame = 'world'
    observation_space = 0
    
    # run
    dt_s = 1 / 60  # sim runs in 60 fps
    decimation = 2  # control runs in 60 / 2 = 30 fps
    episode_length_s = 10  # episode runs in 10 x 30 = 300 timesteps
    # sim
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=dt_s,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    
    # reward_params
    reward_params = {
        # reach action
        'diff_reach_pos_action': -0.1,
        'diff_reach_rot_action': -0.1,
        'diff_reach_joint_action': -0.1,
        'diff_reach_palm_rot_action': -0.1,
        # move action
        'diff_move_pos_action': -0.1,
        'diff_move_rot_action': -0.1,
        # hand object
        'diff_hand_object': -1.0,
        'diff_hand_body_object': -0.0,
        'diff_hand_thumb_object': -0.0,
        'diff_hand_finger_object': -2.0,
        # object goal
        'diff_object_goal': -1.0,
        # grasp flag
        'dist_hand_object': 0.2,
        'dist_thumb_object': 0.1,
        'dist_finger_object': 0.1,
        # bonus
        'bonus_grasp_object': 1.0,
        'bonus_object_goal': 2.0,
    }
    
    # control_params
    control_params = {
        # joint
        'slider_joint_stiffness': 50.0,
        'slider_joint_damping': 10.0,
        'arm_joint_stiffness': 10000.0,
        'arm_joint_damping': 1000.0,
        'hand_joint_stiffness': 2000.0,
        'hand_joint_damping': 10.0,
        # action scales
        'action_dof_scale': 1.0,
        'action_pos_scale': 0.025,
        'action_rot_scale': 0.025,
    }
    
    
    # robot_body_names:
    # ['Root', 'basex', 'base_link', 'body_link1', 'body_link2', 'arm_base_link', 'head_link1', 'arm_l_base_link', 'arm_r_base_link', 'head_link2', 'arm_l_link1', 'arm_r_link1', 'arm_l_link2', 'arm_r_link2', 'arm_l_link3', 'arm_r_link3', 'arm_l_link4', 'arm_r_link4', 'arm_l_link5', 'arm_r_link5', 'arm_l_link6', 'arm_r_link6', 'arm_l_end_link', 'arm_r_end_link', 
    # 'gripper_l_base_link', 'gripper_r_base_link', 'gripper_l_inner_link1', 'gripper_l_outer_link1', 'gripper_l_center_link', 'gripper_r_inner_link1', 'gripper_r_outer_link1', 'gripper_r_center_link', 'gripper_l_inner_link3', 'gripper_l_outer_link3', 'gripper_r_inner_link3', 'gripper_r_outer_link3', 'gripper_l_inner_link4', 'gripper_l_outer_link4', 'gripper_r_inner_link4', 'gripper_r_outer_link4', 'gripper_l_inner_link5', 'gripper_l_inner_link2', 'gripper_l_outer_link5', 'gripper_l_outer_link2', 'gripper_r_inner_link5', 'gripper_r_inner_link2', 'gripper_r_outer_link5', 'gripper_r_outer_link2']
    # robot_joint_names:
    # ['slider_basex', 'slider_basey', 'idx01_body_joint1', 'idx02_body_joint2', 'idx11_head_joint1', 'idx12_head_joint2', 'idx21_arm_l_joint1', 'idx61_arm_r_joint1', 'idx22_arm_l_joint2', 'idx62_arm_r_joint2', 'idx23_arm_l_joint3', 'idx63_arm_r_joint3', 'idx24_arm_l_joint4', 'idx64_arm_r_joint4', 'idx25_arm_l_joint5', 'idx65_arm_r_joint5', 'idx26_arm_l_joint6', 'idx66_arm_r_joint6', 'idx27_arm_l_joint7', 'idx67_arm_r_joint7', 
    # 'idx31_gripper_l_inner_joint1', 'idx41_gripper_l_outer_joint1', 'idx71_gripper_r_inner_joint1', 'idx81_gripper_r_outer_joint1', 'idx32_gripper_l_inner_joint3', 'idx42_gripper_l_outer_joint3', 'idx72_gripper_r_inner_joint3', 'idx82_gripper_r_outer_joint3', 'idx33_gripper_l_inner_joint4', 'idx43_gripper_l_outer_joint4', 'idx73_gripper_r_inner_joint4', 'idx83_gripper_r_outer_joint4', 'idx54_gripper_l_inner_joint0', 'idx53_gripper_l_outer_joint0', 'idx94_gripper_r_inner_joint0', 'idx93_gripper_r_outer_joint0']
    # robot_joint_lower_limit:
    # tensor([-3.1416, -5.0000,  0.0000,  0.0000, -1.5708, -0.3491, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, -3.1416, 
    #         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -3.1416, -3.1416, -3.1416, -3.1416], device='cuda:0')
    # robot_joint_upper_limit:
    # tensor([3.1416, 5.0000, 0.5500, 1.5708, 1.5708, 1.0472, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 
    #         1.0010, 1.0010, 1.0010, 1.0010, 3.1416, 3.1416, 3.1416, 3.1416, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000], device='cuda:0')
    
    # robot_names
    robot_name = 'g1_robot'
    robot_base_body_name = 'base_link'
    robot_hand_body_name = 'gripper_r_center_link'
    robot_arm_body_names = ['base_link', 'body_link1', 'body_link2', 'arm_base_link', 'arm_r_base_link', 'arm_r_link1', 'arm_r_link2', 'arm_r_link3', 'arm_r_link4', 'arm_r_link5', 'arm_r_link6', 'arm_r_end_link']
    robot_hand_body_names = ['gripper_r_center_link', 'gripper_r_inner_link5', 'gripper_r_outer_link5']
    robot_finger_body_names = ['gripper_r_center_link', 'gripper_r_inner_link5', 'gripper_r_outer_link5']
    robot_arm_joint_names = ['slider_basex', 'slider_basey', 'idx61_arm_r_joint1', 'idx62_arm_r_joint2', 'idx63_arm_r_joint3', 'idx64_arm_r_joint4', 'idx65_arm_r_joint5', 'idx66_arm_r_joint6', 'idx67_arm_r_joint7']
    robot_hand_joint_names = ['idx81_gripper_r_outer_joint1']
    # robot_base_body_index = 2
    # robot_hand_body_index = 31
    # robot_arm_body_indices = [2, 3, 4, 5, 8, 11, 13, 15, 17, 19, 21, 23]
    # robot_hand_body_indices = [31, 44, 46]
    # robot_finger_body_indices = [31, 44, 46]
    # robot_arm_joint_indices = [0, 1, 7, 9, 11, 13, 15, 17, 19]
    # robot_hand_joint_indices = [23]
    
    # robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        # UsdFileCfg
        spawn=sim_utils.UsdFileCfg(
            usd_path=osp.join(ASSET_DIR, 'g1_robot/g1_robot.usd'),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=1.0, #linear_damping=1.0, angular_damping=1.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            semantic_tags=[("class", "robot")],
        ),
        # InitialStateCfg
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
            # slider
            "slider_basex": 0.0, "slider_basey": 0.0,
            # body
            "idx01_body_joint1": 0.0, "idx02_body_joint2": 0.0, "idx11_head_joint1": 0.0, "idx12_head_joint2": 0.6,
            # right arm
            "idx61_arm_r_joint1": 0.4, "idx62_arm_r_joint2": -1.4, "idx63_arm_r_joint3": -0.2, "idx64_arm_r_joint4": 1.2, "idx65_arm_r_joint5": -2.9, "idx66_arm_r_joint6": 0.0, "idx67_arm_r_joint7": 0.0,
            # right hand
            "idx71_gripper_r_inner_joint1": 0.0, "idx72_gripper_r_inner_joint3": 0.0, "idx73_gripper_r_inner_joint4": 0.0, "idx94_gripper_r_inner_joint0": 0.0,
            "idx81_gripper_r_outer_joint1": 1.0, "idx82_gripper_r_outer_joint3": 0.0, "idx83_gripper_r_outer_joint4": 0.0, "idx93_gripper_r_outer_joint0": 0.0,
            # left arm
            "idx21_arm_l_joint1": 0.0, "idx22_arm_l_joint2": 1.4, "idx23_arm_l_joint3": 0.0, "idx24_arm_l_joint4": 0.0, "idx25_arm_l_joint5": 0.0, "idx26_arm_l_joint6": 0.0, "idx27_arm_l_joint7": 0.0,
            # left hand
            "idx31_gripper_l_inner_joint1": 0.0, "idx32_gripper_l_inner_joint3": 0.0, "idx33_gripper_l_inner_joint4": 0.0, "idx54_gripper_l_inner_joint0": 0.0,
            "idx41_gripper_l_outer_joint1": 0.0, "idx42_gripper_l_outer_joint3": 0.0, "idx43_gripper_l_outer_joint4": 0.0, "idx53_gripper_l_outer_joint0": 0.0,
            },
            pos=(0.0, -1.5, 0.01),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        # ImplicitActuatorCfg
        actuators={
            "head": ImplicitActuatorCfg(
                joint_names_expr=["idx11_head_joint1", "idx12_head_joint2"],
                stiffness=20.,
                damping=10.,
            ),
            "slider": ImplicitActuatorCfg(
                joint_names_expr=["slider_base.*"],
                stiffness=20.,
                damping=10.,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3", "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7"],
                stiffness=50.,
                damping=10.,
            ),
            "right_hand": ImplicitActuatorCfg(
                # joint_names_expr=["idx71_gripper_r_inner_joint1", "idx72_gripper_r_inner_joint3", "idx73_gripper_r_inner_joint4", "idx94_gripper_r_inner_joint0",
                #                   "idx81_gripper_r_outer_joint1", "idx82_gripper_r_outer_joint3", "idx83_gripper_r_outer_joint4", "idx93_gripper_r_outer_joint0"],
                joint_names_expr=["idx81_gripper_r_outer_joint1"],
                stiffness=1.0,
                damping=0.5,
            ),
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3", "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6", "idx27_arm_l_joint7"],
                stiffness=50.,
                damping=10.,
            ),
            "left_hand": ImplicitActuatorCfg(
                # joint_names_expr=["idx31_gripper_l_inner_joint1", "idx32_gripper_l_inner_joint3", "idx33_gripper_l_inner_joint4", "idx54_gripper_l_inner_joint0",
                #                   "idx41_gripper_l_outer_joint1", "idx42_gripper_l_outer_joint3", "idx43_gripper_l_outer_joint4", "idx53_gripper_l_outer_joint0"],
                joint_names_expr=["idx41_gripper_l_outer_joint1"],
                stiffness=1.0,
                damping=0.5,
            ),
        },
    )
    
    
    # record camera
    record_size = 520
    record_step, save_record_step = 1, 1
    record_period = dt_s * decimation * record_step
    record_split, record_exist, record_episode, record_trajectory = 'train', 0, 10, False
    # camera head
    camera_head_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/head_link2/Camera_Head",
        # PinholeCameraCfg
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=36.0,
        ),
        # OffsetCfg
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=(0.7071, 0.7071, 0.0, 0.0),
            convention="world"
        ),
        # Data
        data_types=["rgb", "distance_to_camera", "semantic_segmentation"],
        update_latest_camera_pose=True,
        update_period=record_period,        
        width=record_size, height=record_size,
    )
    # camera arm
    camera_arm_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_r_base_link/Camera_Arm",
        # PinholeCameraCfg
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=36.0,
        ),
        # OffsetCfg
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.05, 0.05),
            rot=(0.7071, 0.0, -0.7071, 0.0),
            convention="world"
        ),
        # Data
        data_types=["rgb", "distance_to_camera", "semantic_segmentation"],
        update_latest_camera_pose=True,
        update_period=record_period,        
        width=record_size, height=record_size,
    )



class G1RobotEnv(UniManipEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()
    
    cfg: G1RobotEnvCfg
    
    def __init__(self, cfg: G1RobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
