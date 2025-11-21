# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from unimanip.utils.env_model import *



@configclass
class XHandRobotEnvCfg(UniManipEnvCfg):
    """
    Configuration for the XHandRobot Environment based on UniManip Environment, defined with the direct workflow.
    
    XHandRobotEnvCfg defines: sim, robot, camera.
    UniManipEnvCfg defines: scene, object, room, ground, stage, terrain.
    """
    
    # space
    action_space = 6 + 12  # hand pose + hand joint
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
        'arm_joint_stiffness': 100.0,
        'arm_joint_damping': 10.0,
        'hand_joint_stiffness': 1.0,
        'hand_joint_damping': 1.0,
        # action scales
        'action_dof_scale': 1.0,
        'action_pos_scale': 0.025,
        'action_rot_scale': 0.025,
    }
    
    
    # robot_body_names:
    # ['Root', 'basex', 'basey', 'realman_xhand_left_rm_75_6f_description_rm_75_6f_description_base_link', 'realman_xhand_right_rm_75_6f_description_rm_75_6f_description_base_link', 'Left_Arm_Link1', 'Right_Arm_Link1', 'Left_Arm_Link2', 'Right_Arm_Link2', 'Left_Arm_Link3', 'Right_Arm_Link3', 'Left_Arm_Link4', 'Right_Arm_Link4', 'Left_Arm_Link5', 'Right_Arm_Link5', 'Left_Arm_Link6', 'Right_Arm_Link6', 'Left_Arm_Link7', 'Right_Arm_Link7', 
    # 'Left_Cylinder_link', 'Right_Cylinder_link', 'xhand_left_base_link', 'xhand_right_base_link', 'left_hand_index_bend_link', 'left_hand_mid_link1', 'left_hand_pinky_link1', 'left_hand_ring_link1', 'left_hand_thumb_bend_link', 'right_hand_index_bend_link', 'right_hand_mid_link1', 'right_hand_pinky_link1', 'right_hand_ring_link1', 'right_hand_thumb_bend_link', 'left_hand_index_rota_link1', 'left_hand_mid_link2', 'left_hand_pinky_link2', 'left_hand_ring_link2', 'left_hand_thumb_rota_link1', 'right_hand_index_rota_link1', 'right_hand_mid_link2', 'right_hand_pinky_link2', 'right_hand_ring_link2', 'right_hand_thumb_rota_link1', 'left_hand_index_rota_link2', 'left_hand_thumb_rota_link2', 'right_hand_index_rota_link2', 'right_hand_thumb_rota_link2']
    # robot_joint_names:
    # ['slider_basex', 'slider_basey', 'left_arm_joint1', 'right_arm_joint1', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 
    # 'left_hand_index_bend_joint', 'left_hand_mid_joint1', 'left_hand_pinky_joint1', 'left_hand_ring_joint1', 'left_hand_thumb_bend_joint', 'right_hand_index_bend_joint', 'right_hand_mid_joint1', 'right_hand_pinky_joint1', 'right_hand_ring_joint1', 'right_hand_thumb_bend_joint', 'left_hand_index_joint1', 'left_hand_mid_joint2', 'left_hand_pinky_joint2', 'left_hand_ring_joint2', 'left_hand_thumb_rota_joint1', 'right_hand_index_joint1', 'right_hand_mid_joint2', 'right_hand_pinky_joint2', 'right_hand_ring_joint2', 'right_hand_thumb_rota_joint1', 'left_hand_index_joint2', 'left_hand_thumb_rota_joint2', 'right_hand_index_joint2', 'right_hand_thumb_rota_joint2']
    # robot_joint_lower_limit:
    # tensor([-1.0000, -1.0000, -3.1070, -3.1070, -2.2690, -2.2690, -3.1070, -3.1070, -2.3560, -2.3560, -3.1070, -3.1070, -2.2340, -2.2340, -6.2830, -6.2830,
    # -0.1750,  0.0000,  0.0000,  0.0000,  0.0000, -0.1750,  0.0000,  0.0000, 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -1.0500,  0.0000, 0.0000,  0.0000,  0.0000, -1.0500,  0.0000, -0.1700,  0.0000, -0.1750], device='cuda:0')
    # robot_joint_upper_limit:
    # tensor([1.0000, 1.0000, 3.1070, 3.1070, 2.2690, 2.2690, 3.1070, 3.1070, 2.3560, 2.3560, 3.1070, 3.1070, 2.2340, 2.2340, 6.2830, 6.2830, 0.1750, 1.9200,
    # 1.9200, 1.9200, 1.8300, 0.1750, 1.9200, 1.9200, 1.9200, 1.8300, 1.9200, 1.9200, 1.9200, 1.9200, 1.5700, 1.9200, 1.9400, 1.9200, 1.9200, 1.5700, 1.9200, 1.8300, 1.9200, 1.8300], device='cuda:0')
    
    # robot_names
    robot_name = 'xhand_robot'
    robot_base_body_name = 'basey'
    robot_hand_body_name = 'xhand_right_base_link'
    robot_arm_body_names = ['basey', 'realman_xhand_right_rm_75_6f_description_rm_75_6f_description_base_link', 'Right_Arm_Link1', 'Right_Arm_Link2', 'Right_Arm_Link3', 'Right_Arm_Link4', 'Right_Arm_Link5', 'Right_Arm_Link6', 'Right_Arm_Link7']
    robot_hand_body_names = ['xhand_right_base_link', 'right_hand_index_bend_link', 'right_hand_mid_link1', 'right_hand_pinky_link1', 'right_hand_ring_link1', 'right_hand_thumb_bend_link', 'right_hand_index_rota_link1', 'right_hand_mid_link2', 'right_hand_pinky_link2', 'right_hand_ring_link2', 'right_hand_thumb_rota_link1', 'right_hand_index_rota_link2', 'right_hand_thumb_rota_link2']
    robot_finger_body_names = ['right_hand_pinky_link2', 'right_hand_ring_link2', 'right_hand_mid_link2', 'right_hand_index_rota_link2', 'right_hand_thumb_rota_link2']
    robot_arm_joint_names = ['slider_basex', 'slider_basey', 'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6', 'right_arm_joint7']
    robot_hand_joint_names = ['right_hand_index_bend_joint', 'right_hand_mid_joint1', 'right_hand_pinky_joint1', 'right_hand_ring_joint1', 'right_hand_thumb_bend_joint', 'right_hand_index_joint1', 'right_hand_mid_joint2', 'right_hand_pinky_joint2', 'right_hand_ring_joint2', 'right_hand_thumb_rota_joint1', 'right_hand_index_joint2', 'right_hand_thumb_rota_joint2']
    # robot_base_body_index = 2
    # robot_hand_body_index = 22
    # robot_arm_body_indices = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    # robot_hand_body_indices = [22, 28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 45, 46]
    # robot_finger_body_indices = [40, 41, 39, 45, 46]
    # robot_arm_joint_indices = [0, 1, 3, 5, 7, 9, 11, 13, 15]
    # robot_hand_joint_indices = [21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 38, 39]
    
    # robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        # UsdFileCfg
        spawn=sim_utils.UsdFileCfg(
            usd_path=osp.join(ASSET_DIR, 'xhand_robot/xhand_robot.usd'),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=1.0, #linear_damping=1.0, angular_damping=1.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            semantic_tags=[("class", "robot")],
        ),
        # InitialStateCfg
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
            # slider
            "slider_basex": 0.0, "slider_basey": 0.0,
            # right arm
            "right_arm_joint1": 0.0, "right_arm_joint2": -1.5, "right_arm_joint3": -1.3, "right_arm_joint4": 1.0, "right_arm_joint5": 0.0, "right_arm_joint6": 0.5, "right_arm_joint7": 0.0,
            # right xhand
            "right_hand_thumb_bend_joint": 1.5, "right_hand_thumb_rota_joint1": 0.0, "right_hand_thumb_rota_joint2": 0.5,
            "right_hand_index_bend_joint": 0.1, "right_hand_index_joint1": 0.0, "right_hand_index_joint2": 0.5,
            "right_hand_mid_joint1": 0.0, "right_hand_mid_joint2": 0.5,
            "right_hand_ring_joint1": 0.0, "right_hand_ring_joint2": 0.5,
            "right_hand_pinky_joint1": 0.0, "right_hand_pinky_joint2": 0.5,
            # left arm
            "left_arm_joint1": 0.0, "left_arm_joint2": 1.5, "left_arm_joint3": 0.0, "left_arm_joint4": 0.0, "left_arm_joint5": 0.0, "left_arm_joint6": 0.0, "left_arm_joint7": 0.0,
            # left xhand
            "left_hand_thumb_bend_joint": 0.0, "left_hand_thumb_rota_joint1": 1.5, "left_hand_thumb_rota_joint2": 0.0,
            "left_hand_index_bend_joint": 0.0, "left_hand_index_joint1": 0.0, "left_hand_index_joint2": 0.0,
            "left_hand_mid_joint1": 0.0, "left_hand_mid_joint2": 0.0,
            "left_hand_ring_joint1": 0.0, "left_hand_ring_joint2": 0.0,
            "left_hand_pinky_joint1": 0.0, "left_hand_pinky_joint2": 0.0,
            },
            pos=(0.0, -1.25, 0.01),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        # ImplicitActuatorCfg
        actuators={
            "slider": ImplicitActuatorCfg(
                joint_names_expr=["slider_base.*"],
                stiffness=20.,
                damping=10.,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_joint[1-7]"],
                stiffness=50.,
                damping=10.,
            ),
            "right_hand": ImplicitActuatorCfg(
                joint_names_expr=["right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint[1-2]",
                                  "right_hand_index_bend_joint", "right_hand_index_joint[1-2]",
                                  "right_hand_mid_joint[1-2]", "right_hand_ring_joint[1-2]", "right_hand_pinky_joint[1-2]"],
                stiffness=1.0,
                damping=0.5,
            ),
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint[1-7]"],
                stiffness=50.,
                damping=10.,
            ),
            "left_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint[1-2]",
                                  "left_hand_index_bend_joint", "left_hand_index_joint[1-2]",
                                  "left_hand_mid_joint[1-2]", "left_hand_ring_joint[1-2]", "left_hand_pinky_joint[1-2]"],
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
        prim_path="/World/envs/env_.*/Robot/basey/Camera_Head",
        # PinholeCameraCfg
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=36.0,
        ),
        # OffsetCfg
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.1, 1.0),
            rot=(0.6830, -0.1830, 0.1830, 0.6830),  # -30 degrees around x
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
        prim_path="/World/envs/env_.*/Robot/Right_Cylinder_link/Camera_Arm",
        # PinholeCameraCfg
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=36.0,
        ),
        # OffsetCfg
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.1, -0.08),
            rot=(0.0, 0.7071, 0.0, 0.7071),
            convention="world"
        ),
        # Data
        data_types=["rgb", "distance_to_camera", "semantic_segmentation"],
        update_latest_camera_pose=True,
        update_period=record_period,        
        width=record_size, height=record_size,
    )



class XHandRobotEnv(UniManipEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()
    
    cfg: XHandRobotEnvCfg
    
    def __init__(self, cfg: XHandRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
