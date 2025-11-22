# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import cli_args  # isort: skip
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during testing.")
parser.add_argument("--video_length", type=int, default=0, help="Length of the recorded video (in steps).")
parser.add_argument("--record_trajectory", action="store_true", default=False, help="Record trajectory during testing.")
parser.add_argument("--record_episode", type=int, default=10, help="Num of the recorded trajectory.")
parser.add_argument("--record_split", type=str, default='train', help="Split of the recorded room.")

parser.add_argument("--vla_mode", action="store_true", default=False, help="Run with VLA model.")
parser.add_argument("--vla_name", type=str, default='mobilemanivla', help="VLA model name.")
parser.add_argument("--vla_path", type=str, default='train', help="VLA model path.")
parser.add_argument("--vla_action", type=int, default=-1, help="VLA action chunck.")
parser.add_argument("--vla_action_frame", type=str, default='base', help="VLA action frame.")

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--config", type=str, default=None, help="External config for Agent and Env.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")

parser.add_argument("--type", type=str, default=None, help="Type of the object.")
parser.add_argument("--group", type=str, default=None, help="Group of the object.")
parser.add_argument("--index", type=int, default=None, help="Index of the object.")
parser.add_argument("--room_index", type=int, default=0, help="Index of the room.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video: args_cli.enable_cameras = True
if args_cli.record_trajectory: args_cli.video = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from unimanip.utils.config_utils import *
from unimanip.utils.general_utils import *

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx


# load external config
EXTERNAL_CONFIG = load_yaml(osp.join(PROJECT_DIR, 'unimanip/configs/train/{}'.format(args_cli.config)))
# update external config
EXTERNAL_CONFIG['Envs']['object_infos']['type'] = args_cli.type
# YCB Dataset
if EXTERNAL_CONFIG['Envs']['object_infos']['type'] == 'ycb':
    # YCB_ANALYSIS = load_yaml(osp.join(ASSET_DIR, 'ycb/analysis.yaml'))
    EXTERNAL_CONFIG['Envs']['object_infos']['type'] = args_cli.type
    EXTERNAL_CONFIG['Envs']['object_infos']['group'] = args_cli.group
    EXTERNAL_CONFIG['Envs']['object_infos']['index'] = args_cli.index
    EXTERNAL_CONFIG['Envs']['object_infos']['name'] = list(YCB_ANALYSIS[args_cli.group].keys())[args_cli.index]
    EXTERNAL_CONFIG['Envs']['object_infos']['scale'] = 1.0
# PartNet Dataset
if EXTERNAL_CONFIG['Envs']['object_infos']['type'] == 'partnet':
    # PARTNET_ANALYSIS = load_yaml(osp.join(ASSET_DIR, 'partnet/process/{}/analysis.yaml'.format(args_cli.group)))
    EXTERNAL_CONFIG['Envs']['object_infos']['type'] = args_cli.type
    EXTERNAL_CONFIG['Envs']['object_infos']['group'] = args_cli.group
    EXTERNAL_CONFIG['Envs']['object_infos']['index'] = args_cli.index
    EXTERNAL_CONFIG['Envs']['object_infos']['name'] = list(PARTNET_ANALYSIS[args_cli.group].keys())[args_cli.index]
    EXTERNAL_CONFIG['Envs']['object_infos']['scale'] = 1.0
# UniDoor Dataset
if EXTERNAL_CONFIG['Envs']['object_infos']['type'] == 'unidoor':
    # UNIDOOR_ANALYSIS = load_yaml(osp.join(ASSET_DIR, 'unidoor/process/{}/analysis.yaml'.format(args_cli.group)))
    EXTERNAL_CONFIG['Envs']['object_infos']['type'] = args_cli.type
    EXTERNAL_CONFIG['Envs']['object_infos']['group'] = args_cli.group
    EXTERNAL_CONFIG['Envs']['object_infos']['index'] = args_cli.index
    EXTERNAL_CONFIG['Envs']['object_infos']['name'] = list(UNIDOOR_ANALYSIS[args_cli.group].keys())[args_cli.index]
    EXTERNAL_CONFIG['Envs']['object_infos']['scale'] = 1.0


def main():
    
    """================ Init Agent, Env Config ================"""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # override agent_cfg with EXTERNAL_CONFIG
    agent_cfg = update_agent_cfg(agent_cfg, EXTERNAL_CONFIG)
    # override env_cfg with EXTERNAL_CONFIG
    env_cfg = update_env_cfg(env_cfg, EXTERNAL_CONFIG) # type: ignore
    
    # record trajectory with room_id and large env_spacing
    env_cfg.room_id = args_cli.room_index # type: ignore
    env_cfg.record_split = args_cli.record_split # type: ignore
    env_cfg.record_episode = args_cli.record_episode # type: ignore
    env_cfg.record_trajectory = args_cli.record_trajectory # type: ignore
    env_cfg.vla_mode = args_cli.vla_mode # type: ignore
    if env_cfg.vla_mode: env_cfg.action_frame = args_cli.vla_action_frame # type: ignore
    if env_cfg.record_trajectory: env_cfg.scene.env_spacing, env_cfg.viewer.eye = 1e6, [2.0, -2.0, 2.0] # type: ignore
    
    """================ Init Log Folder ================"""
    # specify directory for logging experiments
    log_root_path = osp.abspath(osp.join(LOG_DIR, agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if agent_cfg.run_name: log_root_path = "{}/{}".format(log_root_path, agent_cfg.run_name)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    log_dir = osp.dirname(resume_path)
    
    # update log_dir
    env_cfg.log_dir = osp.join(log_dir, 'videos/play_{:03d}'.format(len(glob.glob(osp.join(log_dir, 'videos/play_*'))))) # type: ignore
    if env_cfg.record_trajectory: env_cfg.log_dir = osp.join(log_dir, 'trajectories/traj_{:03d}'.format(env_cfg.room_id)) # type: ignore
    if env_cfg.vla_mode: env_cfg.log_dir = osp.join(log_dir, 'vla_trajectories_{:03d}/traj_{:03d}'.format(args_cli.vla_action, env_cfg.room_id)) # type: ignore
    os.makedirs(env_cfg.log_dir, exist_ok=True) # type: ignore
    # update test_mode
    env_cfg.test_mode = True # type: ignore
    # default video_length
    if args_cli.video_length == 0: args_cli.video_length = 60 * env_cfg.episode_length_s
    
    # check record_trajectory: remove empty, skip exist
    if env_cfg.record_trajectory: # type: ignore
        # remove empty trajectory episodes
        for episode_dir in sorted(glob.glob(osp.join(env_cfg.log_dir, 'episode_*'))): # type: ignore
            if not osp.exists(osp.join(episode_dir, 'state_infos.pkl')): shutil.rmtree(episode_dir)
        # skip exist episode
        env_cfg.record_exist = len(glob.glob(osp.join(env_cfg.log_dir, 'episode_*'))) # type: ignore
        if env_cfg.record_exist == env_cfg.record_episode: simulation_app.close(); exit() # type: ignore
        # skip failed object
        if not args_cli.vla_mode and osp.exists(resume_path):
            train_success_flag = float(load_list_strings(osp.join(osp.dirname(resume_path), 'log.txt'))[-1].split(';')[1].split(',')[2].split(':')[-1])
            if train_success_flag < 0.1: simulation_app.close(); exit()
    
    """================ Init Isaac Env ================"""
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": env_cfg.log_dir, # type: ignore
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env) # type: ignore
    
    """================ Init VLA/PPO Policy ================"""
    # load vla model from MobileManiVLA
    if args_cli.vla_mode and args_cli.vla_name == 'mobilemanivla':
        from unimanip.utils.adaptive_ensemble import AdaptiveEnsembler
        from robovlms.model.vla_builder import build_vla, load_vla_checkpoint
        # load vla config
        args_cli.vla_path = osp.join(LOG_DIR, '')
        # vla_config = load_json(osp.join(PROJECT_DIR, 'unimanip/configs/train/train_vla.json'))
        vla_config = load_json(osp.join(osp.dirname(osp.dirname(osp.dirname(args_cli.vla_path))), 'config.json'))
        vla_config['model_path']= vla_config['model_path'].replace('/mnt/blob/Desktop/Assets', ASSET_DIR)
        vla_config['model_config'] = vla_config['model_config'].replace('/mnt/blob/Desktop/Assets', ASSET_DIR)
        vla_config['tokenizer']['pretrained_model_name_or_path'] = vla_config['tokenizer']['pretrained_model_name_or_path'].replace('/mnt/blob/Desktop/Assets', ASSET_DIR)
        vla_config['vlm']['pretrained_model_name_or_path'] = vla_config['vlm']['pretrained_model_name_or_path'].replace('/mnt/blob/Desktop/Assets', ASSET_DIR)
        
        # load vla model
        vla_model = build_vla(configs=vla_config, precision=vla_config["trainer"]["precision"])
        vla_model = load_vla_checkpoint(vla_model, args_cli.vla_path)
        vla_model.model.use_bf16 = vla_config["use_bf16"]
        vla_model.use_bf16 = vla_config["use_bf16"]
        vla_model.to(args_cli.device).eval()
        # load vla action ensembler
        vla_action_ensembler = AdaptiveEnsembler(pred_action_horizon=4, adaptive_ensemble_alpha=0.1)
    # load ppo model
    else:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device) # type: ignore
        ppo_runner.load(resume_path)
        # obtain the trained ppo model for inference
        ppo_model = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        # export ppo model to onnx/jit
        export_model_dir = osp.join(osp.dirname(resume_path), "exported")
        export_policy_as_jit(ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")
    
    """================ Start RSL_RL Runner ================"""
    # reset environment
    obs, extras = env.get_observations()
    dones, timestep = torch.zeros(args_cli.num_envs), 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # VLA stepping for DiffusionVLA
            if args_cli.vla_mode and args_cli.vla_name == 'mobilemanivla':
                # unpack trajectory info
                obj_action, obj_type, obj_group, obj_name = env_cfg.action_type, EXTERNAL_CONFIG['Envs']['object_infos']['type'], EXTERNAL_CONFIG['Envs']['object_infos']['group'], EXTERNAL_CONFIG['Envs']['object_infos']['name'] # type: ignore
                # load vla_prompt
                vla_prompt = load_object_prompt(obj_action.lower(), obj_type, obj_group, obj_name)
                # infer rgb_image_head and rgb_image_arm
                if len(extras["observations"]['render']['rgb_image_head']) > 0:
                    # load rgb_image_head and rgb_image_arm
                    vla_image_head = Image.fromarray(extras["observations"]['render']['rgb_image_head'][-1])
                    vla_image_arm = Image.fromarray(extras["observations"]['render']['rgb_image_arm'][-1])
                    # load depth_image_head and depth_image_arm
                    vla_depth_head = Image.fromarray(extras["observations"]['render']['depth_image_head'][-1])
                    vla_depth_arm = Image.fromarray(extras["observations"]['render']['depth_image_arm'][-1])
                    vla_images = [vla_image_head, vla_depth_head, vla_image_arm, vla_depth_arm]
                    
                    # load state
                    base_pose = extras["observations"]['trajectory']['robot_base'][-1].clone()
                    hand_pose = extras["observations"]['trajectory']['robot_hand'][-1].clone()
                    hand_pose[:, :3], hand_pose[:, 3:6] = encode_world_pose_to_local_frame(
                        hand_pose[:, :3], hand_pose[:, 3:6], base_pose[:, :3], base_pose[:, 3:6])
                    vla_state = hand_pose[:, :6]
                    vla_state_mask = torch.tensor(np.asarray([True]), dtype=torch.bool).repeat(1, vla_state.shape[-1]).to(args_cli.device)
                    
                    # infer vla_actions
                    vla_actions = vla_model.predict_action(vla_images, vla_prompt, vla_state, vla_state_mask)
                    # extract actions via action_chunck
                    if args_cli.vla_action >= 0:
                        actions = torch.tensor(vla_actions[args_cli.vla_action]).to(args_cli.device).unsqueeze(0)
                    # extract actions via action_ensembler
                    else:
                        if torch.sum(dones) != 0: vla_action_ensembler.reset()
                        actions = torch.tensor(vla_action_ensembler.ensemble_action(vla_actions)).to(args_cli.device).unsqueeze(0)
                # zero actions at first frame
                else: actions = torch.zeros((args_cli.num_envs, env_cfg.action_space)).to(args_cli.device) # type: ignore
            # PPO stepping
            else: actions = ppo_model(obs)
            # env stepping
            obs, rew, dones, extras = env.step(actions)
        # Exit the play loop
        if env.cfg.exit_flag: break # type: ignore
        # Exit the play loop after recording one video
        if args_cli.video or args_cli.video_length > 0:
            timestep += 1
            if timestep == args_cli.video_length: break
    
    # check record_trajectory: remove empty
    if env_cfg.record_trajectory: # type: ignore
        # remove empty trajectory episodes
        for episode_dir in sorted(glob.glob(osp.join(env_cfg.log_dir, 'episode_*'))): # type: ignore
            if not osp.exists(osp.join(episode_dir, 'state_infos.pkl')): shutil.rmtree(episode_dir)
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
