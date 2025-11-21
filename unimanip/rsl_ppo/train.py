# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import sys
import argparse
import cli_args  # isort: skip
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=8000, help="Interval between video recordings (in steps).")

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--config", type=str, default=None, help="External config for Agent and Env.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

parser.add_argument("--type", type=str, default=None, help="Type of the object.")
parser.add_argument("--group", type=str, default=None, help="Group of the object.")
parser.add_argument("--index", type=int, default=None, help="Index of the object.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video: args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import gymnasium as gym
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

from unimanip.utils.config_utils import *
from unimanip.utils.general_utils import *

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    
    """================ Init Agent, Env Config ================"""
    # override agent_cfg with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    # override agent_cfg with EXTERNAL_CONFIG
    agent_cfg = update_agent_cfg(agent_cfg, EXTERNAL_CONFIG)
    
    # override env_cfg with non-hydra CLI arguments
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # override env_cfg with EXTERNAL_CONFIG
    env_cfg = update_env_cfg(env_cfg, EXTERNAL_CONFIG)
    
    """================ Init Log Folder ================"""
    # specify directory for logging experiments
    log_root_path = osp.abspath(osp.join(LOG_DIR, agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify log_dir
    log_dir = 'train_0'
    if agent_cfg.run_name: log_dir = "{}/{}".format(agent_cfg.run_name, log_dir)
    log_dir = osp.join(log_root_path, log_dir)
    # check train_exist
    for check_dir in sorted(glob.glob(osp.join(osp.dirname(log_dir), 'train_*'))):
        if osp.exists(osp.join(check_dir, 'model_{}.pt'.format(agent_cfg.max_iterations - 1))): simulation_app.close(); exit()
        else: shutil.rmtree(check_dir)
    # specify train_number
    log_dir = osp.join(osp.dirname(log_dir), 'train_{}'.format(len(glob.glob(osp.join(osp.dirname(log_dir), 'train_*')))))
    
    # update log_dir
    env_cfg.log_dir = log_dir # type: ignore
    os.makedirs(env_cfg.log_dir, exist_ok=True) # type: ignore
    # update test_mode
    env_cfg.test_mode = False # type: ignore
    # # update video_length
    # args_cli.video_length = 60 * env_cfg.episode_length_s
    
    # train: dump the configuration into log-directory
    dump_yaml(osp.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(osp.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    """================ Init Isaac Env ================"""
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": osp.join(log_dir, 'videos/train_{}'.format(len(glob.glob(osp.join(log_dir, 'videos/train_*'))))), # type: ignore
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env) # type: ignore
    
    """================ Init RSL_RL Runner ================"""
    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device) # type: ignore
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    
    """================ Start RSL_RL Runner ================"""
    # load the checkpoint
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main() # type: ignore
    # close sim app
    simulation_app.close()
