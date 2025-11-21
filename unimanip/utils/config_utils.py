from .general_utils import *

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

# update agent_cfg with external_cfg
def update_agent_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, external_cfg):
    # name
    agent_cfg.experiment_name, agent_cfg.run_name = external_cfg['Infos']['name'], external_cfg['Agents']['name']
    # object type and name
    if external_cfg['Envs']['object_infos']['type'] == 'ycb':
        agent_cfg.run_name = '{}/ycb/ycb/{:04d}/{}'.format(
            agent_cfg.run_name, external_cfg['Envs']['object_infos']['index'], external_cfg['Envs']['object_infos']['name'])
    if external_cfg['Envs']['object_infos']['type'] == 'unidexgrasp':
        agent_cfg.run_name = '{}/unidexgrasp/{:04d}/{}/{}'.format(
            agent_cfg.run_name, external_cfg['Envs']['object_infos']['index'], external_cfg['Envs']['object_infos']['name'], SCALE2STR[external_cfg['Envs']['object_infos']['scale']])
    if external_cfg['Envs']['object_infos']['type'] == 'partnet':
        agent_cfg.run_name = '{}/partnet/{}/{:04d}/{}'.format(
            agent_cfg.run_name, external_cfg['Envs']['object_infos']['group'], external_cfg['Envs']['object_infos']['index'], external_cfg['Envs']['object_infos']['name'].replace('/', '-'))
    if external_cfg['Envs']['object_infos']['type'] == 'unidoor':
        agent_cfg.run_name = '{}/unidoor/{}/{:04d}/{}'.format(
            agent_cfg.run_name, external_cfg['Envs']['object_infos']['group'], external_cfg['Envs']['object_infos']['index'], external_cfg['Envs']['object_infos']['name'].replace('/', '-'))
    # run
    agent_cfg.empirical_normalization = external_cfg['Agents']['empirical_normalization']
    agent_cfg.num_steps_per_env = external_cfg['Agents']['num_steps_per_env']
    agent_cfg.max_iterations = external_cfg['Agents']['max_iterations']
    agent_cfg.save_interval = external_cfg['Agents']['save_interval']
    # policy
    agent_cfg.policy.actor_hidden_dims = external_cfg['Agents']['policy']['actor_hidden_dims'] # type: ignore
    agent_cfg.policy.critic_hidden_dims = external_cfg['Agents']['policy']['critic_hidden_dims'] # type: ignore
    agent_cfg.policy.init_noise_std = external_cfg['Agents']['policy']['init_noise_std']
    agent_cfg.policy.activation = external_cfg['Agents']['policy']['activation']
    # algorithm
    agent_cfg.algorithm.learning_rate = external_cfg['Agents']['algorithm']['learning_rate']
    agent_cfg.algorithm.num_learning_epochs = external_cfg['Agents']['algorithm']['num_learning_epochs']
    agent_cfg.algorithm.num_mini_batches = external_cfg['Agents']['algorithm']['num_mini_batches'] # type: ignore
    agent_cfg.algorithm.schedule = external_cfg['Agents']['algorithm']['schedule'] # type: ignore
    return agent_cfg

# update env_cfg with external_cfg
def update_env_cfg(env_cfg: DirectRLEnvCfg, external_cfg):
    # sim
    env_cfg.dt_s = 1 / external_cfg['Envs']['dt_s'] # type: ignore
    env_cfg.decimation = external_cfg['Envs']['decimation'] # type: ignore
    env_cfg.sim.dt = env_cfg.dt_s # type: ignore
    env_cfg.sim.render_interval = env_cfg.decimation # type: ignore
    # space
    env_cfg.action_space = external_cfg['Envs']['action_space']
    env_cfg.observation_space = external_cfg['Envs']['observation_space']
    # run
    env_cfg.action_type = external_cfg['Envs']['action_type'] # type: ignore
    env_cfg.episode_length_s = external_cfg['Envs']['episode_length_s']
    # viewer
    env_cfg.viewer.eye = external_cfg['Envs']['viewer']['eye']
    env_cfg.viewer.lookat = external_cfg['Envs']['viewer']['lookat']
    env_cfg.viewer.resolution = external_cfg['Envs']['viewer']['resolution']
    # object
    if external_cfg['Envs']['object_infos']['type'] == 'ycb':
        # update object_infos
        env_cfg.object_infos = external_cfg['Envs']['object_infos'] # type: ignore
        # locate object usd path
        env_cfg.object_usd_path = osp.join(ASSET_DIR, 'ycb/{}/google_16k/textured.usd'.format(external_cfg['Envs']['object_infos']['name'])) # type: ignore
        env_cfg.object_cfg.spawn.usd_path = env_cfg.object_usd_path # type: ignore
        # locate object mesh_path, point_path, init_path
        env_cfg.object_mesh_path = osp.join(ASSET_DIR, 'ycb/{}/google_16k/textured.obj'.format(external_cfg['Envs']['object_infos']['name'])) # type: ignore
    if external_cfg['Envs']['object_infos']['type'] == 'unidexgrasp':
        # update object_infos
        env_cfg.object_infos = external_cfg['Envs']['object_infos'] # type: ignore
        # locate object usd path
        env_cfg.object_usd_path = osp.join(ASSET_DIR, 'unidexgrasp/meshdatav3_scaled/{}/coacd/decomposed_{}.usd'.format( # type: ignore
            external_cfg['Envs']['object_infos']['name'], SCALE2STR[external_cfg['Envs']['object_infos']['scale']]))
        env_cfg.object_cfg.spawn.usd_path = env_cfg.object_usd_path # type: ignore
        # locate object mesh_path, point_path, init_path
        env_cfg.object_mesh_path = osp.join(ASSET_DIR, 'unidexgrasp/meshdatav3_scaled/{}/coacd/decomposed_{}.obj'.format( # type: ignore
            external_cfg['Envs']['object_infos']['name'], SCALE2STR[external_cfg['Envs']['object_infos']['scale']])) 
        env_cfg.object_point_path = osp.join(ASSET_DIR, 'unidexgrasp/meshdatav3_pc_fps/{}/object_pc_{}.pkl'.format( # type: ignore
            external_cfg['Envs']['object_infos']['name'], SCALE2STR[external_cfg['Envs']['object_infos']['scale']]))
        env_cfg.object_init_path = osp.join(ASSET_DIR, 'unidexgrasp/meshdatav3_init/{}/object_init_{}.pkl'.format( # type: ignore
            external_cfg['Envs']['object_infos']['name'], SCALE2STR[external_cfg['Envs']['object_infos']['scale']]))
    if external_cfg['Envs']['object_infos']['type'] == 'partnet':
        # update object_infos
        env_cfg.object_infos = external_cfg['Envs']['object_infos'] # type: ignore
        # locate object usd path
        env_cfg.object_usd_path = osp.join(ASSET_DIR, 'partnet/process/{}/{}/mobility.usd'.format( # type: ignore
            external_cfg['Envs']['object_infos']['group'], external_cfg['Envs']['object_infos']['name'].split('/')[0]))
        env_cfg.object_cfg.spawn.usd_path = env_cfg.object_usd_path # type: ignore
    if external_cfg['Envs']['object_infos']['type'] == 'unidoor':
        # update object_infos
        env_cfg.object_infos = external_cfg['Envs']['object_infos'] # type: ignore
        # locate object usd path
        env_cfg.object_usd_path = osp.join(ASSET_DIR, 'unidoor/process/{}/{}/mobility.usd'.format( # type: ignore
            external_cfg['Envs']['object_infos']['group'], external_cfg['Envs']['object_infos']['name'].split('/')[0]))
        env_cfg.object_cfg.spawn.usd_path = env_cfg.object_usd_path # type: ignore
    # control_params
    env_cfg.control_params = external_cfg['Envs']['control_params'] # type: ignore
    
    # robot body damping
    if 'linear_damping' in external_cfg['Envs']['control_params']: 
        env_cfg.robot_cfg.spawn.rigid_props.linear_damping = external_cfg['Envs']['control_params']['linear_damping'] # type: ignore
    if 'angular_damping' in external_cfg['Envs']['control_params']: 
        env_cfg.robot_cfg.spawn.rigid_props.angular_damping = external_cfg['Envs']['control_params']['angular_damping'] # type: ignore
    # robot body max_depenetration_velocity
    if 'max_depenetration_velocity' in external_cfg['Envs']['control_params'] and external_cfg['Envs']['control_params']['max_depenetration_velocity'] > 0:
        env_cfg.robot_cfg.spawn.rigid_props.max_depenetration_velocity = external_cfg['Envs']['control_params']['max_depenetration_velocity'] # type: ignore
        env_cfg.object_cfg.spawn.rigid_props.max_depenetration_velocity = external_cfg['Envs']['control_params']['max_depenetration_velocity'] # type: ignore
    
    # slider joint
    if 'slider' in env_cfg.robot_cfg.actuators and 'slider_joint_stiffness' in external_cfg['Envs']['control_params']: # type: ignore
        env_cfg.robot_cfg.actuators['slider'].stiffness = external_cfg['Envs']['control_params']['slider_joint_stiffness'] # type: ignore
        env_cfg.robot_cfg.actuators['slider'].damping = external_cfg['Envs']['control_params']['slider_joint_damping'] # type: ignore
    
    # arm joint
    if 'right_arm' in env_cfg.robot_cfg.actuators and 'arm_joint_stiffness' in external_cfg['Envs']['control_params']: # type: ignore
        env_cfg.robot_cfg.actuators['right_arm'].stiffness = external_cfg['Envs']['control_params']['arm_joint_stiffness'] # type: ignore
        env_cfg.robot_cfg.actuators['right_arm'].damping = external_cfg['Envs']['control_params']['arm_joint_damping'] # type: ignore
    if 'left_arm' in env_cfg.robot_cfg.actuators and 'arm_joint_stiffness' in external_cfg['Envs']['control_params']: # type: ignore
        env_cfg.robot_cfg.actuators['left_arm'].stiffness = external_cfg['Envs']['control_params']['arm_joint_stiffness'] # type: ignore
        env_cfg.robot_cfg.actuators['left_arm'].damping = external_cfg['Envs']['control_params']['arm_joint_damping'] # type: ignore
    
    # hand joint
    if 'right_hand' in env_cfg.robot_cfg.actuators and 'hand_joint_stiffness' in external_cfg['Envs']['control_params']: # type: ignore
        env_cfg.robot_cfg.actuators['right_hand'].stiffness = external_cfg['Envs']['control_params']['hand_joint_stiffness'] # type: ignore
        env_cfg.robot_cfg.actuators['right_hand'].damping = external_cfg['Envs']['control_params']['hand_joint_damping'] # type: ignore
    if 'left_hand' in env_cfg.robot_cfg.actuators and 'hand_joint_stiffness' in external_cfg['Envs']['control_params']: # type: ignore
        env_cfg.robot_cfg.actuators['left_hand'].stiffness = external_cfg['Envs']['control_params']['hand_joint_stiffness'] # type: ignore
        env_cfg.robot_cfg.actuators['left_hand'].damping = external_cfg['Envs']['control_params']['hand_joint_damping'] # type: ignore
    
    # reward_params
    env_cfg.reward_params = external_cfg['Envs']['reward_params'] # type: ignore
    return env_cfg
