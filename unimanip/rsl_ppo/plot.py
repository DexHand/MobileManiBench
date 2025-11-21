import os
import glob
import shutil
import argparse
import cv2 as cv
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from unimanip.utils.general_utils import *


# read train log.txt
def read_train_log(log_fn):
    # init train_log_dict
    train_log_dict = {'nepoch': [], 'success': [], 'success_flag': [], 'success_ratio': []}
    # load log_file lines
    log_lines = load_list_strings(log_fn)
    # process log_lines
    for nline in range(len(log_lines)):
        # split line_success
        line_success = log_lines[nline].split(';')[1].split(',')
        # locate nepoch, success, success_flag
        nepoch = nline
        success = float(line_success[1].split(':')[-1])
        success_flag = float(line_success[2].split(':')[-1])
        success_ratio = float(line_success[3].split(':')[-1])
        # append nepoch, success, success_flag
        train_log_dict['nepoch'].append(nepoch)
        train_log_dict['success'].append(success)
        train_log_dict['success_flag'].append(success_flag)
        train_log_dict['success_ratio'].append(success_ratio)
    return train_log_dict

# read train_result_dict
def read_train_result(train_result_dir):
    # load train_result_lines
    train_result_lines = load_list_strings(train_result_dir)
    # init train_result_dict
    train_result_dict = {'object': [], 'success': [], 'success_flag': [], 'success_ratio': [], 'infos': []}
    # process all lines
    for line in train_result_lines[:-2]:
        train_result_dict['infos'].append(line)
        train_result_dict['object'].append(line.replace(' ', '').split(',')[0])
        train_result_dict['success'].append(float(line.replace(' ', '').split(',')[1]))
        train_result_dict['success_flag'].append(float(line.replace(' ', '').split(',')[2]))
        train_result_dict['success_ratio'].append(float(line.replace(' ', '').split(',')[3]))
    train_result_dict['infos'].append(train_result_lines[-2][:-1])
    train_result_dict['infos'].append(train_result_lines[-1][:-1])
    return train_result_dict

# plot train_result_dict
def plot_train_result(save_dir, train_result_dict):
    # Sort success_rate
    x = np.arange(len(train_result_dict['object']))
    sort_values = np.argsort(np.asarray(train_result_dict['success_flag']))
    sort_values = np.asarray(train_result_dict['success_flag'])[list(sort_values)]
    # Create Figure with Bars
    plt.figure(figsize=(15, 5))
    plt.bar(x, sort_values, width=1)
    # Labeling the axes
    plt.xlabel('Objects', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.title('{}: {:.3f} / {}'.format(train_result_dict['infos'][-1], np.mean(train_result_dict['success_flag']), len(train_result_dict['success_flag'])), fontsize=16)
    # Save the chart
    plt.savefig(save_dir, format='png')


# python plot.py --container --type partnet --group table --config train_xhand_robot_control_best.yaml


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script with argparse')
    parser.add_argument('--name', type=str, default='train_0', help='Train Name')
    parser.add_argument("--check", type=str, default='model_3999.pt', help="Check Name")
    
    parser.add_argument("--type", type=str, default=None, help="Object Type")
    parser.add_argument("--group", type=str, default=None, help="Object Group")
    parser.add_argument("--start", type=int, default=None, help="Object Start Index")
    parser.add_argument("--finish", type=int, default=None, help="Object Finish Index")
    parser.add_argument('--config', type=str, default=None, help='Training Config File')
    
    parser.add_argument('--container', action="store_true", default=False, help='Container')
    parser.add_argument('--train', action="store_true", default=False, help='Plot Train')
    args = parser.parse_args()
    
    # locate config
    config = load_yaml(osp.join(PROJECT_DIR, 'unimanip/configs/train', args.config))
    # locate log_dir
    log_dir = osp.join(LOG_DIR, config['Infos']['name'], config['Agents']['name'], args.type, args.group)
    if args.container: log_dir = osp.join('/data0/v-wenbowang/Desktop/Container/Desktop/Logs', config['Infos']['name'], config['Agents']['name'], args.type, args.group)
    
    # locate object_analysis
    if args.type == 'ycb': object_analysis = YCB_ANALYSIS['ycb']
    if args.type == 'partnet': object_analysis = PARTNET_ANALYSIS[args.group]
    if args.type == 'unidoor': object_analysis = UNIDOOR_ANALYSIS[args.group]
    # locate all object_dirs
    object_dirs = sorted(glob.glob(osp.join(log_dir, '0*')))
    if args.start is not None and args.finish is not None:
        object_dirs = [object_dir for object_dir in object_dirs if int(object_dir.split('/')[-1]) >= args.start and int(object_dir.split('/')[-1]) <= args.finish]
    if len(object_dirs) == 0: exit()
    
    # init train_result_dict
    train_result_dict = {'object': [], 'success': [], 'success_flag': [], 'success_ratio': [], 'infos': []}
    
    # process all object_dirs
    for object_dir in object_dirs:
        # locate object_name
        object_id = object_dir.split('/')[-1]
        object_name = list(object_analysis.keys())[int(object_dir.split('/')[-1])].replace('/', '-')
        if args.train: object_log_fn = osp.join(object_dir, object_name, args.name, 'log.txt')
        else: object_log_fn = osp.join(object_dir, object_name, args.name, 'videos/play_000/log.txt')
        object_check_fn = osp.join(object_dir, object_name, args.name, args.check)
        if not osp.exists(object_check_fn) or not osp.exists(object_log_fn): continue
        
        # process object_train_results
        train_log_dict = read_train_log(object_log_fn)
        print(object_id, object_name)
        print('success: {}, success_flag: {}'.format(train_log_dict['success'][-1], train_log_dict['success_flag'][-1]))
        # append train_result_dict
        train_result_dict['object'].append(object_id)
        train_result_dict['success'].append(train_log_dict['success'][-1])
        train_result_dict['success_flag'].append(train_log_dict['success_flag'][-1])
        train_result_dict['success_ratio'].append(train_log_dict['success_ratio'][-1])
        train_result_dict['infos'].append('{}, {:.3f}, {:.3f}, {:.3f}, {},'.format(object_id, train_result_dict['success'][-1], train_result_dict['success_flag'][-1], train_result_dict['success_ratio'][-1], object_name))
    if len(train_result_dict['object']) == 0: exit()
    # process mean success and mean_success_flag
    train_result_dict['infos'].append('number: {}, mean_success: {:.3f}, mean_success_flag: {:.3f}, mean_success_ratio: {:.3f}'.format(len(train_result_dict['object']), np.mean(train_result_dict['success']), np.mean(train_result_dict['success_flag']), np.mean(train_result_dict['success_ratio'])))
    train_result_dict['infos'].append('{}_{}'.format(config['Agents']['name'], args.group))
    print(train_result_dict['infos'][-2])
    # save train_result_dict_infos
    save_list_strings(osp.join(log_dir, 'success_{}.txt'.format('train' if args.train else 'test')), train_result_dict['infos'])
    save_list_strings(osp.join(PROJECT_DIR, 'unimanip/configs/test', 'success_{}_{}_'.format('train' if args.train else 'test', args.group) + args.config.replace('.yaml', '.txt')), train_result_dict['infos'])
    # plot train_result_dict
    plot_train_result(osp.join(PROJECT_DIR, 'unimanip/configs/test', 'plot_{}_{}_'.format('train' if args.train else 'test', args.group) + args.config.replace('.yaml', '.png')), train_result_dict)