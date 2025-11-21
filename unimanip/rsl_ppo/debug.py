import os, sys, runpy

# export LD_LIBRARY_PATH=/home/v-wenbowang/anaconda3/envs/dexgrasp/lib
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__))))


# """
# Debug G1-Robot Cases
# """
# # export CUDA_VISIBLE_DEVICES=0
args = "python train.py --headless --video --video_length 100 --video_interval 100 --num_envs 9 --device cuda:0 --task Isaac-G1-Robot-Direct-v0 --config train_g1_robot_open_best_0.yaml --type partnet --group laptop --index 0"
# args = "python play.py --headless --video --num_envs 1024 --device cuda:0 --task Isaac-G1-Robot-Direct-v0 --config train_g1_robot_open_best_0.yaml --type partnet --group laptop --index 0"
# args = "python play.py --headless --video --video_length 3000 --num_envs 1 --device cuda:0 --task Isaac-G1-Robot-Direct-v0 --config train_g1_robot_open_best_0.yaml --type partnet --group laptop --index 0 --room_index 0 --record_trajectory --record_episode 2 --record_split train"


"""
Debug XHand-Robot Cases
"""
# export CUDA_VISIBLE_DEVICES=0
# args = "python train.py --headless --video --video_length 100 --video_interval 100 --num_envs 9 --device cuda:0 --task Isaac-XHand-Robot-Direct-v0 --config train_xhand_robot_open_best_0.yaml --type partnet --group laptop --index 0"
# args = "python play.py --headless --video --num_envs 9 --device cuda:0 --task Isaac-XHand-Robot-Direct-v0 --config train_xhand_robot_open_best_0.yaml --type partnet --group laptop --index 0"
# args = "python play.py --headless --video --video_length 3000 --num_envs 1 --device cuda:0 --task Isaac-XHand-Robot-Direct-v0 --config train_xhand_robot_open_best_0.yaml --type partnet --group laptop --index 0 --room_index 0 --record_trajectory --record_episode 2 --record_split train"


args = args.split()
if args[0] == 'python':
    args.pop(0)
fun = runpy.run_path
sys.argv.extend(args[1:])
fun(args[0], run_name='__main__')

