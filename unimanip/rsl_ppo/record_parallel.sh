# # Record trained model:
# bash record_parallel.sh 0 9 10 0 15 Isaac-XHand-Robot-Direct-v0 train_xhand_robot_open_best_*.yaml partnet box ppo
# bash record_parallel.sh 0 9 10 0 15 Isaac-G1-Robot-Direct-v0 train_g1_robot_open_best_*.yaml partnet box ppo

# Define Start, Finish Lines, Episode Num, Start, Finish Rooms, Task_Name, Config_Dir, Object_Type, Object_Group, Policy
Start=$1
Finish=$2
Episode=$3
Start_Room=$4
Finish_Room=$5
Task_Name=$6
Config_Dir=$7
Object_Type=$8
Object_Group=$9
Policy=${10}
Device_Number=$(nvidia-smi --list-gpus | wc -l)

# Process Target Objects in Episode
Target_Episode_List=$(seq 1 $Episode)
Target_Room_List=$(seq $Start_Room $Finish_Room)
Batch_Size=$(((Finish - Start + 1) / Episode))

# Record Target Objects with Room
for nroom in $Target_Room_List
do
    # Record Target Objects with Episode
    for nepisode in $Target_Episode_List
    do
        # Update Target_Line_List at Current Episode
        Target_Line_List=$(seq $((Start + Batch_Size*(nepisode-1))) $((Start + Batch_Size*nepisode - 1)))
        
        ncount=0
        # Parallel Record Single Objects
        for nobj in $Target_Line_List
        do
            (
            cuda_id=$((Device_Number - 1 - ncount % Device_Number))
            export CUDA_VISIBLE_DEVICES=$cuda_id
            
            # run ppo policy
            if [ "$Policy" == "ppo" ]; then
                # Test & Record ppo policy in train set
                echo "Test & Record ppo policy: $nobj, train room: $nroom, episode: $nepisode, cuda:$cuda_id, task:$Task_Name, config: $Config_Dir, type: $Object_Type"
                python play.py --headless --video --video_length 6000 --num_envs 1 --device cuda:0 \
                --task $Task_Name --config $Config_Dir --type $Object_Type --group $Object_Group --index $nobj --room_index $nroom \
                --record_trajectory --record_episode 10 --record_split train
                # # Test & Record ppo policy in test set
                # echo "Test & Record dedicated policy: $nobj, test room: $nroom, episode: $nepisode, cuda:$cuda_id, task:$Task_Name, config: $Config_Dir, type: $Object_Type"
                # python play.py --headless --video --video_length 6000 --num_envs 1 --device cuda:0 \
                # --task $Task_Name --config $Config_Dir --type $Object_Type --group $Object_Group --index $nobj --room_index $nroom \
                # --record_trajectory --record_episode 10 --record_split test
            
            # run vla policy
            elif [ "$Policy" == "vla" ]; then
                # Test & Record vla policy in train set
                echo "Test & Record vla policy: $nobj, train room: $nroom, episode: $nepisode, cuda:$cuda_id, task:$Task_Name, config: $Config_Dir, type: $Object_Type"
                python play.py --headless --video --video_length 6000 --num_envs 1 --device cuda:0 \
                --task $Task_Name --config $Config_Dir --type $Object_Type --group $Object_Group --index $nobj --room_index $nroom \
                --record_trajectory --record_episode 10 --record_split test --vla_mode --vla_name mobilemanivla --vla_action -1 --vla_action_frame base
                # # Test & Record vla policy in test set
                # echo "Test & Record dedicated policy: $nobj, test room: $nroom, episode: $nepisode, cuda:$cuda_id, task:$Task_Name, config: $Config_Dir, type: $Object_Type"
                # python play.py --headless --video --video_length 6000 --num_envs 1 --device cuda:0 \
                # --task $Task_Name --config $Config_Dir --type $Object_Type --group $Object_Group --index $nobj --room_index $nroom \
                # --record_trajectory --record_episode 10 --record_split test --vla_mode
            
            fi
            
            ) &
            
            ncount=$((ncount + 1))
        done
        wait
    done
done

