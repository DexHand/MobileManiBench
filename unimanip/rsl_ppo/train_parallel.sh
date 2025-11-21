# # Train, Test from scratch:
# bash train_parallel.sh 0 9 10 Isaac-XHand-Robot-Direct-v0 train_xhand_robot_open_best_*.yaml partnet box local
# bash train_parallel.sh 0 9 10 Isaac-G1-Robot-Direct-v0 train_g1_robot_open_best_*.yaml partnet box local

# Define Start, Finish Lines, Episode Num, Config_Dir, Object_File
Start=$1
Finish=$2
Episode=$3
Task_Name=$4
Config_Dir=$5
Object_Type=$6
Object_Group=$7
Device_Place=$8
Device_Number=$(nvidia-smi --list-gpus | wc -l)

# Process Target Objects in Episode
Target_Episode_List=$(seq 1 $Episode)
Batch_Size=$(((Finish - Start + 1) / Episode))


# Train Target Objects with Episode
for nepisode in $Target_Episode_List
do
    # Update Target_Line_List at Current Episode
    Target_Line_List=$(seq $((Start + Batch_Size*(nepisode-1))) $((Start + Batch_Size*nepisode - 1)))
    
    ncount=0
    # Parallel Train Single Objects
    for nobj in $Target_Line_List
    do
        (
        cuda_id=$((Device_Number - 1 - ncount % Device_Number))
        export CUDA_VISIBLE_DEVICES=$cuda_id
        
        # running on local device
        if [ "$Device_Place" == "local" ]; then
            echo "Running on local"

            echo "Train dedicated policy: $nobj, episode: $nepisode, cuda:$cuda_id, task:$Task_Name, config: $Config_Dir, type: $Object_Type"
            python train.py --headless --video --video_interval 20000 --num_envs 1024 --device cuda:0 \
            --task $Task_Name --config $Config_Dir --type $Object_Type --group $Object_Group --index $nobj
            
            echo "Test dedicated policy: $nobj, episode: $nepisode, cuda:$cuda_id, task:$Task_Name, config: $Config_Dir, type: $Object_Type"
            python play.py --headless --video --num_envs 1024 --device cuda:0 \
            --task $Task_Name --config $Config_Dir --type $Object_Type --group $Object_Group --index $nobj
        
        # running on server device
        elif [ "$Device_Place" == "server" ]; then
            echo "Running on server"

            echo "Train dedicated policy: $nobj, episode: $nepisode, cuda:$cuda_id, task:$Task_Name, config: $Config_Dir, type: $Object_Type"
            python train.py --headless --num_envs 1024 --device cuda:0 \
            --task $Task_Name --config $Config_Dir --type $Object_Type --group $Object_Group --index $nobj

            echo "Test dedicated policy: $nobj, episode: $nepisode, cuda:$cuda_id, task:$Task_Name, config: $Config_Dir, type: $Object_Type"
            python play.py --headless --num_envs 1024 --device cuda:0 \
            --task $Task_Name --config $Config_Dir --type $Object_Type --group $Object_Group --index $nobj
        
        # invalid device
        else
            echo "Invalid input. Please use 'local' or 'server'."
        fi
        
        ) &
        
        ncount=$((ncount + 1))
    done
    wait
done

