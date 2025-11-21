# Plot train results
# bash plot_parallel.sh 0 0 partnet train_xhand_robot_open
# bash plot_parallel.sh 0 0 partnet train_xhand_robot_open table
# bash plot_parallel.sh 0 0 unidoor train_xhand_robot_open
# bash plot_parallel.sh 0 0 unidoor train_xhand_robot_open car

Start=$1
Finish=$2
Object_Type=$3
Config_Base_Name=$4
Object_Group=$5

Target_Line_List=$(seq $((Start)) $((Finish)))

if [ "$Object_Type" == "partnet" ]; then
    Object_Group_List=("box" "cart" "dishwasher" "faucet" "laptop" "microwave" "oven" "refrigerator" "table" "toilet" "trashcan" "washingmachine")
elif [ "$Object_Type" == "unidoor" ]; then
    Object_Group_List=("cabinet" "car" "fridge" "lever_door" "round_door" "window" "safe")
elif [ "$Object_Type" == "ycb" ]; then
    Object_Group_List=("ycb")
fi


if [ -z "$Object_Group" ]; then
    # Parallel Plot Group Objects
    for nline in $Target_Line_List
    do
        for Object_Group in "${Object_Group_List[@]}"; do
        (
            echo "Plot" ${Config_Base_Name}_${nline}.yaml for ${Object_Type} ${Object_Group}
            python plot.py --container --type $Object_Type --group $Object_Group --config ${Config_Base_Name}_${nline}.yaml
        ) &
        done
        wait
    done
else
    # Parallel Plot Single Object
    for nline in $Target_Line_List
    do
        (
            echo "Plot" ${Config_Base_Name}_${nline}.yaml
            python plot.py --container --type $Object_Type --group $Object_Group --config ${Config_Base_Name}_${nline}.yaml
        ) &
    done
    wait
fi
