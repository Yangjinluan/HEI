#!/bin/bash
########fb100 penn94
dataset=$1
sub_dataset=${2:-''}

# warm_up_epochs_lst=(0 5 10 20)
# distance_weight_lst=(0.005 0.01 0.03 0.05 0.1 0.3 0.5 1)
# lr_lst=(1e-2 5e-3 1e-3 5e-4)
# weight_decay_lst=(1e-3 5e-3 1e-2)

warm_up_epochs_lst=(10 15 20)
distance_weight_lst=(0.005 0.01 0.03 0.05 0.08 0.1 0.3 0.5 0.8 1)
lr_lst=(1e-2 5e-3 3e-3 1e-3 5e-4 1e-4)
weight_decay_lst=(1e-3 3e-3 5e-3 1e-2 3e-2)
num_layers_lst=(2)
hidden_channels_lst=(256)

for warm_up_epochs in "${warm_up_epochs_lst[@]}"; do
    for distance_weight in "${distance_weight_lst[@]}"; do
        for lr in "${lr_lst[@]}"; do
            for weight_decay in "${weight_decay_lst[@]}"; do
                for num_layers in "${num_layers_lst[@]}"; do
                    for hidden_channels in "${hidden_channels_lst[@]}"; do
                        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                            echo "Running $dataset "
                            python full_main_prototype.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkx  --device "cuda:2" --runs 5 --warm_up_epochs $warm_up_epochs --num_layers $num_layers   --hidden_channels $hidden_channels --distance_weight $distance_weight --lr $lr --weight_decay $weight_decay --display_step 10  --prototype_loss  "global"     --directed
                        else
                            python full_main_prototype.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkx  --device "cuda:2" --runs 5 --warm_up_epochs $warm_up_epochs --num_layers $num_layers   --hidden_channels $hidden_channels --distance_weight $distance_weight --lr $lr --weight_decay $weight_decay --display_step 10  --prototype_loss  "global"  
                        fi
                    done
                done
            done
        done
    done
done

# warm_up_epochs_lst=(5 10 20)
# distance_weight_lst=(0.005 0.01 0.05 0.1 0.3 0.5 1)
# lr_lst=(1e-2 5e-3 1e-3 5e-4)
# weight_decay_lst=(1e-3 5e-3 1e-2)