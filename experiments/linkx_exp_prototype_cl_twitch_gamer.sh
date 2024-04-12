#!/bin/bash
# twitch-gamer
dataset=$1
sub_dataset=${2:-''}
warm_up_epochs_lst=(0 5 10)
temperature_lst=(0.2 0.4 0.6 1 2 3 5)
cl_weight_lst=(0.1 0.3 0.5 1 3 5)
lr_lst=(1e-2 5e-3 1e-3 5e-4)
weight_decay_lst=(1e-3 1e-2)

num_layers_lst=(1)
hidden_channels_lst=(64)

for warm_up_epochs in "${warm_up_epochs_lst[@]}"; do
    for cl_weight in "${cl_weight_lst[@]}"; do
        for temperature in "${temperature_lst[@]}"; do
            for lr in "${lr_lst[@]}"; do
                for weight_decay in "${weight_decay_lst[@]}"; do
                    for num_layers in "${num_layers_lst[@]}"; do
                        for hidden_channels in "${hidden_channels_lst[@]}"; do
                            if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                                echo "Running $dataset "
                                python full_main_prototype.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkx  --device "cuda:1" --runs 5 --warm_up_epochs $warm_up_epochs --num_layers $num_layers   --hidden_channels $hidden_channels --temperature $temperature  --cl_weight $cl_weight --lr $lr --weight_decay $weight_decay --display_step 10 --prototype_loss "cl" --directed
                            else
                                python full_main_prototype.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkx  --device "cuda:1" --runs 5 --warm_up_epochs $warm_up_epochs --num_layers $num_layers   --hidden_channels $hidden_channels --temperature $temperature  --cl_weight $cl_weight --lr $lr --weight_decay $weight_decay --display_step 10  --prototype_loss "cl"
                            fi
                        done
                    done
                done
            done
        done
    done
done