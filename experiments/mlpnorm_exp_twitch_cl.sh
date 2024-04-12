#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

dataset=twitch-gamer
sub_dataset=${2:-''}

warm_up_epochs_lst=(0 5 10 20)
cl_weight_lst=(0.1 0.3 0.5 1)
temperature_lst=(1 2 3 5)
lr_lst=(1e-3 5e-3 1e-2)
weight_decay_lst=(1e-1 1e-2)

hidden_channels_lst=(256)
dropout_lst=(0.8)
alpha_lst=(1)
beta_lst=(0.1)
gamma_lst=(0.1)
delta_lst=(0.2)
norm_layers_lst=(1)
orders_lst=(1)
epochs=500
runs=5
norm_func_id=2
order_func_id=2

for warm_up_epochs in "${warm_up_epochs_lst[@]}"; do
    for cl_weight in "${cl_weight_lst[@]}"; do
        for temperature in "${temperature_lst[@]}"; do
            for lr in "${lr_lst[@]}"; do
                for weight_decay in "${weight_decay_lst[@]}"; do
                    for hidden_channels in "${hidden_channels_lst[@]}"; do
                        for beta in "${beta_lst[@]}"; do
                            for gamma in "${gamma_lst[@]}"; do
                                for norm_layers in "${norm_layers_lst[@]}"; do
                                    for dropout in "${dropout_lst[@]}"; do
                                        for orders in "${orders_lst[@]}"; do
                                            for alpha in "${alpha_lst[@]}"; do
                                                for delta in "${delta_lst[@]}"; do
                                                    if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                                                        python -u main_glo.py --device "cuda:0" --dataset $dataset --sub_dataset ${sub_dataset:-''} --method mlpnorm --epochs $epochs --temperature $temperature --warm_up_epochs $warm_up_epochs  --cl_weight $cl_weight --prototype_loss  "cl"   --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --delta $delta --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 1 --runs $runs --directed
                                                    else
                                                        python -u main_glo.py --device "cuda:0" --dataset $dataset --sub_dataset ${sub_dataset:-''} --method mlpnorm --epochs $epochs --temperature $temperature --warm_up_epochs $warm_up_epochs --cl_weight $cl_weight --prototype_loss  "cl"  --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --delta $delta --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 1 --runs $runs
                                                    fi
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done 
        done
    done
done


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 