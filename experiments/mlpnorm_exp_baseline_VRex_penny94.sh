#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

penalty_weight_lst=(1e-3 1e-2 0.1 1 10 100 1000)
penalty_anneal_iters_lst=(0 10 30 50)
l2_regularizer_weight_lst=(1e-3 1e-2)
split_lst=(2 4 8)
lr_lst=(1e-2 5e-3 1e-3)
weight_decay_lst=(1e-2 1e-3)


dataset=fb100
sub_dataset=Penn94
hidden_channels_lst=(256)
dropout_lst=(0.5)
alpha_lst=(0.0)
beta_lst=(1.0)
gamma_lst=(0.5)
delta_lst=(0.5)
norm_layers_lst=(2)
orders_lst=(2)
epochs=500
runs=5
norm_func_id=2
order_func_id=2

for split in "${split_lst[@]}"; do
    for penalty_anneal_iters in "${penalty_anneal_iters_lst[@]}"; do
        for penalty_weight in "${penalty_weight_lst[@]}"; do
            for l2_regularizer_weight in "${l2_regularizer_weight_lst[@]}"; do
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
                                                            echo "Running $dataset "
                                                            python -u main_base_other_two.py --device "cuda:0" --dataset $dataset --sub_dataset ${sub_dataset:-''} --runs 5 --method mlpnorm --epochs $epochs   --baseline "V-Rex" --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --delta $delta --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 10   --split $split    --penalty_anneal_iters $penalty_anneal_iters  --penalty_weight $penalty_weight   --l2_regularizer_weight $l2_regularizer_weight  --directed
                                                        else
                                                            python -u main_base_other_two.py --device "cuda:0" --dataset $dataset --sub_dataset ${sub_dataset:-''} --runs 5 --method mlpnorm --epochs $epochs   --baseline "V-Rex" --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --delta $delta --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 10   --split $split    --penalty_anneal_iters $penalty_anneal_iters  --penalty_weight $penalty_weight   --l2_regularizer_weight $l2_regularizer_weight 
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
done
