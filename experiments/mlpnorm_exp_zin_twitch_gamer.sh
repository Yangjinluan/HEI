#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

dataset=twitch-gamer
sub_dataset=${2:-''}

infer_env_lr_lst=(1e-4 5e-4 1e-3 5e-3)
z_class_num_lst=(2 4 6 8)
hidden_dim_infer_lst=(16)
penalty_anneal_iters_lst=(5 10 15)
penalty_weight_lst=(1000 5000 10000.0 20000.0 50000.0)
l2_regularizer_weight_lst=(1e-3 5e-3)
lr_lst=(1e-2 5e-3 1e-3 5e-4)
weight_decay_lst=(1e-3 5e-4)


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

for infer_env_lr in "${infer_env_lr_lst[@]}"; do
    for z_class_num in "${z_class_num_lst[@]}"; do
        for hidden_dim_infer in "${hidden_dim_infer_lst[@]}"; do
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
                                                                    python -u main_zin.py --device "cuda:0" --dataset $dataset --sub_dataset ${sub_dataset:-''} --runs 5 --method mlpnorm --epochs $epochs   --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --delta $delta --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 10  --infer_env_lr $infer_env_lr --z_class_num $z_class_num  --hidden_dim_infer $hidden_dim_infer  --penalty_anneal_iters $penalty_anneal_iters  --penalty_weight $penalty_weight   --l2_regularizer_weight $l2_regularizer_weight  --directed
                                                                else
                                                                    python -u main_zin.py --device "cuda:0" --dataset $dataset --sub_dataset ${sub_dataset:-''} --runs 5 --method mlpnorm --epochs $epochs   --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --delta $delta --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 10  --infer_env_lr $infer_env_lr --z_class_num $z_class_num  --hidden_dim_infer $hidden_dim_infer  --penalty_anneal_iters $penalty_anneal_iters  --penalty_weight $penalty_weight   --l2_regularizer_weight $l2_regularizer_weight 
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
    done
done

# infer_env_lr_lst=(1e-3 5e-3 1e-2 5e-4)
# z_class_num_lst=(2 4 6 8)
# hidden_dim_infer_lst=(16 32 64)
# penalty_anneal_iters_lst=(5 10 15)
# penalty_weight_lst=(10000.0 20000.0 50000.0)
# l2_regularizer_weight_lst=(1e-3 5e-3 1e-2 5e-4)
# lr_lst=(1e-2 5e-3 1e-3 5e-4)
# weight_decay_lst=(1e-3 5e-4)
# num_layers_lst=(1)
# hidden_channels_lst=(64)