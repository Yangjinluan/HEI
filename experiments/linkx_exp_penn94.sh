#!/bin/bash
# fb100
dataset=$1
sub_dataset=${2:-''}

# infer_env_lr_lst=(1e-3 5e-3 1e-2 5e-4)
infer_env_lr_lst=(1e-4 5e-4 1e-3)
z_class_num_lst=(2 4 6 8)
hidden_dim_infer_lst=(16 32 64)
penalty_anneal_iters_lst=(5 10 15)
penalty_weight_lst=(10000.0 20000.0 50000.0)
l2_regularizer_weight_lst=(1e-3 5e-3 1e-2 5e-4)
lr_lst=(1e-2 5e-3 1e-3 5e-4)
weight_decay_lst=(1e-3 5e-4)
num_layers_lst=(2)
hidden_channels_lst=(256)

for infer_env_lr in "${infer_env_lr_lst[@]}"; do
    for z_class_num in "${z_class_num_lst[@]}"; do
        for hidden_dim_infer in "${hidden_dim_infer_lst[@]}"; do
            for penalty_anneal_iters in "${penalty_anneal_iters_lst[@]}"; do
                for penalty_weight in "${penalty_weight_lst[@]}"; do
                    for l2_regularizer_weight in "${l2_regularizer_weight_lst[@]}"; do
                        for lr in "${lr_lst[@]}"; do
                            for weight_decay in "${weight_decay_lst[@]}"; do
                                for num_layers in "${num_layers_lst[@]}"; do
                                    for hidden_channels in "${hidden_channels_lst[@]}"; do
                                        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                                            echo "Running $dataset "
                                            python main_HEI.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkx  --device "cuda:3" --runs 5  --num_layers $num_layers   --hidden_channels $hidden_channels  --lr $lr --weight_decay $weight_decay --display_step 10  --infer_env_lr $infer_env_lr --z_class_num $z_class_num  --hidden_dim_infer $hidden_dim_infer  --penalty_anneal_iters $penalty_anneal_iters  --penalty_weight $penalty_weight   --l2_regularizer_weight $l2_regularizer_weight  --directed
                                        else
                                            python main_HEI.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkx  --device "cuda:3" --runs 5 --num_layers $num_layers   --hidden_channels $hidden_channels --lr $lr --weight_decay $weight_decay --display_step 10  --infer_env_lr $infer_env_lr --z_class_num $z_class_num  --hidden_dim_infer $hidden_dim_infer  --penalty_anneal_iters $penalty_anneal_iters  --penalty_weight $penalty_weight   --l2_regularizer_weight $l2_regularizer_weight 
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
