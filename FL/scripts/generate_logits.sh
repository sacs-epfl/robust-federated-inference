#!/bin/bash

root_dir=<absolute_path_to_the_root_of_this_repository>
save_dir=logits

env_python=<point_to_your_conda_python_interpreter>

dataset=CIFAR100
datapath=<point_to_your_dataset_path>
seeds=(90)
proxy_ratio=0.1
alphas=(0.5)
size=17

for alpha in ${alphas[@]}; do
    
    for seed in ${seeds[@]}; do

        modelpath=${root_dir}/results/local_training/${dataset}_localSGD_${alpha}_M${size}_${seed}_${proxy_ratio}
        
        name=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}
        log_dir=${root_dir}/results/${save_dir}/${name}
        mkdir -p $log_dir

        # count time for experiment in hh mm ss
        start=`date +%s`

        $env_python $root_dir/generate_logits_v2.py \
            --model ViT_B32 \
            --dataset $dataset \
            --datapath $datapath \
            --modelpath $modelpath \
            --partition_method dirichlet \
            --save_dir $log_dir \
            --alpha $alpha \
            --seed $seed \
            --proxy_ratio $proxy_ratio \
            --batch_size 64 \
            --gpu

        end=`date +%s`
        runtime=$((end-start))
        # Print time in hh:mm:ss
        echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"
    
    done

done
