#!/bin/bash
# This script is used to train the models for the MW and DC benchmarks
# Logs for each run will be saved in the "logs/" directory

# Create the logs directory if it doesn't exist
mkdir -p logs_t2ad

# Define the parameters
models=("tcn" "itransformer" "pyraformer")
t0_subsets=("mw" "dc")
t1_subsets=("n2o" "co2")
modules=("All")
experiments=("temporal" "spatial")
folds=(0)

# Loop through all combinations of parameters for training
for model in "${models[@]}"; do
    for t0_subset in "${t0_subsets[@]}"; do
        for t1_subset in "${t1_subsets[@]}"; do
            for module in "${modules[@]}"; do
                for exp in "${experiments[@]}"; do
                    for fold in "${folds[@]}"; do
                        log_file="logs_t2ad/train_${model}_t0${t0_subset}_t1${t1_subset}_${module}_${exp}_fold${fold}.log"
                        echo "Running: python train_t2_ad.py --model $model --t0_subset $t0_subset --t1_subset $t1_subset --module $module --exp $exp --fold $fold"
                        python train_t2_ad.py --model "$model" --t0_subset "$t0_subset" --t1_subset "$t1_subset" --module "$module" --exp "$exp" --fold "$fold" > "$log_file" 2>&1
                    done
                done
            done
        done
    done
done

# Create logs directory for evaluation
mkdir -p logs_t2ad_eval

# Loop through all combinations of parameters for evaluation
for model in "${models[@]}"; do
    for t0_subset in "${t0_subsets[@]}"; do
        for t1_subset in "${t1_subsets[@]}"; do
            for module in "${modules[@]}"; do
                for exp in "${experiments[@]}"; do
                    for fold in "${folds[@]}"; do
                        log_file="logs_t2ad_eval/eval_${model}_t0${t0_subset}_t1${t1_subset}_${module}_${exp}_fold${fold}.log"
                        echo "Running: python evaluate_t2_ad.py --model $model --t0_subset $t0_subset --t1_subset $t1_subset --module $module --exp $exp --fold $fold --task t2"
                        python evaluate_t2_ad.py --model "$model" --t0_subset "$t0_subset" --t1_subset "$t1_subset" --module "$module" --exp "$exp" --fold "$fold" --task t2ad > "$log_file" 2>&1
                    done
                done
            done
        done
    done
done

echo "Transfer learning training and evaluation completed!"