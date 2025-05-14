#!/bin/bash
# This script is used to train the models for the MW and DC benchmarks
# Logs for each run will be saved in the "logs/" directory

# Create the logs directory if it doesn't exist
mkdir -p logs_eval

# Define the parameters
# models=("lstm")  # Add more models here in the future
# subsets=("mw" "dc")
# modules=("Combined" "Water" "Nitrogen" "Thermal" "Carbon")
# experiments=("temporal" "spatial")
# folds=(0 1 2 3 4)

# Define the parameters
models=("pyraformer")  # Add more models here in the future
subsets=("mw" "dc")
modules=("All")
experiments=("temporal" "spatial")
folds=(0 1 2 3 4)

# evaluate_t0.py --model "lstm" --subset "dc" --module "Combined" --task t0 --exp temporal --fold 0
# Loop through all combinations of parameters
for model in "${models[@]}"; do
    for subset in "${subsets[@]}"; do
        for module in "${modules[@]}"; do
            for exp in "${experiments[@]}"; do
                if [ "$exp" == "temporal" ]; then
                    # Temporal experiment only uses fold 0
                    log_file="logs_eval/eval_${model}_${subset}_${module}_${exp}_fold0.log"
                    echo "Running: python evaluate_t0.py --model $model --subset $subset --module $module --task t0 --exp $exp --fold 0"
                    python evaluate_t0.py --model "$model" --subset "$subset" --module "$module" --task t0 --exp "$exp" --fold 0 > "$log_file" 2>&1
                else
                    # Spatial experiment uses all folds
                    for fold in "${folds[@]}"; do
                        log_file="logs_eval/eval_${model}_${subset}_${module}_${exp}_fold${fold}.log"
                        echo "Running: python evaluate_t0.py --model $model --subset $subset --module $module --task t0 --exp $exp --fold $fold"
                        python evaluate_t0.py --model "$model" --subset "$subset" --module "$module" --task t0 --exp "$exp" --fold "$fold" > "$log_file" 2>&1
                    done
                fi
            done
        done
    done
done