#!/bin/bash
# This script is used to train the models for the MW and DC benchmarks
# Logs for each run will be saved in the "logs/" directory

# Create the logs directory if it doesn't exist
mkdir -p logs

# Define the parameters
# models=("lstm")  # Add more models here in the future
# subsets=("mw" "dc")
# modules=("Combined" "Water" "Nitrogen" "Thermal" "Carbon")
# experiments=("temporal" "spatial")
# folds=(0 1 2 3 4)


# Define the parameters
# models=("lstm" "ealstm" "tcn" "transformer" "itransformer" "pyraformer")  # Add more models here in the future
models=("lstm" "ealstm" "tcn" "transformer" "itransformer" "pyraformer")
subsets=("mw" "dc")
modules=("All")
experiments=("temporal" "spatial")
folds=(0 1 2 3 4)

# train_t0.py --model "lstm" --subset "mw" --module "Combined" --task t0 --exp temporal --fold 0

# Loop through all combinations of parameters
for model in "${models[@]}"; do
    for subset in "${subsets[@]}"; do
        for module in "${modules[@]}"; do
            for exp in "${experiments[@]}"; do
                if [ "$exp" == "temporal" ]; then
                    # Temporal experiment only uses fold 0
                    log_file="logs/${model}_${subset}_${module}_${exp}_fold0.log"
                    echo "Running: python train_t0.py --model $model --subset $subset --module $module --task t0 --exp $exp --fold 0"
                    python train_t0.py --model "$model" --subset "$subset" --module "$module" --task t0 --exp "$exp" --fold 0 > "$log_file" 2>&1
                else
                    # Spatial experiment uses all folds
                    for fold in "${folds[@]}"; do
                        log_file="logs/${model}_${subset}_${module}_${exp}_fold${fold}.log"
                        echo "Running: python train_t0.py --model $model --subset $subset --module $module --task t0 --exp $exp --fold $fold"
                        python train_t0.py --model "$model" --subset "$subset" --module "$module" --task t0 --exp "$exp" --fold "$fold" > "$log_file" 2>&1
                    done
                fi
            done
        done
    done
done


mkdir -p logs_eval

# Define the parameters
models=("lstm" "ealstm" "tcn" "transformer" "itransformer" "pyraformer")  # Add more models here in the future
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

#!/bin/bash
# This script is used to train the models for the MW and DC benchmarks
# Logs for each run will be saved in the "logs/" directory

# Create the logs directory if it doesn't exist
mkdir -p logs_t1

# Define the parameters
models=("lstm" "ealstm" "tcn" "transformer" "itransformer" "pyraformer")  # Add more models here in the future
subsets=("n2o" "co2")
modules=("All")
experiments=("temporal" "spatial")
folds=(0 1 2 3 4)
# train_t1.py --model "lstm" --subset "n2o" --module "Combined" --task t1 --exp "temporal" --fold 0
# Loop through all combinations of parameters
for model in "${models[@]}"; do
    for subset in "${subsets[@]}"; do
        for module in "${modules[@]}"; do
            for exp in "${experiments[@]}"; do
                if [ "$exp" == "temporal" ]; then
                    # Temporal experiment only uses fold 0
                    log_file="logs_t1/${model}_${subset}_${module}_${exp}_fold0.log"
                    echo "Running: python train_t1.py --model $model --subset $subset --module $module --task t1 --exp $exp --fold 0"
                    python train_t1.py --model "$model" --subset "$subset" --module "$module" --task t1 --exp "$exp" --fold 0 > "$log_file" 2>&1
                else
                    # Spatial experiment uses all folds
                    for fold in "${folds[@]}"; do
                        log_file="logs_t1/${model}_${subset}_${module}_${exp}_fold${fold}.log"
                        echo "Running: python train_t1.py --model $model --subset $subset --module $module --task t1 --exp $exp --fold $fold"
                        python train_t1.py --model "$model" --subset "$subset" --module "$module" --task t1 --exp "$exp" --fold "$fold" > "$log_file" 2>&1
                    done
                fi
            done
        done
    done
done

mkdir -p logs_t1_eval
for model in "${models[@]}"; do
    for subset in "${subsets[@]}"; do
        for module in "${modules[@]}"; do
            for exp in "${experiments[@]}"; do
                if [ "$exp" == "temporal" ]; then
                    # Temporal experiment only uses fold 0
                    log_file="logs_t1_eval/eval_${model}_${subset}_${module}_${exp}_fold0.log"
                    echo "Running: python evaluate_t1.py --model $model --subset $subset --module $module --task t1 --exp $exp --fold 0"
                    python evaluate_t1.py --model "$model" --subset "$subset" --module "$module" --task t1 --exp "$exp" --fold 0 > "$log_file" 2>&1
                else
                    # Spatial experiment uses all folds
                    for fold in "${folds[@]}"; do
                        log_file="logs_t1_eval/eval_${model}_${subset}_${module}_${exp}_fold${fold}.log"
                        echo "Running: python evaluate_t1.py --model $model --subset $subset --module $module --task t1 --exp $exp --fold $fold"
                        python evaluate_t1.py --model "$model" --subset "$subset" --module "$module" --task t1 --exp "$exp" --fold "$fold" > "$log_file" 2>&1
                    done
                fi
            done
        done
    done
done

#!/bin/bash
# This script runs transfer learning experiments from t0 to t1 models
# Logs for each run will be saved in the "logs_t2/" directory

# Create the logs directory if it doesn't exist
mkdir -p logs_t2

# Define the parameters
models=("lstm" "ealstm" "tcn" "transformer" "itransformer" "pyraformer")
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
                        log_file="logs_t2/train_${model}_t0${t0_subset}_t1${t1_subset}_${module}_${exp}_fold${fold}.log"
                        echo "Running: python train_t2.py --model $model --t0_subset $t0_subset --t1_subset $t1_subset --module $module --exp $exp --fold $fold"
                        python train_t2.py --model "$model" --t0_subset "$t0_subset" --t1_subset "$t1_subset" --module "$module" --exp "$exp" --fold "$fold" > "$log_file" 2>&1
                    done
                done
            done
        done
    done
done

# Create logs directory for evaluation
mkdir -p logs_t2_eval

# Loop through all combinations of parameters for evaluation
for model in "${models[@]}"; do
    for t0_subset in "${t0_subsets[@]}"; do
        for t1_subset in "${t1_subsets[@]}"; do
            for module in "${modules[@]}"; do
                for exp in "${experiments[@]}"; do
                    for fold in "${folds[@]}"; do
                        log_file="logs_t2_eval/eval_${model}_t0${t0_subset}_t1${t1_subset}_${module}_${exp}_fold${fold}.log"
                        echo "Running: python evaluate_t2.py --model $model --t0_subset $t0_subset --t1_subset $t1_subset --module $module --exp $exp --fold $fold --task t2"
                        python evaluate_t2.py --model "$model" --t0_subset "$t0_subset" --t1_subset "$t1_subset" --module "$module" --exp "$exp" --fold "$fold" --task t2 > "$log_file" 2>&1
                    done
                done
            done
        done
    done
done

echo "Transfer learning training and evaluation completed!"