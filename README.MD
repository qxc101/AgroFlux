# AgroFlux: A Spatial-Temporal Benchmark for Carbon and Nitrogen Flux Prediction in Agricultural Ecosystems

This repository contains the official implementation of paper AgroFlux: A Spatial-Temporal Benchmark for Carbon and Nitrogen Flux Prediction in Agricultural Ecosystems. AgroFlux is comprehensive benchmark that contains data for training, evaluating, and performing transfer learning on machine learning models, specifically focusing on CO₂, N₂O, and GPP (Gross Primary Production) predictions.

## Overview

The framework implements multiple time series machine learning models to predict greenhouse gas emissions and carbon uptake across different datasets and environmental scenarios. It supports:

- Training models on simulation data (T0)
- Training models on observational data (T1)
- Transfer learning from simulation to observational data (T2) using two approaches:
  - Fine-tuning based transfer learning
  - Adversarial domain adaptation

## Models Implemented

- LSTM (Long Short-Term Memory)
- EALSTM (Entity-Aware LSTM)
- TCN (Temporal Convolutional Network)
- Transformer
- iTransformer
- Pyraformer

## Datasets

The framework works with several datasets:

- **Simulation datasets (T0)**:
  - `mw`: Ecosys simulations
  - `dc`: DayCent simulations

- **Observation datasets (T1)**:
  - `n2o`: Nitrous oxide flux field measurements
  - `co2`: Carbon dioxide flux field measurements (includes GPP)

## Experiments

Two experiment types are supported:

- **Temporal**: Train and evaluate models based on temporal splits
- **Spatial**: Train and evaluate models using cross-validation across different spatial locations

## Usage
### Download dataset
Download the unzip the data files from:
https://drive.google.com/drive/folders/1keaLQtcfsG_F9IeED9uDSN9P9SnhHsFQ?usp=sharing

If you just want to use the data to do your own research, we also host this dataset on hugging face in format:
https://huggingface.co/datasets/qic69/AgroFlux_1

They are exactly the same data.

### Creating Environment
Create conda env using environment.yaml:
```bash
conda env create -f environment.yaml
```

Clone TSLib project into the models folder:
https://github.com/thuml/Time-Series-Library.git

### Training Models

#### T0 Models (Simulation Data)
To reproduce the result for predicting simulated data in the paper, run the following bash command. 
If you want to train individual models, run the following python command.
```bash
# Train all T0 models with full parameter sweep
./train_all_t0.sh

# Individual model training
python train_t0.py --model "lstm" --subset "mw" --module "All" --task t0 --exp temporal --fold 0
```

#### T1 Models (Observation Data)
To reproduce the result for predicting observation data in the paper, run the following bash command. 
If you want to train individual models, run the following python command.
```bash
# Train all T1 models with full parameter sweep
./train_all_t1.sh

# Individual model training
python train_t1.py --model "lstm" --subset "n2o" --module "All" --task t1 --exp temporal --fold 0
```

#### T2 Models (Transfer Learning)
To reproduce the result for transfer learning in the paper, run the following two bash commands. 
If you want to train individual models, run the following python command.
```bash
# Train all T2 models with fine-tuning approach
./train_all_t2_ft.sh

# Train all T2 models with adversarial domain adaptation
./train_all_t2_ad.sh

# Individual transfer learning
python train_t2.py --model "lstm" --t0_subset "mw" --t1_subset "n2o" --module "All" --exp temporal --fold 0
```

### Evaluation
The above bash scripts (e.g. train_all_t0.sh, train_all_t1.sh...) already contains evaluations in the paper.
However, if you want to evaluate individual models, use the following.
```bash
# Evaluate T0 models
python evaluate_t0.py --model "lstm" --subset "mw" --module "All" --task t0 --exp temporal --fold 0

# Evaluate T1 models
python evaluate_t1.py --model "lstm" --subset "n2o" --module "All" --task t1 --exp temporal --fold 0

# Evaluate T2 models
python evaluate_t2.py --model "lstm" --t0_subset "mw" --t1_subset "n2o" --module "All" --exp temporal --fold 0 --task t2
```

### Visualization

#### Generate Evaluation Tables
To reproduce the tables in the paper, use the following commands after running the training scripts in section Training Models above. 
```bash
# Generate LaTeX tables of evaluation metrics for T0 models
python generate_table_t0.py

# Generate LaTeX tables of evaluation metrics for T1 models
python generate_table_t1.py

# Generate LaTeX tables of evaluation metrics for T2 models
python generate_table_t2.py
```

#### Generate County Heatmaps
To reproduce the heatmap in the paper, use the following commands. 
```bash
# Generate heatmap of model performance by county
python generate_heatMap.py --metric R2 --county_file "county_metrics/Ecosys_99points.csv" --metrics_file "county_metrics/t0_mw_All_lstm_temporal_0_county_metrics.csv"
```

#### Generate Bar Chart
To reproduce the model comparsion bar chart in the paper, use the following commands. 
```bash
# Generate heatmap of model performance by county
python train_t0.py --model "lstm" --subset "mw" --module "Carbon" --task t0 --exp temporal --fold 0
python train_t0.py --model "lstm" --subset "mw" --module "Nitrogen" --task t0 --exp temporal --fold 0
python train_t0.py --model "lstm" --subset "mw" --module "Thermal" --task t0 --exp temporal --fold 0
python train_t0.py --model "lstm" --subset "mw" --module "Water" --task t0 --exp temporal --fold 0
python generate_heatMap.py --metric R2 --county_file "county_metrics/Ecosys_99points.csv" --metrics_file "county_metrics/t0_mw_All_lstm_temporal_0_county_metrics.csv"
```

## Data Exploration
Please refer to scale_data_t0.ipynb to see the plots and checks of the actual data. 
Please note we already have the scaled data, and the scalers.

## Citation

Citation will be added upon publication.

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
