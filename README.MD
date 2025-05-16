# AgroFlux: A Spatial-Temporal Benchmark for Carbon and Nitrogen Flux Prediction in Agricultural Ecosystems

This repository contains the official implementation of paper AgroFlux: A Spatial-Temporal Benchmark for Carbon and Nitrogen Flux Prediction in Agricultural Ecosystems. AgroFlux is comprehensive benchmark that contains data for training, evaluating, and performing transfer learning on machine learning models, specifically focusing on CO₂, N₂O, and GPP (Gross Primary Production) predictions.

## Overview

The framework implements multiple time series machine learning models to predict greenhouse gas emissions and carbon uptake across different datasets and environmental scenarios. It supports:

- Training models on simulation data (T0)
- Training models on observational data (T1)
- Transfer learning from simulation to observational data (T2) using two approaches:
  - Fine-tuning based transfer learning
  - Adversarial domain adaptation

## Datasets

The framework works with several data categories:

- **Simulation datasets (T0)**:
  - `mw`: Ecosys model simulations
  - `dc`: DC model simulations

- **Observation datasets (T1)**:
  - `n2o`: Nitrous oxide field measurements
  - `co2`: Carbon dioxide field measurements (includes GPP)

## Experiments

Two experiment types are supported:

- **Temporal**: Train and evaluate models based on temporal splits
- **Spatial**: Train and evaluate models using cross-validation across different spatial locations

## Usage

### Training Models

#### T0 Models (Simulation Data)
To reproduce the result for predicting simulated data in the paper, run the following bash command. 
If you want to train individual models, tran the following python command.
```bash
# Train all T0 models with full parameter sweep
./train_all_t0.sh

# Individual model training
python train_t0.py --model "lstm" --subset "mw" --module "All" --task t0 --exp temporal --fold 0
```

#### T1 Models (Observation Data)
To reproduce the result for predicting observation data in the paper, run the following bash command. 
If you want to train individual models, tran the following python command.
```bash
# Train all T1 models with full parameter sweep
./train_all_t1.sh

# Individual model training
python train_t1.py --model "lstm" --subset "n2o" --module "All" --task t1 --exp temporal --fold 0
```

#### T2 Models (Transfer Learning)
To reproduce the result for transfer learning in the paper, run the following two bash commands. 
If you want to train individual models, tran the following python command.
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
```bash
# Generate LaTeX tables of evaluation metrics for T0 models
python generate_table_t0.py

# Generate LaTeX tables of evaluation metrics for T1 models
python generate_table_t1.py

# Generate LaTeX tables of evaluation metrics for T2 models
python generate_table_t2.py
```

#### Generate County Heatmaps
```bash
# Generate heatmap of model performance by county
python generate_heatMap.py --metric R2 --county_file "county_metrics/Ecosys_99points.csv" --metrics_file "county_metrics/t0_mw_All_lstm_temporal_0_county_metrics.csv"
```

## Results Analysis

The framework provides tools to analyze model performance across:

- Different machine learning architectures
- Simulation vs. observational data
- Geographical regions (counties across IL, IN, IA)
- Different greenhouse gases and environmental variables
- Transfer learning effectiveness

## File Structure

- **Training Scripts**:
  - `train_all_t0.sh`: Train all T0 models
  - `train_all_t1.sh`: Train all T1 models
  - `train_all_t2.sh`: Train all T2 models with fine-tuning
  - `train_all_t2_ad.sh`: Train all T2 models with adversarial domain adaptation

- **Evaluation Scripts**:
  - `generate_table_t0.py`: Create LaTeX tables for T0 model evaluations
  - `generate_table_t1.py`: Create LaTeX tables for T1 model evaluations
  - `generate_table_t2.py`: Create LaTeX tables for T2 model evaluations
  - `generate_heatMap.py`: Generate county-level performance heatmaps

## Dependencies

- Python 3.7+
- PyTorch
- pandas
- matplotlib
- numpy
- geopandas (optional, for advanced mapping)
- shapely (optional, for advanced mapping)

## Citation

If you use this code in your research, please cite:

```
[Citation information here]
```

## License

[License information here]
