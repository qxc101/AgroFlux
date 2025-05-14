import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from dataset.dataset import DayCent_Dataset1, MW_Dataset1
from models.lstm import LSTMModel
import csv 
from train_t0 import Config
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.models_new import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def get_model(model_name, device, input_size=11, output_size=6):
    config = Config(output_size=output_size)
    if model_name == "lstm":
        return LSTM(configs=config).to(device)
    elif model_name == "ealstm":
        return MyEALSTM(configs=config).to(device)
    elif model_name == "tcn":
        return MultiTCN(config=config).to(device)
    elif model_name == "transformer":
        return Transformer(config=config).to(device)
    elif model_name == "itransformer":
        return iTransformer(config).to(device)
    elif model_name == "pyraformer":
        return Pyraformer(config).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main(module, task="t0", subset="mw", model_name="lstm", exp="temporal", fold=0):
   
    
    print(f"Training model: {model_name} on subset: {subset} for exp: {exp}, task: {task}")
    # Load the tensors
    print("Loading data...")
    
    data = MW_Dataset1(module_name=module, task=task, exp=exp, fold=fold)
    test_X, test_y = data.X_test, data.Y_test


    print(f"test_X shape: {test_X.shape}, test_y shape: {test_y.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)  # Check PyTorch version
    print(torch.cuda.is_available())  # Check if CUDA is available
    print(torch.version.cuda)  # Check the CUDA version PyTorch was built with

    if module == "All":
        Y_carbon = np.load(f'data_files/{task}_scalers/Y_scaler_Carbon.npy')
        Y_thermal = np.load(f'data_files/{task}_scalers/Y_scaler_Thermal.npy')
        Y_water = np.load(f'data_files/{task}_scalers/Y_scaler_Water.npy')
        Y_nitrogen = np.load(f'data_files/{task}_scalers/Y_scaler_Nitrogen.npy')
        y_train_mean = np.concatenate((Y_carbon[:, 0], Y_thermal[:, 0], Y_water[:, 0], Y_nitrogen[:, 0]))
        y_train_std = np.concatenate((Y_carbon[:, 1], Y_thermal[:, 1], Y_water[:, 1], Y_nitrogen[:, 1]))
    else:
        y_train_mean = np.load(f"data_files/t0_scalers/Y_scaler_{module}.npy")[:, 0]
        y_train_std = np.load(f"data_files/t0_scalers/Y_scaler_{module}.npy")[:, 1]
    print(f"y_train_mean: {y_train_mean.shape}, y_train_std: {y_train_std.shape}")


    batch_size = 128

    test_dataset = TensorDataset(test_X.float(), test_y.float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  
    # Define the model, loss function, and optimizer
    model = get_model(model_name, device, input_size=test_X.shape[2], output_size=test_y.shape[2])
    print(f"model_chkpoint/{task}_{subset}_{module}_{model_name}_{exp}_{fold}_model.pth")
    model.load_state_dict(torch.load(f"model_chkpoint/{task}_{subset}_{module}_{model_name}_{exp}_{fold}_model.pth"))
    # model = nn.DataParallel(model, device_ids=[0, 1]) 
    criterion = nn.MSELoss()

    # Validation loop
    model.eval()
    test_predictions = []
    test_targets = []
    test_loss = 0.0
    test_loader_tqdm = tqdm(test_loader, desc="testidation")
    with torch.no_grad():
        for inputs, targets in test_loader_tqdm:
            if model == "itransformer" or model == "transformer" or model == "pyraformer":
                if inputs.shape[0] != batch_size:
                    continue
            _, outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            test_loss += loss.item()
            test_loader_tqdm.set_postfix(loss=test_loss/len(test_loader))
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
    
    print(f"test Loss: {test_loss/len(test_loader)}")
    # # Calculate metrics for 3x dataset
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    print(f"test_predictions shape: {test_predictions.shape}, test_targets shape: {test_targets.shape}")
    # Reverse normalization
    test_predictions = test_predictions * y_train_std + y_train_mean
    test_targets = test_targets * y_train_std + y_train_mean

    # Initialize lists to store metrics for each feature
    feature_metrics = []

    # Calculate metrics for each feature
    for i in range(test_predictions.shape[2]):  # Iterate over features
        feature_R2 = r2_score(test_targets[:, :, i].flatten(), test_predictions[:, :, i].flatten())
        feature_rmse = np.sqrt(mean_squared_error(test_targets[:, :, i].flatten(), test_predictions[:, :, i].flatten()))
        feature_mae = mean_absolute_error(test_targets[:, :, i].flatten(), test_predictions[:, :, i].flatten())
        feature_metrics.append((feature_R2, feature_rmse, feature_mae))
        print(f"Feature {i + 1}: R2: {feature_R2:.4f}, RMSE: {feature_rmse:.4f}, MAE: {feature_mae:.4f}")

    return feature_metrics


import pickle

if __name__ == "__main__":
    # Create a directory to store cached results
    cache_dir = os.path.join("cache", "evaluations")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "evaluation_results.pkl")
    
    # Check if cached results exist
    if os.path.exists(cache_file):
        print(f"Loading cached evaluation results from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_results = pickle.load(f)
            All = cached_results['All']
            Carbon = cached_results['Carbon']
            Nitrogen = cached_results['Nitrogen']
            Thermal = cached_results['Thermal']
            Water = cached_results['Water']
    else:
        # Run evaluations if cache doesn't exist
        print("No cached results found. Running evaluations...")
        All = main(module="All")
        Carbon = main(module="Carbon")
        Nitrogen = main(module="Nitrogen")
        Thermal = main(module="Thermal")
        Water = main(module="Water")
        
        # Save results to cache
        cached_results = {
            'All': All,
            'Carbon': Carbon,
            'Nitrogen': Nitrogen,
            'Thermal': Thermal,
            'Water': Water
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_results, f)
        print(f"Evaluation results saved to {cache_file}")

    # Configure plot style
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300
    })
    
    # Define feature names for each module
    carbon_features = ['Reco', 'GPP', 'CO2_FLUX', 'Yield', 'Delta_SOC']
    nitrogen_features = ['N2O_FLUX', 'NH4_1', 'NO3_1', 'NO3_3', 'NO3_5']
    thermal_features = ['TMAX_SOIL_1', 'TMIN_SOIL_1', 'TMAX_SOIL_3', 'TMIN_SOIL_3', 'TMAX_SOIL_5', 'TMIN_SOIL_5']
    water_features = ['WTR_1', 'WTR_3', 'WTR_5', 'ET']
    
    # Define representative features for comparison
    selected_features = {
        'Carbon': 'CO2_FLUX',  # Index 2 in Carbon module
        'Nitrogen': 'N2O_FLUX',  # Index 0 in Nitrogen module
        'Thermal': 'TMAX_SOIL_1',  # Index 0 in Thermal module
        'Water': 'WTR_1'  # Index 0 in Water module
    }
    
    # Create a mapping to find indices in the All module
    all_features = carbon_features + thermal_features + water_features + nitrogen_features
    all_indices = {feature: idx for idx, feature in enumerate(all_features)}
    
    # Create dataset for bar chart
    data = []
    for module_name, feature_name in selected_features.items():
        # Find the index in the respective module
        if module_name == 'Carbon':
            module_idx = carbon_features.index(feature_name)
            module_metrics = Carbon[module_idx]
        elif module_name == 'Nitrogen':
            module_idx = nitrogen_features.index(feature_name)
            module_metrics = Nitrogen[module_idx]
        elif module_name == 'Thermal':
            module_idx = thermal_features.index(feature_name)
            module_metrics = Thermal[module_idx]
        elif module_name == 'Water':
            module_idx = water_features.index(feature_name)
            module_metrics = Water[module_idx]
        
        # Find the index in the All module
        all_idx = all_indices[feature_name]
        all_metrics = All[all_idx]
        
        # Store the R2 scores (index 0 in the metrics tuple)
        data.append((module_name, feature_name, all_metrics[0], module_metrics[0]))
        print(f"Module: {module_name}, Feature: {feature_name}")
        print(f"  All model R2: {all_metrics[0]:.4f}")
        print(f"  Individual model R2: {module_metrics[0]:.4f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Extract data for plotting
    modules = [item[0] for item in data]
    features = [item[1] for item in data]
    r2_all = [item[2] for item in data]
    r2_individual = [item[3] for item in data]
    
    # Create x-axis labels with both module and feature names
    x_labels = [f"{m}\n({f})" for m, f in zip(modules, features)]
    
    # Set width of bars and positions
    bar_width = 0.2
    index = np.arange(len(modules))
    
    # Create the bars
    bars1 = ax.bar(index - bar_width/2, r2_all, 
                bar_width, label='All', color='#E1812C', edgecolor='black', linewidth=0.5,
                )
    bars2 = ax.bar(index + bar_width/2, r2_individual, 
                bar_width, label='Individual', color='#3274A1', edgecolor='black', linewidth=0.5,
                )
    
    # Add labels
    ax.set_ylabel('R$^2$', labelpad=8)
    ax.set_xticks(index)
    ax.set_xticklabels(x_labels)
    
    # Set y-axis limits
    y_max = max(max(r2_all), max(r2_individual)) * 1.1
    ax.set_ylim(0, y_max)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=True, edgecolor='black')
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=7)
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plt.savefig('performance_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()