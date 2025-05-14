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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataset.dataset import DayCent_Dataset1, MW_Dataset1
from models.lstm import LSTMModel
import csv 
from train_t0 import Config
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.models_new import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer

def get_model(model_name, device, input_size=11, output_size=6):
    config = Config()
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

def main():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["lstm", 'ealstm', 'tcn', 'transformer', "itransformer", "pyraformer"],
                        help="Model name to use for training.")
    parser.add_argument("--subset", type=str, required=True, 
                    choices=['mw', 'dc', 'n2o'],
                    help="subset name for t0")
    parser.add_argument("--module", type=str, required=True, 
                    choices=['Combined', 'Water', 'Nitrogen', 'Thermal', 'Carbon', 'All'],
                    help="subset name for t0")
    parser.add_argument("--task", type=str, required=True, 
            choices=['t0', 't1'],
            help="which experiment to run")
    parser.add_argument("--exp", type=str, required=True, 
                choices=['temporal', 'spatial'],
                help="which experiment to run")
    parser.add_argument("--fold", type=int, required=True, 
            choices=[0, 1, 2, 3, 4],
            help="which fold to run")
    parser.add_argument("--save_path", type=str, default="county_metrics", 
                        help="Path to save the county-level metrics")
    args = parser.parse_args()
    print(f"Training model: {args.model} on subset: {args.subset} for exp: {args.exp}, task: {args.task}")
    
    # Load the tensors
    print("Loading data...")
    if args.subset == 'mw':
        data = MW_Dataset1(module_name=args.module, task=args.task, exp=args.exp, fold=args.fold)
        test_X, test_y = data.X_test, data.Y_test
    elif args.subset == 'dc':
        data = DayCent_Dataset1(module_name=args.module, task=args.task, exp=args.exp, fold=args.fold)
        test_X, test_y = data.X_test, data.Y_test

    print(f"test_X shape: {test_X.shape}, test_y shape: {test_y.shape}")
    
    # Only take the last year data (365 days)
    # Reshape to [3, 99, 20, 365, features] to extract only the last year
    X_reshaped = test_X.reshape(3, 99, 20, 365, test_X.shape[2])
    y_reshaped = test_y.reshape(3, 99, 20, 365, test_y.shape[2])
    
    # Take only the last year
    last_year_X = X_reshaped[2]  # shape: [99, 20, 365, 11]
    last_year_y = y_reshaped[2]  # shape: [99, 20, 365, 20]
    
    print(f"Last year X shape: {last_year_X.shape}, Last year y shape: {last_year_y.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)  # Check PyTorch version
    print(torch.cuda.is_available())  # Check if CUDA is available
    print(torch.version.cuda)  # Check the CUDA version PyTorch was built with

    if args.module == "All":
        Y_carbon = np.load(f'data_files/{args.task}_scalers/Y_scaler_Carbon.npy')
        Y_thermal = np.load(f'data_files/{args.task}_scalers/Y_scaler_Thermal.npy')
        Y_water = np.load(f'data_files/{args.task}_scalers/Y_scaler_Water.npy')
        Y_nitrogen = np.load(f'data_files/{args.task}_scalers/Y_scaler_Nitrogen.npy')
        y_train_mean = np.concatenate((Y_carbon[:, 0], Y_thermal[:, 0], Y_water[:, 0], Y_nitrogen[:, 0]))
        y_train_std = np.concatenate((Y_carbon[:, 1], Y_thermal[:, 1], Y_water[:, 1], Y_nitrogen[:, 1]))
    else:
        y_train_mean = np.load(f"data_files/t0_scalers/Y_scaler_{args.module}.npy")[:, 0]
        y_train_std = np.load(f"data_files/t0_scalers/Y_scaler_{args.module}.npy")[:, 1]
    print(f"y_train_mean: {y_train_mean.shape}, y_train_std: {y_train_std.shape}")

    # Define the model and load weights
    model = get_model(args.model, device, input_size=test_X.shape[2], output_size=test_y.shape[2])
    model.load_state_dict(torch.load(f"model_chkpoint/{args.task}_{args.subset}_{args.module}_{args.model}_{args.exp}_{args.fold}_model.pth"))
    criterion = nn.MSELoss()

    # Store metrics for each county and feature
    # [99 counties, 20 features, 3 metrics (R2, RMSE, MAE)]
    county_metrics = np.zeros((99, 20, 3))
    
    # Create a dummy batch to determine model output shape
    dummy_batch = last_year_X[0, 0].clone().detach().float().to(device)
    dummy_batch = dummy_batch.unsqueeze(0)  # Add batch dimension
    print(f"Dummy batch shape: {dummy_batch.shape}")
    
    with torch.no_grad():
        try:
            _, dummy_output = model(dummy_batch)
            print(f"Model output shape (from tuple): {dummy_output.shape}")
            use_tuple_output = True
        except:
            try:
                dummy_output = model(dummy_batch)
                print(f"Model output shape (direct): {dummy_output.shape}")
                use_tuple_output = False
            except Exception as e:
                print(f"Error with dummy batch: {e}")
                print("Please check model implementation.")
                return
    
    # Evaluation loop for each county
    model.eval()
    with torch.no_grad():
        for county_idx in tqdm(range(99), desc="Processing counties"):
            # Initialize arrays to store predictions and targets for all experiments and features
            all_exp_predictions = []
            all_exp_targets = []
            
            for exp_idx in range(20):
                # Get data for this county and experiment
                county_exp_X = last_year_X[county_idx, exp_idx]  # shape: [365, 11]
                county_exp_y = last_year_y[county_idx, exp_idx]  # shape: [365, 20]
                
                # Add batch dimension for model input
                batch_X = county_exp_X.clone().detach().float().to(device)
                batch_X = batch_X.unsqueeze(0)  # shape: [1, 365, 11]
                
                # Make prediction
                try:
                    if use_tuple_output:
                        _, outputs = model(batch_X)
                    else:
                        outputs = model(batch_X)
                    
                    # Convert outputs to numpy and keep the original shape (no squeeze)
                    pred = outputs.cpu().numpy()
                    
                    # Remove batch dimension if it exists
                    if pred.shape[0] == 1:
                        pred = pred[0]  # Now shape: [365, 20] or similar
                except Exception as e:
                    print(f"Error during model inference for county {county_idx}, exp {exp_idx}: {e}")
                    continue
                
                # Store predictions and targets
                all_exp_predictions.append(pred)
                all_exp_targets.append(county_exp_y.cpu().numpy() if isinstance(county_exp_y, torch.Tensor) else county_exp_y)
            
            if not all_exp_predictions:
                print(f"No valid predictions for county {county_idx}. Skipping.")
                continue
                
            # Convert to arrays for easier processing
            all_exp_predictions = np.array(all_exp_predictions)
            all_exp_targets = np.array(all_exp_targets)
            
            print(f"County {county_idx} - Predictions shape: {all_exp_predictions.shape}, Targets shape: {all_exp_targets.shape}")
            
            # Reverse normalization
            all_exp_predictions = all_exp_predictions * y_train_std + y_train_mean
            all_exp_targets = all_exp_targets * y_train_std + y_train_mean
            
            # Calculate metrics for each feature across all experiments for this county
            for feature_idx in range(20):
                # Get all predictions and targets for this feature across all experiments
                feature_preds = all_exp_predictions[..., feature_idx].flatten()
                feature_targets = all_exp_targets[..., feature_idx].flatten()
                
                # Calculate metrics
                feature_R2 = r2_score(feature_targets, feature_preds)
                feature_rmse = np.sqrt(mean_squared_error(feature_targets, feature_preds))
                feature_mae = mean_absolute_error(feature_targets, feature_preds)
                
                # Store metrics
                county_metrics[county_idx, feature_idx, 0] = feature_R2
                county_metrics[county_idx, feature_idx, 1] = feature_rmse
                county_metrics[county_idx, feature_idx, 2] = feature_mae
    
    # Print overall average metrics
    print("\nOverall Average Metrics:")
    avg_metrics = np.mean(county_metrics, axis=0)  # Average across counties, shape: [20, 3]
    
    for feature_idx in range(20):
        print(f"Feature {feature_idx + 1}: R2: {avg_metrics[feature_idx, 0]:.4f}, RMSE: {avg_metrics[feature_idx, 1]:.4f}, MAE: {avg_metrics[feature_idx, 2]:.4f}")
    
    # Save county-level metrics
    os.makedirs(args.save_path, exist_ok=True)
    np.save(f"{args.save_path}/{args.task}_{args.subset}_{args.module}_{args.model}_{args.exp}_{args.fold}_county_metrics.npy", county_metrics)
    
    print(f"\nCounty-level metrics saved to: {args.save_path}/{args.task}_{args.subset}_{args.module}_{args.model}_{args.exp}_{args.fold}_county_metrics.npy")
    
    # Also save as CSV for easier analysis
    csv_file = f"{args.save_path}/{args.task}_{args.subset}_{args.module}_{args.model}_{args.exp}_{args.fold}_county_metrics.csv"
    selected_features = [3]
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["County", "Feature", "R2", "RMSE", "MAE"])
        
        for county_idx in range(99):
            for feature_idx in range(20):
                if feature_idx+1 in selected_features:
                    writer.writerow([
                        county_idx + 1,
                        feature_idx + 1,
                        round(county_metrics[county_idx, feature_idx, 0], 4),
                        round(county_metrics[county_idx, feature_idx, 1], 4),
                        round(county_metrics[county_idx, feature_idx, 2], 4)
                    ])
    
    print(f"County-level metrics also saved to CSV: {csv_file}")

if __name__ == "__main__":
    main()