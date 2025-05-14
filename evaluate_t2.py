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
from dataset.dataset import t1_n2o_Dataset1, t1_co2_Dataset1
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
    parser = argparse.ArgumentParser(description="Transfer learning from t0 to t1.")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["lstm", 'ealstm', 'tcn', 'transformer', "itransformer", "pyraformer"],
                        help="Model name to use for training.")
    parser.add_argument("--t0_subset", type=str, required=True, 
                    choices=['mw', 'dc'],
                    help="t0 dataset used for pre-training")
    parser.add_argument("--t1_subset", type=str, required=True, 
                    choices=['n2o', 'co2'],
                    help="t1 dataset used for fine-tuning")
    parser.add_argument("--module", type=str, required=True, 
                    choices=['Combined', 'Water', 'Nitrogen', 'Thermal', 'Carbon', 'All'],
                    help="Module name for t0 pre-training")
    parser.add_argument("--exp", type=str, required=True, 
                choices=['temporal', 'spatial'],
                help="which experiment to run")
    parser.add_argument("--fold", type=int, required=True, 
            choices=[0, 1, 2, 3, 4],
            help="which fold to run")
    parser.add_argument("--task", type=str, required=True, 
        choices=['t2'],
        help="which experiment to run")
    args = parser.parse_args()
    
    # Load the tensors
    print("Loading data...")
    if args.t1_subset == 'n2o':
        data = t1_n2o_Dataset1(module_name=args.module, task=args.task, exp=args.exp, fold=args.fold)
    elif args.t1_subset == 'co2':
        data = t1_co2_Dataset1(module_name=args.module, task=args.task, exp=args.exp, fold=args.fold)
    test_X, test_y = data.X_test, data.Y_test

    print(f"test_X shape: {test_X.shape}, test_y shape: {test_y.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)  # Check PyTorch version
    print(torch.cuda.is_available())  # Check if CUDA is available
    print(torch.version.cuda)  # Check the CUDA version PyTorch was built with
    if args.t1_subset == 'n2o':
        y_train_mean = np.load(f'data_files/t0_scalers/Y_scaler_Nitrogen.npy')[0, 0]
        y_train_std = np.load(f"data_files/t0_scalers/Y_scaler_Nitrogen.npy")[0, 1]
    elif args.t1_subset == 'co2':
        y_train_mean = np.load(f'data_files/t0_scalers/Y_scaler_Carbon.npy')[[2,1], 0]
        y_train_std = np.load(f'data_files/t0_scalers/Y_scaler_Carbon.npy')[[2,1], 1]

    print(f"y_train_mean: {y_train_mean.shape}, y_train_std: {y_train_std.shape}")
    # Create DataLoader for batching

    # Create DataLoader for batching
    if args.model == "itransformer" :
        batch_size=10
    else:
        batch_size = 128

    test_dataset = TensorDataset(test_X.float(), test_y.float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  
    # Define the model, loss function, and optimizer
    model = get_model(args.model, device, input_size=test_X.shape[2], output_size=test_y.shape[2])
    model.load_state_dict(torch.load(f"model_chkpoint/t2_t0{args.t0_subset}_t1{args.t1_subset}_{args.model}_{args.exp}_{args.fold}_model.pth"))
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
            # if args.model == "itransformer" or args.model == "transformer" or args.model == "pyraformer":
            #     if inputs.shape[0] != batch_size:
            #         continue
            _, outputs = model(inputs.to(device))
            
            if args.t1_subset == 'n2o':
                mask = ~torch.isnan(targets[:, :, 0])
                y_true_filtered = targets[:, :, 0][mask]
                y_pred_filtered = outputs[:, :, 15][mask]
                
                loss = criterion(y_pred_filtered, y_true_filtered.float().to(device))
            elif args.t1_subset == 'co2':
                # only finetuning CO2_FLUX and GPP
                mask_CO2 = ~torch.isnan(targets[:, :, 1])
                y_true_filtered_CO2 = targets[:, :, 1][mask_CO2]
                y_pred_filtered_CO2 = outputs[:, :, 2][mask_CO2]

                mask_GPP = ~torch.isnan(targets[:, :, 0])
                y_true_filtered_GPP = targets[:, :, 0][mask_GPP]
                y_pred_filtered_GPP = outputs[:, :, 1][mask_GPP]
                y_pred_filtered = torch.cat([y_pred_filtered_CO2.unsqueeze(dim=-1), y_pred_filtered_GPP.unsqueeze(dim=-1)], dim=1)
                y_true_filtered = torch.cat([y_true_filtered_CO2.unsqueeze(dim=-1), y_true_filtered_GPP.unsqueeze(dim=-1)], dim=1)
                loss = criterion(y_pred_filtered, y_true_filtered.to(device))
        
            test_loss += loss.item()
            test_loader_tqdm.set_postfix(loss=test_loss/len(test_loader))
            test_predictions.append(y_pred_filtered.cpu().numpy())
            test_targets.append(y_true_filtered.cpu().numpy())
    
    print(f"test Loss: {test_loss/len(test_loader)}")
    # # Calculate metrics for 3x dataset
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    # Reverse normalization
    print(  f"test_predictions shape: {test_predictions.shape}, test_targets shape: {test_targets.shape}")
    test_predictions = test_predictions * y_train_std + y_train_mean
    test_targets = test_targets * y_train_std + y_train_mean

    # Initialize lists to store metrics for each feature
    feature_metrics = []

    # Calculate metrics for each feature
    if args.t1_subset == 'n2o':
        feature_R2 = r2_score(test_targets.flatten(), test_predictions.flatten())
        feature_rmse = np.sqrt(mean_squared_error(test_targets.flatten(), test_predictions.flatten()))
        feature_mae = mean_absolute_error(test_targets.flatten(), test_predictions.flatten())
        feature_metrics.append((feature_R2, feature_rmse, feature_mae))
        print(f"Feature {1}: R2: {feature_R2:.4f}, RMSE: {feature_rmse:.4f}, MAE: {feature_mae:.4f}")
    elif args.t1_subset == 'co2':
        for i in range(test_predictions.shape[1]):  # Iterate over features
            feature_R2 = r2_score(test_targets[:, i].flatten(), test_predictions[:, i].flatten())
            feature_rmse = np.sqrt(mean_squared_error(test_targets[:, i].flatten(), test_predictions[:, i].flatten()))
            feature_mae = mean_absolute_error(test_targets[:, i].flatten(), test_predictions[:, i].flatten())
            feature_metrics.append((feature_R2, feature_rmse, feature_mae))
            print(f"Feature {i + 1}: R2: {feature_R2:.4f}, RMSE: {feature_rmse:.4f}, MAE: {feature_mae:.4f}")

    # Save results to a CSV file
    results_file = "evaluation_results_t2.csv"
    file_exists = os.path.isfile(results_file)

    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file is being created
            writer.writerow(["Model", "t0_Subset", "t1_Subset", "Module", "Task", "Experiment", "Fold", "Feature", "R2", "RMSE", "MAE"])
        # Append the results for each feature
        for i, (feature_R2, feature_rmse, feature_mae) in enumerate(feature_metrics):
            writer.writerow([args.model, args.t0_subset, args.t1_subset, args.module, args.task, args.exp, args.fold, f"Feature {i + 1}", round(feature_R2, 3), round(feature_rmse, 3), round(feature_mae, 3)])

    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()