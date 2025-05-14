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
from dataset.dataset import DayCent_Dataset1, MW_Dataset1, t1_n2o_Dataset1
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.models_new import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer


class Config:
    def __init__(self, batch_size=128, enc_in=11, n_hidden=8, output_size=20):
        self.device = "cuda"
        self.batch_size = batch_size
        self.enc_in = enc_in
        self.hidden_size = 50
        self.batch_first = True
        self.num_layers = 3
        self.input_size = 11
        self.output_size = output_size
        self.seq_len = 365
        self.initial_forget_bias = 0
        self.input_size_dyn = 5
        self.input_size_stat = 10
        self.dropout = 0.2
        self.concat_static = False
        self.no_static = False
        self.num_channels = [1, 1, 1]
        self.kernel_size = 5
        self.d_model = 50
        self.e_layers = 3
        self.d_layers = 1
        self.n_heads = 5
        self.factor = 3
        self.embed = "time"
        self.freq = 'd'
        self.d_ff = 4 * self.d_model
        self.c_out = 20  # Updated to 20
        self.activation = 'relu'
        self.output_attention = False
        self.pred_len = 365
        self.task_name = 'long_term_forecast'

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
    args = parser.parse_args()
    print(f"Training model: {args.model} on subset: {args.subset} for exp: {args.exp}, task: {args.task}")
    # Load the tensors
    print("Loading data...")
    if args.subset == 'mw':
        data = MW_Dataset1(module_name=args.module, task=args.task, exp=args.exp, fold=args.fold)
        train_X, train_y, val_X, val_y = data.X_train, data.Y_train, data.X_val, data.Y_val
    elif args.subset == 'dc':
        data = DayCent_Dataset1(module_name=args.module, task=args.task, exp=args.exp, fold=args.fold)
        train_X, train_y, val_X, val_y = data.X_train, data.Y_train, data.X_val, data.Y_val
    

    print(f"train_X shape: {train_X.shape}, train_y shape: {train_y.shape}")
    print(f"val_X shape: {val_X.shape}, val_y shape: {val_y.shape}")
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
    # Create DataLoader for batching
    if args.model == "itransformer" :
        batch_size=10
    else:
        batch_size = 256
   
    train_dataset = TensorDataset(train_X.float(), train_y.float())
    val_dataset = TensorDataset(val_X.float(), val_y.float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the model, loss function, and optimizer
    model = get_model(args.model, device, input_size=train_X.shape[2], output_size=train_y.shape[2])
    # model = nn.DataParallel(model, device_ids=[0, 1]) 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in train_loader_tqdm:
            if args.model == "itransformer" or args.model == "transformer" or args.model == "pyraformer":
                if inputs.shape[0] != batch_size:
                    continue
            optimizer.zero_grad()
            _, outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Validation loop
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for inputs, targets in val_loader_tqdm:
                _, outputs = model(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
                val_loss += loss.item()
                val_loader_tqdm.set_postfix(loss=val_loss/len(val_loader))
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        print(f"Validation Loss: {val_loss/len(val_loader)}")
        # Calculate metrics for 3x dataset
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        # Reverse normalization
       
        val_predictions = val_predictions * y_train_std + y_train_mean
        val_targets = val_targets * y_train_std + y_train_mean
        val_R2 = r2_score(val_targets.flatten(), val_predictions.flatten())
        val_rmse = np.sqrt(mean_squared_error(val_targets.flatten(), val_predictions.flatten()))
        val_mae = mean_absolute_error(val_targets.flatten(), val_predictions.flatten())
        print(f"val R2: {val_R2:.4f}, val RMSE: {val_rmse:.4f}, val MAE: {val_mae:.4f}")

    # Save the model

    torch.save(model.state_dict(), f"model_chkpoint/{args.task}_{args.subset}_{args.module}_{args.model}_{args.exp}_{args.fold}_model.pth")  
    print("Model saved successfully.")

if __name__ == "__main__":
    main()