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
from dataset.dataset import DayCent_Dataset1, MW_Dataset1, t1_n2o_Dataset1, t1_co2_Dataset1
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.models_new import LSTM, MyEALSTM, MultiTCN, Transformer, iTransformer, Pyraformer
from train_t0 import Config, get_model

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
    parser.add_argument("--freeze_layers", action='store_true',
                help="Whether to freeze some layers of the pre-trained model")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                help="Learning rate for fine-tuning")
    args = parser.parse_args()
    
    # 1. Load t1 data for fine-tuning
    print(f"Loading t1 data ({args.t1_subset})...")
    if args.t1_subset == 'n2o':
        data = t1_n2o_Dataset1(module_name='All', task='t1', exp=args.exp, fold=args.fold)
        train_X, train_y, val_X, val_y = data.X_train, data.Y_train, data.X_val, data.Y_val
    elif args.t1_subset == 'co2':
        data = t1_co2_Dataset1(module_name='All', task='t1', exp=args.exp, fold=args.fold)
        train_X, train_y, val_X, val_y = data.X_train, data.Y_train, data.X_val, data.Y_val

    print(f"train_X shape: {train_X.shape}, train_y shape: {train_y.shape}")
    print(f"val_X shape: {val_X.shape}, val_y shape: {val_y.shape}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model in ["itransformer", "transformer", "pyraformer"]:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2. Initialize the model with pre-trained weights
    model = get_model(args.model, device, input_size=train_X.shape[2], output_size=train_y.shape[2])
    
    # Load pre-trained weights from t0 model
    t0_model_path = f"model_chkpoint/t0_{args.t0_subset}_{args.module}_{args.model}_{args.exp}_{args.fold}_model.pth"
    print(f"Loading pre-trained model from {t0_model_path}")
    
    try:
        state_dict = torch.load(t0_model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pre-trained weights")
    except Exception as e:
        print(f"Warning: Could not load pre-trained weights: {e}")
        print("Training from scratch instead")
    
    # 3. Optional: Freeze some layers
    if args.freeze_layers:
        print("Freezing early layers...")
        # Freeze parameters in the first few layers based on model type
        if args.model in ["lstm", "ealstm"]:
            for name, param in model.named_parameters():
                if "lstm" in name:
                    param.requires_grad = False
        elif args.model == "tcn":
            for name, param in model.named_parameters():
                if "network" in name and "0" in name:  # First TCN layer
                    param.requires_grad = False
        elif args.model in ["transformer", "itransformer", "pyraformer"]:
            for name, param in model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False
    
    # 4. Set up data loaders for fine-tuning
    batch_size = 2 if args.model == "itransformer" else 2  # Smaller batch size for fine-tuning
    
    train_dataset = TensorDataset(train_X.float(), train_y.float())
    val_dataset = TensorDataset(val_X.float(), val_y.float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                         lr=args.learning_rate)  # Typically use lower learning rate for fine-tuning
    
    # 6. Set up normalization parameters for t1 data
    if args.t1_subset == 'n2o':
        y_train_mean = np.load(f'data_files/t0_scalers/Y_scaler_Nitrogen.npy')[0, 0]
        y_train_std = np.load(f"data_files/t0_scalers/Y_scaler_Nitrogen.npy")[0, 1]
    elif args.t1_subset == 'co2':
        y_train_mean = np.load(f'data_files/t0_scalers/Y_scaler_Carbon.npy')[[2,1], 0]
        y_train_std = np.load(f'data_files/t0_scalers/Y_scaler_Carbon.npy')[[2,1], 1]
    print(f"y_train_mean: {y_train_mean.shape}, y_train_std: {y_train_std.shape}")
    
    # 7. Training loop for fine-tuning
    best_val_loss = float('inf')
    num_epochs = 20  # Can be shorter for fine-tuning
    patience = 5  # For early stopping
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for inputs, targets in train_loader_tqdm:
            optimizer.zero_grad()
            _, outputs = model(inputs.to(device))
            
            # Handle different target variables based on dataset
            if args.t1_subset == 'n2o':
                mask = ~torch.isnan(targets[:, :, 0])
                y_true_filtered = targets[:, :, 0][mask]
                y_pred_filtered = outputs[:, :, 15][mask]
                loss = criterion(y_pred_filtered, y_true_filtered.to(device))
                
            elif args.t1_subset == 'co2':
                # Only fine-tuning CO2_FLUX and GPP
                mask_CO2 = ~torch.isnan(targets[:, :, 1])
                y_true_filtered_CO2 = targets[:, :, 1][mask_CO2]
                y_pred_filtered_CO2 = outputs[:, :, 2][mask_CO2]

                mask_GPP = ~torch.isnan(targets[:, :, 0])
                y_true_filtered_GPP = targets[:, :, 0][mask_GPP]
                y_pred_filtered_GPP = outputs[:, :, 1][mask_GPP]

                loss = criterion(torch.cat([y_pred_filtered_CO2, y_pred_filtered_GPP], dim=0), 
                               torch.cat([y_true_filtered_CO2, y_true_filtered_GPP], dim=0).to(device))
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        avg_train_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")

        # Validation step
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for inputs, targets in val_loader_tqdm:
                _, outputs = model(inputs.to(device))
                if len(outputs.shape) != 3:
                    outputs = outputs.unsqueeze(0)
                if args.t1_subset == 'n2o':
                    mask = ~torch.isnan(targets[:, :, 0])
                    y_true_filtered = targets[:, :, 0][mask]
                    y_pred_filtered = outputs[:, :, 15][mask]
                    
                    loss = criterion(y_pred_filtered, y_true_filtered.float().to(device))
                    val_predictions.append(y_pred_filtered.cpu().numpy())
                    val_targets.append(y_true_filtered.cpu().numpy())
                    
                elif args.t1_subset == 'co2':
                    # Only evaluating CO2_FLUX and GPP
                    print(f"outputs shape: {outputs.shape}, targets shape: {targets.shape}")
                    mask_CO2 = ~torch.isnan(targets[:, :, 1])
                    y_true_filtered_CO2 = targets[:, :, 1][mask_CO2]
                    y_pred_filtered_CO2 = outputs[:, :, 2][mask_CO2]

                    mask_GPP = ~torch.isnan(targets[:, :, 0])
                    y_true_filtered_GPP = targets[:, :, 0][mask_GPP]
                    y_pred_filtered_GPP = outputs[:, :, 1][mask_GPP]
                    
                    y_pred_filtered = torch.cat([y_pred_filtered_CO2, y_pred_filtered_GPP], dim=0)
                    y_true_filtered = torch.cat([y_true_filtered_CO2, y_true_filtered_GPP], dim=0)
                    
                    loss = criterion(y_pred_filtered, y_true_filtered.to(device))
                    val_predictions.append(y_pred_filtered.cpu().numpy())
                    val_targets.append(y_true_filtered.cpu().numpy())
            
                val_loss += loss.item()
                val_loader_tqdm.set_postfix(loss=val_loss/len(val_loader))
        
        avg_val_loss = val_loss/len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # Calculate metrics
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        # Reverse normalization for proper metrics
        if args.t1_subset == 'n2o':
            val_predictions = val_predictions * y_train_std + y_train_mean
            val_targets = val_targets * y_train_std + y_train_mean
        elif args.t1_subset == 'co2':
            # Handle the concatenated predictions for CO2 and GPP
            val_predictions_gpp = val_predictions[:val_predictions.shape[0]//2] * y_train_std[0] + y_train_mean[0]
            val_predictions_co2 = val_predictions[val_predictions.shape[0]//2:] * y_train_std[1] + y_train_mean[1]
            val_targets_gpp = val_targets[:val_targets.shape[0]//2] * y_train_std[0] + y_train_mean[0]
            val_targets_co2 = val_targets[val_targets.shape[0]//2:] * y_train_std[1] + y_train_mean[1]
            
            # Calculate metrics separately for GPP and CO2
            gpp_r2 = r2_score(val_targets_gpp.flatten(), val_predictions_gpp.flatten())
            gpp_rmse = np.sqrt(mean_squared_error(val_targets_gpp.flatten(), val_predictions_gpp.flatten()))
            co2_r2 = r2_score(val_targets_co2.flatten(), val_predictions_co2.flatten())
            co2_rmse = np.sqrt(mean_squared_error(val_targets_co2.flatten(), val_predictions_co2.flatten()))
            
            print(f"GPP - R2: {gpp_r2:.4f}, RMSE: {gpp_rmse:.4f}")
            print(f"CO2 - R2: {co2_r2:.4f}, RMSE: {co2_rmse:.4f}")
            
            # Recombine for overall metrics
            val_predictions = np.concatenate([val_predictions_gpp, val_predictions_co2])
            val_targets = np.concatenate([val_targets_gpp, val_targets_co2])
        
        val_R2 = r2_score(val_targets.flatten(), val_predictions.flatten())
        val_rmse = np.sqrt(mean_squared_error(val_targets.flatten(), val_predictions.flatten()))
        val_mae = mean_absolute_error(val_targets.flatten(), val_predictions.flatten())
        print(f"Overall - R2: {val_R2:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            model_save_path = f"model_chkpoint/t2_t0{args.t0_subset}_t1{args.t1_subset}_{args.model}_{args.exp}_{args.fold}_model.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print("Transfer learning completed successfully.")
    model_save_path = f"model_chkpoint/t2_t0{args.t0_subset}_t1{args.t1_subset}_{args.model}_{args.exp}_{args.fold}_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")
if __name__ == "__main__":
    main()