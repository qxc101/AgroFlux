
import json
from torch.utils.data import Dataset
import torch
import os
import math
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def Z_norm(X):
    X_mean=X.mean()
    X_std=np.std(np.array(X))
    return (X-X_mean)/X_std, X_mean, X_std

def reverse_Z_norm(Z, X_mean, X_std):
    X_original = Z * X_std + X_mean
    return X_original

class DayCent_Dataset1(Dataset):
    def __init__(self, module_name="Combined", task="t0", exp="temporal", fold=0):
        self.fln = 0
        self.sln = 0
        self.n_f = 0
        self.n_out = 0
        self.fn_ind = 0
        self.stat_ind = []
        self.flux_ind = []
        self.annual_feature_ind = []
        # self.X_feature_names, self.Y_feature_names = None, ["GPP", "N2O_FLUX", "ET_t", "Yield", "Delta_SOC"]

        # self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.load_data(module_name)
        # Scenarios = ["C_S_150N_Planting", "C_S_150N_30afPlanting", "S_C_150N_Planting", "S_C_150N_30afPlanting"]
        Scenarios = ["C_S_150N_Planting"]
        
        # self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test, self.x_not_scaled, self.y_not_scaled = self.combined_scenerios(module_name, Scenarios)
        if exp == "temporal":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.load_scaled_data(module_name=module_name, task=task)
        elif exp == "spatial":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test= self.load_scaled_data_spatial(module_name=module_name, task=task, fold=fold)
        else:
            raise ValueError("exp must be either 'temporal' or 'spatial'")
    
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return 
    
    def load_scaled_data(self, module_name="All", task="t0"):
        X = np.load(f'data_files/{task}_dc/X_dc_scaled.npy')
        if module_name == "All":
            Y_carbon = torch.from_numpy(np.load(f'data_files/{task}_dc/Y_dc_scaled_Carbon.npy'))
            Y_thermal = torch.from_numpy(np.load(f'data_files/{task}_dc/Y_dc_scaled_Thermal.npy'))
            Y_water = torch.from_numpy(np.load(f'data_files/{task}_dc/Y_dc_scaled_Water.npy'))
            Y_nitrogen = torch.from_numpy(np.load(f'data_files/{task}_dc/Y_dc_scaled_Nitrogen.npy'))
            Y = torch.cat((Y_carbon, Y_thermal, Y_water, Y_nitrogen), dim=-1)
        else:
            Y = np.load(f'data_files/{task}_dc/Y_dc_scaled_{module_name}.npy')
            Y=torch.from_numpy(Y)
        X=torch.from_numpy(X)
        print(f"loaded size {X.shape, Y.shape}")
        start_year = 2000
        end_year = 2020
        days_per_year = 365

        # Reshape the data to have a year dimension
        
        num_years = end_year - start_year + 1
        X_yearly = X.view(num_years, days_per_year, X.shape[1],  X.shape[2])
        Y_yearly = Y.view(num_years, days_per_year, Y.shape[1],  Y.shape[2])

        # Split the dataset by year for train, validation, and test
        train_years = int(num_years * 0.8)
        val_years = int(num_years * 0.1)
        test_years = num_years - train_years - val_years

        X_train = X_yearly[:train_years].contiguous().view(-1, X.shape[1], X.shape[2])
        X_val = X_yearly[train_years:train_years + val_years].contiguous().view(-1, X.shape[1], X.shape[2])
        X_test = X_yearly[train_years + val_years:].contiguous().view(-1, X.shape[1], X.shape[2])

        Y_train = Y_yearly[:train_years].contiguous().view(-1, Y.shape[1], Y.shape[2])
        Y_val = Y_yearly[train_years:train_years + val_years].contiguous().view(-1, Y.shape[1], Y.shape[2])
        Y_test = Y_yearly[train_years + val_years:].contiguous().view(-1, Y.shape[1], Y.shape[2])
        print("step 1: ", X_train.size(), X_val.size(), X_test.size(), Y_train.size(), Y_val.size(), Y_test.size())
        X_train = X_train.permute(1, 0, 2).contiguous().view(-1, days_per_year, X.shape[2])
        X_val = X_val.permute(1, 0, 2).contiguous().view(-1, days_per_year, X.shape[2])
        X_test = X_test.permute(1, 0, 2).contiguous().view(-1, days_per_year, X.shape[2])
        Y_train = Y_train.permute(1, 0, 2).contiguous().view(-1, days_per_year, Y.shape[2])
        Y_val = Y_val.permute(1, 0, 2).contiguous().view(-1, days_per_year, Y.shape[2])
        Y_test = Y_test.permute(1, 0, 2).contiguous().view(-1, days_per_year, Y.shape[2])
        print("step 2: ", X_train.size(), X_val.size(), X_test.size(), Y_train.size(), Y_val.size(), Y_test.size())
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def load_scaled_data_spatial(self, module_name="Combined", task="t0", fold=0):
        X = np.load(f'data_files/{task}_dc/X_dc_scaled.npy')
        if module_name == "All":
            Y_carbon = torch.from_numpy(np.load(f'data_files/{task}_dc/Y_dc_scaled_Carbon.npy'))
            Y_thermal = torch.from_numpy(np.load(f'data_files/{task}_dc/Y_dc_scaled_Thermal.npy'))
            Y_water = torch.from_numpy(np.load(f'data_files/{task}_dc/Y_dc_scaled_Water.npy'))
            Y_nitrogen = torch.from_numpy(np.load(f'data_files/{task}_dc/Y_dc_scaled_Nitrogen.npy'))
            Y = torch.cat((Y_carbon, Y_thermal, Y_water, Y_nitrogen), dim=-1)
        else:
            Y = np.load(f'data_files/{task}_dc/Y_dc_scaled_{module_name}.npy')
            Y=torch.from_numpy(Y)
        X=torch.from_numpy(X)
        print(f"loaded size {X.shape, Y.shape}")

        num_sites = 99
        exp_per_site = 20
        # Reshape the data to have a year dimension
        
        X_sitewise = X.view(X.shape[0], num_sites, exp_per_site,  X.shape[2])
        Y_sitewise = Y.view(Y.shape[0], num_sites, exp_per_site,  Y.shape[2])

        # Split the dataset by year for train, validation, and test
        fold_size = num_sites // 5
        indices = list(range(num_sites))
        val_start = fold * fold_size
        val_end = val_start + fold_size
        test_start = (fold + 1) % 5 * fold_size
        test_end = test_start + fold_size

        val_indices = indices[val_start:val_end]
        test_indices = indices[test_start:test_end]
        train_indices = [i for i in indices if i not in val_indices and i not in test_indices]

        X_train = X_sitewise[:, train_indices, :, :].contiguous().view(X.shape[0], -1, X.shape[2])
        X_val = X_sitewise[:, val_indices, :, :].contiguous().view(X.shape[0], -1, X.shape[2])
        X_test = X_sitewise[:, test_indices, :, :].contiguous().view(X.shape[0], -1, X.shape[2])

        Y_train = Y_sitewise[:, train_indices, :, :].contiguous().view(Y.shape[0], -1, Y.shape[2])
        Y_val = Y_sitewise[:, val_indices, :, :].contiguous().view(Y.shape[0], -1, Y.shape[2])
        Y_test = Y_sitewise[:, test_indices, :, :].contiguous().view(Y.shape[0], -1, Y.shape[2])

        X_train = X_train.permute(1, 0, 2).contiguous().view(-1, 365, X.shape[2])
        X_val = X_val.permute(1, 0, 2).contiguous().view(-1, 365, X.shape[2])
        X_test = X_test.permute(1, 0, 2).contiguous().view(-1, 365, X.shape[2])
        Y_train = Y_train.permute(1, 0, 2).contiguous().view(-1, 365, Y.shape[2])
        Y_val = Y_val.permute(1, 0, 2).contiguous().view(-1, 365, Y.shape[2])
        Y_test = Y_test.permute(1, 0, 2).contiguous().view(-1, 365, Y.shape[2])

        print(X_train.size(), X_val.size(), X_test.size(), Y_train.size(), Y_val.size(), Y_test.size())
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def combined_scenerios(self, module_name, Scenarios):
        X_list, Y_list = [], []
        for s in Scenarios:
            print(s)
            print(module_name)
            tx, ty = self.load_data(module_name, Scenario=s)
            X_list.append(tx)
            Y_list.append(ty)
            print(tx.size(), ty.size())
        X = torch.cat(X_list, dim=1)
        Y = torch.cat(Y_list, dim=1)
        print(X.size(), Y.size())

        # Apply Z norm
        Xscaler=np.zeros([len(self.X_feature_names),2])
        Yscaler=np.zeros([len(self.Y_feature_names),2])

        x_not_scaled = X
        y_not_scaled = Y
        for i in range(len(self.X_feature_names)):
            X[:,:,i],Xscaler[i,0],Xscaler[i,1]=Z_norm(X[:,:,i])
        for i in range(len(self.Y_feature_names)):
            Y[:,:,i],Yscaler[i,0],Yscaler[i,1]=Z_norm(Y[:,:,i])
        self.x_not_scaled1 = X
        self.y_not_scaled1 = Y
         # Constants
        start_year = 2000
        end_year = 2020
        days_per_year = 365
        X_feature_names = self.X_feature_names
        Y_feature_names = self.Y_feature_names
        # Reshape the data to have a year dimension
        
        num_years = end_year - start_year + 1
        print(X_feature_names)
        print(X.size())
        X_yearly = X.view(num_years, days_per_year, X.shape[1], len(X_feature_names))
        Y_yearly = Y.view(num_years, days_per_year, X.shape[1], len(Y_feature_names))

        # Split the dataset by year for train, validation, and test
        train_years = int(num_years * 0.8)
        val_years = int(num_years * 0.1)
        test_years = num_years - train_years - val_years

        X_train = X_yearly[:train_years].contiguous().view(-1, X.shape[1], len(X_feature_names))
        X_val = X_yearly[train_years:train_years + val_years].contiguous().view(-1, X.shape[1], len(X_feature_names))
        X_test = X_yearly[train_years + val_years:].contiguous().view(-1, X.shape[1], len(X_feature_names))

        Y_train = Y_yearly[:train_years].contiguous().view(-1, X.shape[1], len(Y_feature_names))
        Y_val = Y_yearly[train_years:train_years + val_years].contiguous().view(-1, X.shape[1], len(Y_feature_names))
        Y_test = Y_yearly[train_years + val_years:].contiguous().view(-1, X.shape[1], len(Y_feature_names))
        print(X_train.size(), X_val.size(), X_test.size(), Y_train.size(), Y_val.size(), Y_test.size())

        return X_train, X_val, X_test, Y_train, Y_val, Y_test, x_not_scaled, y_not_scaled


class MW_Dataset1(Dataset):
    def __init__(self, module_name="Combined", task="t0", exp="temporal", fold=0):
        self.fln = 0
        self.sln = 0
        self.n_f = 0
        self.n_out = 0
        self.fn_ind = 0
        # self.stat_ind = []
        # self.flux_ind = []
        # self.X_feature_names, self.Y_feature_names = None, None
        # out_names=["GPP", "N2O_FLUX", "ET_t"]
        # out_names_annual = ["Yield", "Delta_SOC"]
        self.x_not_scaled, self.y_not_scaled = None, None
        # self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.load_data(module_name)
        if exp == "temporal":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.load_scaled_data(module_name=module_name, task=task)
        elif exp == "spatial":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test= self.load_scaled_data_spatial(module_name=module_name, task=task, fold=fold)
        else:
            raise ValueError("exp must be either 'temporal' or 'spatial'")
        
    
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return 
    
    def load_scaled_data(self, module_name="Combined", task="t0"):
    
        X = np.load(f'data_files/{task}_mw/X_mw_scaled.npy')
        if module_name == "All":
            Y_carbon = torch.from_numpy(np.load(f'data_files/{task}_mw/Y_mw_scaled_Carbon.npy'))
            Y_thermal = torch.from_numpy(np.load(f'data_files/{task}_mw/Y_mw_scaled_Thermal.npy'))
            Y_water = torch.from_numpy(np.load(f'data_files/{task}_mw/Y_mw_scaled_Water.npy'))
            Y_nitrogen = torch.from_numpy(np.load(f'data_files/{task}_mw/Y_mw_scaled_Nitrogen.npy'))
            Y = torch.cat((Y_carbon, Y_thermal, Y_water, Y_nitrogen), dim=-1)
        else:
            Y = np.load(f'data_files/{task}_mw/Y_mw_scaled_{module_name}.npy')
            Y=torch.from_numpy(Y)
        X=torch.from_numpy(X).float()
        print(f"loaded size {X.shape, Y.shape}")
        start=1
        end=18
        days_per_year = 365

        # Reshape the data to have a year dimension
        
        num_years = end - start + 1  # Number of years in the dataset
        X_yearly = X.view(num_years, days_per_year, X.shape[1],  X.shape[2])
        Y_yearly = Y.view(num_years, days_per_year, Y.shape[1],  Y.shape[2])

        # Split the dataset by year for train, validation, and test
        train_years = int(num_years * 0.8)
        val_years = int(num_years * 0.1)
        test_years = num_years - train_years - val_years

        X_train = X_yearly[:train_years].contiguous().view(-1, X.shape[1], X.shape[2])
        X_val = X_yearly[train_years:train_years + val_years].contiguous().view(-1, X.shape[1], X.shape[2])
        X_test = X_yearly[train_years + val_years:].contiguous().view(-1, X.shape[1], X.shape[2])

        Y_train = Y_yearly[:train_years].contiguous().view(-1, Y.shape[1], Y.shape[2])
        Y_val = Y_yearly[train_years:train_years + val_years].contiguous().view(-1, Y.shape[1], Y.shape[2])
        Y_test = Y_yearly[train_years + val_years:].contiguous().view(-1, Y.shape[1], Y.shape[2])
        print("step 1: ", X_train.size(), X_val.size(), X_test.size(), Y_train.size(), Y_val.size(), Y_test.size())
        X_train = X_train.permute(1, 0, 2).contiguous().view(-1, days_per_year, X.shape[2])
        X_val = X_val.permute(1, 0, 2).contiguous().view(-1, days_per_year, X.shape[2])
        X_test = X_test.permute(1, 0, 2).contiguous().view(-1, days_per_year, X.shape[2])
        Y_train = Y_train.permute(1, 0, 2).contiguous().view(-1, days_per_year, Y.shape[2])
        Y_val = Y_val.permute(1, 0, 2).contiguous().view(-1, days_per_year, Y.shape[2])
        Y_test = Y_test.permute(1, 0, 2).contiguous().view(-1, days_per_year, Y.shape[2])
        print("step 2: ", X_train.size(), X_val.size(), X_test.size(), Y_train.size(), Y_val.size(), Y_test.size())
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def load_scaled_data_spatial(self, module_name="Combined", task="t0", fold=0):
        X = np.load(f'data_files/{task}_mw/X_mw_scaled.npy')
        if module_name == "All":
            Y_carbon = torch.from_numpy(np.load(f'data_files/{task}_mw/Y_mw_scaled_Carbon.npy'))
            Y_thermal = torch.from_numpy(np.load(f'data_files/{task}_mw/Y_mw_scaled_Thermal.npy'))
            Y_water = torch.from_numpy(np.load(f'data_files/{task}_mw/Y_mw_scaled_Water.npy'))
            Y_nitrogen = torch.from_numpy(np.load(f'data_files/{task}_mw/Y_mw_scaled_Nitrogen.npy'))
            Y = torch.cat((Y_carbon, Y_thermal, Y_water, Y_nitrogen), dim=-1)
        else:
            Y = np.load(f'data_files/{task}_mw/Y_mw_scaled_{module_name}.npy')
            Y=torch.from_numpy(Y)
        X=torch.from_numpy(X)
        print(f"loaded size {X.shape, Y.shape}")

        num_sites = 99
        exp_per_site = 20
        # Reshape the data to have a year dimension
        
        X_sitewise = X.view(X.shape[0], num_sites, exp_per_site,  X.shape[2])
        Y_sitewise = Y.view(Y.shape[0], num_sites, exp_per_site,  Y.shape[2])

        # Split the dataset by year for train, validation, and test
        fold_size = num_sites // 5
        indices = list(range(num_sites))
        val_start = fold * fold_size
        val_end = val_start + fold_size
        test_start = (fold + 1) % 5 * fold_size
        test_end = test_start + fold_size

        val_indices = indices[val_start:val_end]
        test_indices = indices[test_start:test_end]
        train_indices = [i for i in indices if i not in val_indices and i not in test_indices]

        X_train = X_sitewise[:, train_indices, :, :].contiguous().view(X.shape[0], -1, X.shape[2])
        X_val = X_sitewise[:, val_indices, :, :].contiguous().view(X.shape[0], -1, X.shape[2])
        X_test = X_sitewise[:, test_indices, :, :].contiguous().view(X.shape[0], -1, X.shape[2])

        Y_train = Y_sitewise[:, train_indices, :, :].contiguous().view(Y.shape[0], -1, Y.shape[2])
        Y_val = Y_sitewise[:, val_indices, :, :].contiguous().view(Y.shape[0], -1, Y.shape[2])
        Y_test = Y_sitewise[:, test_indices, :, :].contiguous().view(Y.shape[0], -1, Y.shape[2])

        X_train = X_train.permute(1, 0, 2).contiguous().view(-1, 365, X.shape[2])
        X_val = X_val.permute(1, 0, 2).contiguous().view(-1, 365, X.shape[2])
        X_test = X_test.permute(1, 0, 2).contiguous().view(-1, 365, X.shape[2])
        Y_train = Y_train.permute(1, 0, 2).contiguous().view(-1, 365, Y.shape[2])
        Y_val = Y_val.permute(1, 0, 2).contiguous().view(-1, 365, Y.shape[2])
        Y_test = Y_test.permute(1, 0, 2).contiguous().view(-1, 365, Y.shape[2])

        print(X_train.size(), X_val.size(), X_test.size(), Y_train.size(), Y_val.size(), Y_test.size())
        return X_train, X_val, X_test, Y_train, Y_val, Y_test


class t1_n2o_Dataset1(Dataset):
    def __init__(self, module_name="Combined", task="t0", exp="temporal", fold=0):
        self.fln = 0
        self.sln = 0
        self.n_f = 0
        self.n_out = 0
        self.fn_ind = 0
        # self.stat_ind = []
        # self.flux_ind = []
        # self.X_feature_names, self.Y_feature_names = None, None
        # out_names=["GPP", "N2O_FLUX", "ET_t"]
        # out_names_annual = ["Yield", "Delta_SOC"]
        self.x_not_scaled, self.y_not_scaled = None, None
        # self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.load_data(module_name)
        if exp == "temporal":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.load_scaled_data(module_name=module_name, task=task)
        elif exp == "spatial":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test= self.load_scaled_data_spatial(module_name=module_name, task=task, fold=fold)
        else:
            raise ValueError("exp must be either 'temporal' or 'spatial'")
        
    
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return 
    
    def load_scaled_data(self, module_name="Combined", task="t0"):
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        X_test = []
        Y_test = []
        for chamber in range(6):
            X_load = np.load(f"data_files/t1_n2o/X_real_scaled_{chamber}.npy")
            Y_load = np.load(f"data_files/t1_n2o/Y_real_scaled_Combined_{chamber}.npy")

            X_load = torch.from_numpy(X_load)
            Y_load = torch.from_numpy(Y_load)

            X_train.append(X_load[:, 0:2, :])
            Y_train.append(Y_load[:, 0:2, :])
            X_val.append(X_load[:, 1, :].unsqueeze(dim=-1))
            Y_val.append(Y_load[:, 1, :].unsqueeze(dim=-1))
            X_test.append(X_load[:, 2, :].unsqueeze(dim=-1))
            Y_test.append(Y_load[:, 2, :].unsqueeze(dim=-1))

        X_train = torch.cat(X_train, dim=1).permute(1, 0, 2).float()
        Y_train = torch.cat(Y_train, dim=1).permute(1, 0, 2).float()
        X_val = torch.cat(X_val, dim=-1).permute(2, 0, 1).float()
        Y_val = torch.cat(Y_val, dim=-1).permute(2, 0, 1).float()
        X_test = torch.cat(X_test, dim=-1).permute(2, 0, 1).float()
        Y_test = torch.cat(Y_test, dim=-1).permute(2, 0, 1).float()
        print(X_train.size(), X_val.size(), X_test.size(), Y_train.size(), Y_val.size(), Y_test.size())
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def load_scaled_data_spatial(self, module_name="Combined", task="t0", fold=0):
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        X_test = []
        Y_test = []

        for chamber in range(6):
            X_load = np.load(f"data_files/t1_n2o/X_real_scaled_{chamber}.npy")
            Y_load = np.load(f"data_files/t1_n2o/Y_real_scaled_Combined_{chamber}.npy")

            X_load = torch.from_numpy(X_load)
            Y_load = torch.from_numpy(Y_load)
        
            # Determine the sets based on the fold
            if chamber == fold:  # Validation set
                X_val.append(X_load)
                Y_val.append(Y_load)
            elif chamber == (fold + 1) % 6:  # Test set
                X_test.append(X_load)
                Y_test.append(Y_load)
            else:  # Training set
                X_train.append(X_load)
                Y_train.append(Y_load)
        # Concatenate data from all chambers
        X_train = torch.cat(X_train, dim=1).float().permute(1, 0, 2)
        Y_train = torch.cat(Y_train, dim=1).float().permute(1, 0, 2)
        X_val = torch.cat(X_val, dim=1).float().permute(1, 0, 2)
        Y_val = torch.cat(Y_val, dim=1).float().permute(1, 0, 2)
        X_test = torch.cat(X_test, dim=1).float().permute(1, 0, 2)
        Y_test = torch.cat(Y_test, dim=1).float().permute(1, 0, 2)
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test

class t1_co2_Dataset1(Dataset):
    def __init__(self, module_name="Combined", task="t0", exp="temporal", fold=0):
        self.fln = 0
        self.sln = 0
        self.n_f = 0
        self.n_out = 0
        self.fn_ind = 0
        # self.stat_ind = []
        # self.flux_ind = []
        # self.X_feature_names, self.Y_feature_names = None, None
        # out_names=["GPP", "N2O_FLUX", "ET_t"]
        # out_names_annual = ["Yield", "Delta_SOC"]
        self.x_not_scaled, self.y_not_scaled = None, None
        # self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.load_data(module_name)
        if exp == "temporal":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.load_scaled_data(module_name=module_name, task=task)
        elif exp == "spatial":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test= self.load_scaled_data_spatial(module_name=module_name, task=task, fold=fold)
        else:
            raise ValueError("exp must be either 'temporal' or 'spatial'")
        
    
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return 
    
    def load_scaled_data(self, module_name="Combined", task="t0"):
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        X_test = []
        Y_test = []
        for chamber in range(11):
            X_load = np.load(f'data_files/t1_co2/X_real_scaled_ECfluxnet_{chamber}.npy')
            Y_load = np.load(f'data_files/t1_co2/Y_real_scaled_ECfluxnet_Combined_{chamber}.npy')
            
            X_load = torch.from_numpy(X_load)
            Y_load = torch.from_numpy(Y_load)

            n_years = X_load.shape[0] // 365
            train_years = int(n_years * 0.8)
            val_years = int(n_years * 0.1)
            test_years = n_years - train_years - val_years

            # Split the dataset by year for train, validation, and test
            X_train.append(X_load[:train_years * 365, :, :].view(-1, 365, X_load.shape[2]))
            Y_train.append(Y_load[:train_years * 365, :, :].view(-1, 365, Y_load.shape[2]))
            if val_years > 0:
                X_val.append(X_load[train_years * 365:train_years * 365 + val_years * 365, :, :].view(-1, 365, X_load.shape[2]))
                Y_val.append(Y_load[train_years * 365:train_years * 365 + val_years * 365, :, :].view(-1, 365, Y_load.shape[2]))
            X_test.append(X_load[train_years * 365 + val_years * 365:, :, :].view(-1, 365, X_load.shape[2]))
            Y_test.append(Y_load[train_years * 365 + val_years * 365:, :, :].view(-1, 365, Y_load.shape[2]))
            print(n_years, train_years, val_years, test_years)

        X_train = torch.cat(X_train, dim=0).float()
        Y_train = torch.cat(Y_train, dim=0).float()
        X_val = torch.cat(X_val, dim=0).float()
        Y_val = torch.cat(Y_val, dim=0).float()
        X_test = torch.cat(X_test, dim=0).float()
        Y_test = torch.cat(Y_test, dim=0).float()
        print(X_train.size(), X_val.size(), X_test.size(), Y_train.size(), Y_val.size(), Y_test.size())

        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    def load_scaled_data_spatial(self, module_name="Combined", task="t0", fold=0):
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        X_test = []
        Y_test = []
        for chamber in range(11):
            X_load = np.load(f'data_files/t1_co2/X_real_scaled_ECfluxnet_{chamber}.npy')
            Y_load = np.load(f'data_files/t1_co2/Y_real_scaled_ECfluxnet_Combined_{chamber}.npy')
            
            X_load = torch.from_numpy(X_load)
            Y_load = torch.from_numpy(Y_load)

            print(X_load.shape, Y_load.shape)
            # Determine the sets based on the fold
            if chamber == fold:  # Validation set
                X_val.append(X_load.view(-1, 365, X_load.shape[2]))
                Y_val.append(Y_load.view(-1, 365, Y_load.shape[2]))
            elif chamber == (fold + 1) % 11:  # Test set
                X_test.append(X_load.view(-1, 365, X_load.shape[2]))
                Y_test.append(Y_load.view(-1, 365, Y_load.shape[2]))
            else:  # Training set
                X_train.append(X_load.view(-1, 365, X_load.shape[2]))
                Y_train.append(Y_load.view(-1, 365, Y_load.shape[2]))

        X_train = torch.cat(X_train, dim=0).float()
        Y_train = torch.cat(Y_train, dim=0).float()
        X_val = torch.cat(X_val, dim=0).float()
        Y_val = torch.cat(Y_val, dim=0).float()
        X_test = torch.cat(X_test, dim=0).float()
        Y_test = torch.cat(Y_test, dim=0).float()
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
