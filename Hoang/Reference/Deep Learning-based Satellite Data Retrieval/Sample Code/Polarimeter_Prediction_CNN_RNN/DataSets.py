import torch
import torch.nn as nn
import torch.optim as optim
from astropy.utils.metadata.utils import dtype
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import h5py


# define the dataset
class Combined_TimeSeriesDataset(Dataset):
    def __init__(self, file_paths, window_size):
        """
        Initializes the dataset by loading the .h5 file.

        Parameters:
        - file_path (str): Path to the .h5 file.
        """
        self.file_paths = file_paths
        self.window_size = window_size
        self.inputs, self.targets = self._load_data_from_files(file_paths, window_size)


    def _load_data_from_files(self, file_paths, window_size):
        # data is the meta data that saves all inputs and outputs from multiple h5 files.
        data = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            with h5py.File(file_path, 'r') as f:
                # Assuming the keys 'geometry', 'inputs', and 'outputs' are present in each H5 file
                geometry = f['geometry'][:]
                inputs = f['inputs'][:]
                outputs = f['outputs'][:]

                # Combine geometry and inputs together
                # In this case, geometry and inputs are inputs
                combined_inputs = np.concatenate((geometry, inputs), axis=1)

                # outputs are the target labels
                for i in range(len(inputs)):
                    data.append((combined_inputs[i], outputs[i]))

        # inputs and targets are lists that save input-output pairs in the metadata
        inputs, targets = [], []
        for i in range(len(data)):
            sample_inputs, sample_targets = data[i]
            inputs.append(sample_inputs)
            targets.append(sample_targets)

        # time_combined_inputs and time_combined_targets are lists that save time_combined inputs and outputs for time-series prediction
        time_combined_inputs, time_combined_targets = [], []

        for i in range(window_size - 1, len(inputs)):
            time_combined_inputs.append(inputs[i - (window_size - 1) : i + 1])
            time_combined_targets.append(targets[i])

        # convert input / output into tensor
        time_combined_inputs = torch.tensor(time_combined_inputs, dtype=torch.float32)
        time_combined_targets = torch.tensor(time_combined_targets, dtype=torch.float32)

        return time_combined_inputs, time_combined_targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]



class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, window_size):
        """
        Initializes the dataset by loading the .h5 file.

        Parameters:
        - file_path (str): Path to the .h5 file.
        """
        self.file_path = file_path
        self.window_size = window_size
        self.inputs, self.targets = self._load_data_from_files(file_path, window_size)


    def _load_data_from_files(self, file_path, window_size):
        # time_combined_inputs and time_combined_targets are lists that save time_combined inputs and outputs for time-series prediction
        time_combined_inputs, time_combined_targets = [], []
        with h5py.File(file_path, 'r') as f:
            # Assuming the keys 'geometry', 'inputs', and 'outputs' are present in each H5 file
            geometry = f['geometry'][:]
            inputs = f['inputs'][:]
            outputs = f['outputs'][:]

            # Combine geometry and inputs together
            # In this case, geometry and inputs are inputs
            combined_inputs = np.concatenate((geometry, inputs), axis=1)


            for i in range(window_size - 1, len(combined_inputs)):
                time_combined_inputs.append(combined_inputs[i - (window_size - 1): i + 1])
                time_combined_targets.append(outputs[i])

        # convert input / output into tensor
        time_combined_inputs = torch.tensor(time_combined_inputs, dtype=torch.float32)
        time_combined_targets = torch.tensor(time_combined_targets, dtype=torch.float32)

        return time_combined_inputs, time_combined_targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]