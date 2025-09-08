from torch.utils.data import Dataset
import os
import h5py
import torch


# Custom Dataset class
# Combined multiple h5 files into a Dataset
class CombinedH5Dataset_DoubleHeads(Dataset):
    def __init__(self, file_paths):
        """
        Initializes the dataset by loading the .h5 file.

        Parameters:
        - file_path (str): Path to the .h5 file.
        """
        self.file_paths = file_paths
        self.data = self._load_data_from_files(file_paths)

    def _load_data_from_files(self, file_paths):
        data = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            with h5py.File(file_path, 'r') as f:
                # Assuming the keys 'geometry', 'inputs', and 'outputs' are present in each H5 file
                geometry = f['geometry'][:]
                inputs = f['inputs'][:]
                outputs = f['outputs'][:]

                outputs_head1 = f['outputs'][:, 0:11]
                outputs_head2 = f['outputs'][:, 11:22]

                # Convert to torch.float32 for consistency in training
                geometry = torch.tensor(geometry, dtype=torch.float32)
                inputs = torch.tensor(inputs, dtype=torch.float32)
                outputs = torch.tensor(outputs, dtype=torch.float32)
                outputs_head1 = torch.tensor(outputs_head1, dtype=torch.float32)
                outputs_head2 = torch.tensor(outputs_head2, dtype=torch.float32)

                # Combine the data in a way that fits your model input
                combined_inputs = torch.cat((geometry, inputs), dim=-1)

                # In this case, geometry and inputs are inputs, and outputs are the target labels
                for i in range(len(inputs)):
                    data.append((combined_inputs[i], outputs_head1[i], outputs_head2[i]))

        return data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: (input_tensor, output_tensor)
        """
        combined_inputs, output_head1, output_head2 = self.data[idx]


        return combined_inputs, output_head1, output_head2

class CombinedH5Dataset(Dataset):
    def __init__(self, file_paths):
        """
        Initializes the dataset by loading the .h5 file.

        Parameters:
        - file_path (str): Path to the .h5 file.
        """
        self.file_paths = file_paths
        self.data = self._load_data_from_files(file_paths)

    def _load_data_from_files(self, file_paths):
        data = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            with h5py.File(file_path, 'r') as f:
                # Assuming the keys 'geometry', 'inputs', and 'outputs' are present in each H5 file
                geometry = f['geometry'][:]
                inputs = f['inputs'][:]
                outputs = f['outputs'][:]

                # Convert to torch.float32 for consistency in training
                geometry = torch.tensor(geometry, dtype=torch.float32)
                inputs = torch.tensor(inputs, dtype=torch.float32)
                outputs = torch.tensor(outputs, dtype=torch.float32)

                # Combine geometry and inputs together
                # In this case, geometry and inputs are inputs
                combined_inputs = torch.cat((geometry, inputs), dim=-1)

                # outputs are the target labels
                for i in range(len(inputs)):
                    data.append((combined_inputs[i], outputs[i]))

        return data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: (input_tensor, output_tensor)
        """
        combined_inputs, outputs = self.data[idx]


        return combined_inputs, outputs


# Make Single h5 file into a Dataset
class H5Dataset_DoubleHeads(Dataset):
    def __init__(self, file_path):
        """
        Initializes the dataset by loading the .h5 file.

        Parameters:
        - file_path (str): Path to the .h5 file.
        """
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as h5_file:
            # Read the data to determine dataset size
            self.geometry = h5_file['geometry'][:]
            self.inputs = h5_file['inputs'][:]
            self.outputs = h5_file['outputs'][:]

            self.outputs_head1 = h5_file['outputs'][:, 0:11]
            self.outputs_head2 = h5_file['outputs'][:, 11:22]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.outputs)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: (input_tensor, output_tensor)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Combine geometry and inputs
        input_data = torch.tensor(self.geometry[idx], dtype=torch.float32)
        additional_input = torch.tensor(self.inputs[idx], dtype=torch.float32)
        combined_input = torch.cat((input_data, additional_input), dim=0)

        # Get outputs
        output_data = torch.tensor(self.outputs[idx], dtype=torch.float32)

        output_head1 = torch.tensor(self.outputs_head1[idx], dtype=torch.float32)
        output_head2 = torch.tensor(self.outputs_head2[idx], dtype=torch.float32)

        return combined_input, output_head1, output_head2

class H5Dataset(Dataset):
    def __init__(self, file_path):
        """
        Initializes the dataset by loading the .h5 file.

        Parameters:
        - file_path (str): Path to the .h5 file.
        """
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as h5_file:
            # Read the data to determine dataset size
            self.geometry = h5_file['geometry'][:]
            self.inputs = h5_file['inputs'][:]
            self.outputs = h5_file['outputs'][:]


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.outputs)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: (input_tensor, output_tensor)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Combine geometry and inputs
        input_data = torch.tensor(self.geometry[idx], dtype=torch.float32)
        additional_input = torch.tensor(self.inputs[idx], dtype=torch.float32)
        combined_input = torch.cat((input_data, additional_input), dim=0)

        # Get outputs
        output_data = torch.tensor(self.outputs[idx], dtype=torch.float32)

        return combined_input, output_data