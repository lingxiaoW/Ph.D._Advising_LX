from torch.utils.data import random_split, DataLoader
from DataSets import H5Dataset, CombinedH5Dataset, CombinedH5Dataset_DoubleHeads, H5Dataset_DoubleHeads


######
# Load the dataset
#####



# Function to split dataset into training and validation
def split_dataset(dataset, train_ratio=0.8):
    """
    Splits the dataset into training and validation sets.

    Parameters:
    - dataset (Dataset): The dataset to split.
    - train_ratio (float): Ratio of training data (default: 0.8).

    Returns:
    - train_dataset, val_dataset: Split datasets.
    """
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def load_data(train_location, test_location, model_selection):
    if model_selection == 'DoubleHead':
        dataset = CombinedH5Dataset_DoubleHeads(train_location)
        test_dataset = H5Dataset_DoubleHeads(test_location)
    else:
        dataset = CombinedH5Dataset(train_location)
        test_dataset = H5Dataset(test_location)

    # Split the dataset
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)

    # Create DataLoaders for training and validation
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Check the splits
    print(f"Total dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")

    # Iterate through DataLoader
    if model_selection == 'DoubleHead':
        for batch_idx, (inputs, outputs1, outputs2) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1} - Input shape: {inputs.shape}, "
                  f"Output 1 shape: {outputs1.shape}, "
                  f"Output 2 shape: {outputs2.shape}.")
            break  # Just check one batch
    else:
        for batch_idx, (inputs, outputs) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1} - Input shape: {inputs.shape}, "
                  f"Output shape: {outputs.shape}.")
            break  # Just check one batch

    return train_loader, val_loader, test_loader


