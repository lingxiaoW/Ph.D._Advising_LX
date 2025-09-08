from torch.utils.data import random_split, DataLoader
from DataSets import Combined_TimeSeriesDataset, TimeSeriesDataset


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



def load_data(train_location, test_location, window_size):
    dataset = Combined_TimeSeriesDataset(train_location, window_size)
    test_dataset = TimeSeriesDataset(test_location, window_size)

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
    for batch_idx, (inputs, outputs) in enumerate(train_loader):
        print(f"Training Data - Batch {batch_idx + 1} - Input shape: {inputs.shape}, "
              f"Output shape: {outputs.shape}. ")
        break  # Just check one batch

    # Iterate through DataLoader
    for batch_idx, (inputs, outputs) in enumerate(test_loader):
        print(f"Testing Data - Batch {batch_idx + 1} - Input shape: {inputs.shape}, "
              f"Output shape: {outputs.shape}. ")
        break  # Just check one batch

    return train_loader, val_loader, test_loader