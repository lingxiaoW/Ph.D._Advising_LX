import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic time-series data
def generate_data(num_samples=1000, timesteps=100, input_size=14, output_size=22):
    inputs = np.random.randn(num_samples, timesteps, input_size)
    outputs = np.random.randn(num_samples, timesteps, output_size)
    return inputs, outputs

# Dataset for time series input-output pairs
class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs, self.targets = [], []
        for sample_in, sample_out in zip(inputs, outputs):
            for i in range(2, len(sample_in)):  # Starting from t2
                self.inputs.append(sample_in[i-2:i+1])  # t0, t1, t2
                self.targets.append(sample_out[i])  # Output at t2
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# CNN Model
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2, stride=1)
        self.fc = nn.Linear(32 * (input_size - 2), output_size)  # Adjust for kernel reduction

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, epochs=20, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# Main script
if __name__ == "__main__":
    num_samples = 1000
    timesteps = 100
    input_size = 14
    output_size = 22

    # Generate synthetic data
    inputs, outputs = generate_data(num_samples, timesteps, input_size, output_size)

    # Train-test split
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(train_inputs, train_outputs)
    val_dataset = TimeSeriesDataset(val_inputs, val_outputs)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize and train model
    model = TimeSeriesCNN(input_size=input_size, output_size=output_size)
    train_model(model, train_loader, val_loader)
