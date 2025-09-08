import torch
import torch.nn as nn


# CNN Model
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_size, output_size, window_size):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=window_size, out_channels=256, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=2, stride=1)
        self.fc = nn.Linear(256 * (input_size - 2), 256)  # Adjust for kernel reduction
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        return x





