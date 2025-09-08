import torch
import torch.nn as nn


# LSTM Model
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # Only the hidden state from the last time step is needed
        hn = hn[-1]  # Take the last layer's hidden state
        output = self.fc(hn)
        return output