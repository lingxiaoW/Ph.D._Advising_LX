import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class DNNWithTwoHeads(nn.Module):
    def __init__(self, input_size):
        super(DNNWithTwoHeads, self).__init__()

        # Shared Backbone using nn.Sequential
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Two separate heads using nn.Sequential
        self.head1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        )

        self.head2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        )

    def forward(self, x):
        # Pass input through the shared backbone
        x = self.shared_backbone(x)

        # Generate outputs from both heads
        output1 = self.head1(x)
        output2 = self.head2(x)

        return output1, output2