import torch
import torch.nn as nn


class BasicLinearNet(nn.Module):
    def __init__(self, num_hidden_units=128, T=24):
        super(BasicLinearNet, self).__init__()
        # Calculate the size of the input layer
        # 24 hours * (5 weather conditions + load) + 43 calendar variables + num_regions for one-hot encoded regions
        input_size = T * (4 + 4 + 1) + 7 + 12 + 31 + 7 + 24 + 3

        # The output layer size is 24 since we're predicting loads for the next 24 hours
        output_size = 24

        # Define the architecture
        self.layer1 = nn.Linear(input_size, num_hidden_units)
        self.layer2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.output_layer = nn.Linear(num_hidden_units, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


def build_basic_model() -> BasicLinearNet:
    return BasicLinearNet()
