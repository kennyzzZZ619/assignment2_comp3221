import torch.nn as nn
import torch


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size=8, seed=False):
        if seed:
            torch.manual_seed(22)
        super(LinearRegressionModel, self).__init__()
        # Create a linear transformation to the incoming data
        self.linear = nn.Linear(input_size, 1)

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Apply linear transformation
        output = self.linear(x)
        return output.reshape(-1)
