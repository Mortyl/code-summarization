import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAttn(nn.Module):
    """
    Placeholder class for now.
    """
    def __init__(self):
        super(ConvAttn, self).__init__()

        self.fc = nn.Linear(300, 25)

    def forward(self, x):

        x = x.float()
        return self.fc(x)