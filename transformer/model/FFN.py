import torch.nn as nn
import torch
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

