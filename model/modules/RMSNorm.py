import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.size = size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(size))

    def forward(self, x):
        
        # Calculate the RMS value for each sample in the batch and each feature across all other dimensions.
        norm_x = x.pow(2)
        norm_x = norm_x.mean(dim=-1, keepdim=True)
        norm_x = norm_x.add(self.eps)
        norm_x = norm_x.sqrt()
        return x / norm_x * self.gamma
