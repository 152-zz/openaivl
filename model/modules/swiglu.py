import torch
from torch import nn
import torch.nn.functional as F

class FFNSWIGLU(nn.Module):
    def __init__(self, hidden_size: int, forward_expansion: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size*forward_expansion, bias=False)
        self.w2 = nn.Linear(hidden_size*forward_expansion, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, hidden_size*forward_expansion, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_size)
        # w1(x) -> (batch_size, seq_len, intermediate_size)
        # w1(x) -> (batch_size, seq_len, intermediate_size)
        # w2(*) -> (batch_size, seq_len, hidden_size)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
