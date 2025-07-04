from typing import Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DQN(nn.Module):
    def __init__(self,
                 state_size: int,
                 num_actions: int,
                 hidden_sizes: list[int] = [],
                 dropout: float = 0.15) -> None:
        super().__init__()

        if not hidden_sizes:
            hidden_sizes = [state_size]
        
        layers: list[Union[nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout]]  = []
        prev_size: int = state_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())

            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, num_actions))

        self.network: nn.Sequential = nn.Sequential(*layers)

        self._init_weights()
    

    def forward(self,
                state: Tensor) -> Tensor:
        return self.network(state)
    

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)