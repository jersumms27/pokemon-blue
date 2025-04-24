from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DQN(nn.Module):
    def __init__(self, state_size: int,
                 num_actions: int,
                 hidden_sizes: list[int]=[],
                 nonlinearity: Callable[[Tensor], Tensor]=F.relu) -> None:
        super().__init__()

        if not hidden_sizes:
            hidden_sizes = [state_size]
        self.nonlinearity: Callable[[Tensor], Tensor] = nonlinearity

        self.input_layer: nn.Linear = nn.Linear(state_size, hidden_sizes[0])
        self.hidden_layers: nn.ModuleList = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer: nn.Linear = nn.Linear(hidden_sizes[-1], num_actions)
    
    
    def forward(self, state: Tensor) -> Tensor:
        output: Tensor = self.nonlinearity(self.input_layer(state))

        for hidden_layer in self.hidden_layers:
            output = self.nonlinearity(hidden_layer(output))
        
        q_values: Tensor = self.output_layer(output)
        return q_values
