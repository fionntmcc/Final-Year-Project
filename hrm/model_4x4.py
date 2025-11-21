import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class HRM_4x4(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # Embedding layer for puzzle state (0-4)
        self.embed = nn.Embedding(5, 16)
        
        # Simple feedforward layers for planner. This needs to be much more complex to solve 9x9 Sudoku grid.
        # This should work for a 4x4 grid.
        self.planner = nn.Sequential(
            nn.Linear(16 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # High-level planner: puzzle -> abstract representation
        # Linear applies correlations across all 16 cells
        # ReLU applies non-linearity, allowing for "if-then" reasoning (needed for Sudoku rules)
        # Linear reduces the reasoning to a final state that the decoder can use to make a prediction
        self.worker = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        