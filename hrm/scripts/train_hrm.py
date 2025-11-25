import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from model_hrm import HRM_Sudoku
from generator import generate_dataset, print_puzzle


class SudokuDataset(Dataset):
    def __init__(self, puzzles: np.ndarray, solutions: np.ndarray):
        self.puzzles = torch.from_numpy(puzzles).long()
        
        # Target: first empty cell and its digit
        self.targets = []
        for puzzle, solution in zip(puzzles, solutions):
            empty = np.argwhere(puzzle == 0)
            if len(empty) > 0:
                r, c = empty[0]
                cell_idx = r * 4 + c
                digit = solution[r, c] - 1  # 0-indexed
            else:
                cell_idx = 0
                digit = 0
            self.targets.append((cell_idx, digit))
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        return self.puzzles[idx], self.targets[idx][0], self.targets[idx][1]