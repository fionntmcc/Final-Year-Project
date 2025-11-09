"""
Decoder module for the HRM model.
Ensures actions from the Worker are translated into valid updates on the board.

Each Worker action is validated against Sudoku rules before being applied.
"""

import torch
import numpy as np

class Decoder:
    """
    Decoder class to translate Worker actions into board updates.
    
    config: Configuration dictionary for the decoder.
    Will contain parameters like board size, value range, etc.
    """
    def __init__(self, config):
        self.config = config

    """
    Parses Worker action and ensures legality before updating the board.
    """
    def decode(self, action, board):
        # Validate action (e.g., check if value is valid in row/col/block)
        row, col = action["cell"] # Location of update
        value = action["value"] # Value to place
        if self.is_valid_move(board, row, col, value):
            board[row, col] = value # Update board
        return board 

    """
    Checks if placing 'value' at (row, col) is legal according to Sudoku rules.
    """
    def is_valid_move(self, board, row, col, value):
        # Check Sudoku rules
        if value in board[row, :] or value in board[:, col]:
            return False
        block_row, block_col = (row // 3) * 3, (col // 3) * 3
        if value in board[block_row:block_row+3, block_col:block_col+3]:
            return False
        return True