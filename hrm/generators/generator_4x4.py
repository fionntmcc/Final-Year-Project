import numpy as np
import random
from typing import Tuple


def generate_puzzle(num_clues: int = 10, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 4x4 Sudoku puzzle
    
    Args:
        num_clues: Number of given cells (10 = easy)
        seed: Random seed for reproducibility
        
    Returns:
        puzzle: 4x4 array with 0 for empty cells
        solution: 4x4 array with complete solution
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate complete board
    board = np.zeros((4, 4), dtype=int)
    fill_board(board)
    solution = board.copy()
    
    # Remove cells to create puzzle
    positions = [(i, j) for i in range(4) for j in range(4)]
    random.shuffle(positions)
    
    for i in range(16 - num_clues):
        row, col = positions[i]
        board[row, col] = 0
    
    return board, solution


def fill_board(board: np.ndarray) -> bool:
    """Fill board using backtracking"""
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                digits = list(range(1, 5))
                random.shuffle(digits)
                
                for digit in digits:
                    if is_valid(board, i, j, digit):
                        board[i, j] = digit
                        if fill_board(board):
                            return True
                        board[i, j] = 0
                return False
    return True


def is_valid(board: np.ndarray, row: int, col: int, digit: int) -> bool:
    """Check if placing digit is valid"""
    # Check row, column, box
    if digit in board[row, :]:
        return False
    if digit in board[:, col]:
        return False
    
    box_row, box_col = 2 * (row // 2), 2 * (col // 2)
    if digit in board[box_row:box_row+2, box_col:box_col+2]:
        return False
    
    return True