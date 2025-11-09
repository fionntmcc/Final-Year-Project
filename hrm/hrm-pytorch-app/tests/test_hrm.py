import numpy as np
import pytest
from src.models.decoder import Decoder

@pytest.fixture
def decoder():
    return Decoder({'board_size': 9}) # May implement size config later...

# Test empty board update
def test_valid_move_updates_board(decoder):
    board = np.zeros((9, 9), dtype=int)
    action = {"cell": (0, 0), "value": 5}
    updated = decoder.decode(action, board)
    assert updated[0, 0] == 5
    assert np.array_equal(updated, board)

# Test invalid row move
def test_invalid_move_row_conflict(decoder):
    board = np.array([[5, 0, 0], [0, 0, 0], [0, 0, 0]])  # 5 in row 0
    action = {"cell": (0, 1), "value": 5}  # Conflict in row
    updated = decoder.decode(action, board)
    assert updated[0, 1] == 0  # No update
    assert np.array_equal(updated, board)

# Test invalid column move
def test_invalid_move_column_conflict(decoder):
    board = np.array([[5], [0], [0]])  # 5 in col 0
    action = {"cell": (1, 0), "value": 5}  # Conflict in col
    updated = decoder.decode(action, board)
    assert updated[1, 0] == 0


# Test invalid block move
def test_invalid_move_block_conflict(decoder):
    board = np.zeros((9, 9), dtype=int)
    board[0, 0] = 5  # In top-left 3x3 block
    action = {"cell": (1, 1), "value": 5}  # Same block
    updated = decoder.decode(action, board)
    assert updated[1, 1] == 0

# Test above bounds cell
def test_edge_case_above_bounds_cell(decoder):
    board = np.zeros((9, 9), dtype=int)
    action = {"cell": (10, 10), "value": 5}  # Out of bounds
    with pytest.raises(IndexError):
        decoder.decode(action, board)
        
# Test below bounds cell
def test_edge_case_below_bounds_cell(decoder):
    board = np.zeros((9, 9), dtype=int)
    action = {"cell": (-1, -1), "value": 5}  # Out of bounds
    with pytest.raises(IndexError):
        decoder.decode(action, board)

def test_edge_case_above_bounds_value(decoder):
    board = np.zeros((9, 9), dtype=int)
    action = {"cell": (0, 0), "value": 10}  # Value > 9
    updated = decoder.decode(action, board)
    assert updated[0, 0] == 0  # No update (assuming 1-9 range)
    
def test_edge_case_below_bounds_value(decoder):
    board = np.zeros((9, 9), dtype=int)
    action = {"cell": (0, 0), "value": -1}  # Value < 1
    updated = decoder.decode(action, board)
    assert updated[0, 0] == 0  # No update (assuming 1-9 range)

def test_decoder_on_small_board(decoder):
    board = np.array([[1, 0], [0, 0]]) # Trivial 2x2 board
    initial_unsolved = np.sum(board == 0)  # Calculate before decode
    action = {"cell": (0, 1), "value": 2}
    updated = decoder.decode(action, board)
    assert updated[0, 1] == 2
    final_unsolved = np.sum(updated == 0)
    assert final_unsolved < initial_unsolved  # Now 2 < 3