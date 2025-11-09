"""
Decoder module for the HRM model.
Ensures actions from the Worker are translated into valid updates on the board.

Each Worker action is validated against Sudoku rules before being applied.
"""

class Decoder:
    """
    Decoder class to translate Worker actions into board updates.
    
    config: Configuration dictionary for the decoder.
    Will contain parameters like board size, value range, etc.
    """
    
    board_size = 9  # Default board size
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
        # Check bounds
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise IndexError("Cell position out of bounds")
        
        # Check if cell is empty
        if board[row, col] != 0:
            return False
        
        # ----- Check Sudoku rules -----
        # Valid value check
        if value < 1 or value > 9:
            return False
        # Row and column check
        if value in board[row, :] or value in board[:, col]:
            return False
        block_row, block_col = (row // 3) * 3, (col // 3) * 3
        # Block check
        if value in board[block_row:block_row+3, block_col:block_col+3]:
            return False
        return True