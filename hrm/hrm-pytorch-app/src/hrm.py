import torch
import numpy as np
import random

def calculate_residual(board):
    return np.sum(board == 0)  # Count unsolved cells

def is_solved(board):
    return calculate_residual(board) == 0

def fixed_point_loop(board, planner, worker, decoder, max_outer=10, max_inner=5, tolerance=1e-6):
    residual = float('inf')
    for outer_iter in range(max_outer):
        plan = planner.plan(board)
        for inner_iter in range(max_inner):
            action = worker.act(plan, board)
            board = decoder.decode(action, board)
            new_residual = calculate_residual(board)
            if abs(new_residual - residual) < tolerance:
                break
            residual = new_residual
        if is_solved(board):
            break
    return board, outer_iter + 1

def run_hrm(board, config):
    random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    planner = Planner(config['planner'])
    worker = Worker(config['worker'])
    decoder = Decoder(config['decoder'])
    
    solved_board, iters = fixed_point_loop(board, planner, worker, decoder)
    return solved_board, iters

if __name__ == '__main__':
    # Example usage
    config = {'seed': 42, 'planner': {}, 'worker': {}, 'decoder': {}}
    board = np.zeros((9, 9))  # Trivial board
    solved, iters = run_hrm(board, config)
    print(f"Solved in {iters} iterations")