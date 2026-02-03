"""
Comprehensive tests for GPU game engine.
Verifies that GPU implementation matches CPU implementation exactly.
"""

import torch
import numpy as np
import random
import time
from typing import List

from .game import BlokusGame, Move
from .game_gpu import BlokusGameBatchGPU, PieceDataGPU


def test_initial_state():
    """Test that initial state matches between CPU and GPU."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cpu_game = BlokusGame()
    gpu_batch = BlokusGameBatchGPU.from_cpu_game(cpu_game, device)
    gpu_game = gpu_batch.to_cpu_game(0)

    assert np.array_equal(cpu_game.board, gpu_game.board), "Boards don't match"
    assert cpu_game.current_player == gpu_game.current_player, "Current player doesn't match"
    assert cpu_game.turn_number == gpu_game.turn_number, "Turn number doesn't match"

    for p in range(4):
        assert cpu_game.players[p].remaining_pieces == gpu_game.players[p].remaining_pieces
        assert cpu_game.players[p].has_played == gpu_game.players[p].has_played
        assert cpu_game._valid_corners[p] == gpu_game._valid_corners[p]

    print("Initial state test PASSED")


def test_move_validation():
    """Test that move validation matches between CPU and GPU."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cpu_game = BlokusGame()
    gpu_batch = BlokusGameBatchGPU.from_cpu_game(cpu_game, device)

    # Get all legal moves from CPU
    legal_moves = cpu_game.get_legal_moves()

    # Test some legal moves
    for move in legal_moves[:10]:
        valid = gpu_batch.validate_moves(
            batch_indices=torch.tensor([0], device=device),
            piece_indices=torch.tensor([move.piece_idx], device=device),
            orient_indices=torch.tensor([move.orientation_idx], device=device),
            rows=torch.tensor([move.row], device=device),
            cols=torch.tensor([move.col], device=device),
        )
        assert valid[0].item(), f"Legal move {move} marked as invalid by GPU"

    # Test some illegal moves (wrong piece position)
    for piece_idx in range(5):
        for row in range(5, 10):  # Middle of board - no starting corner
            valid = gpu_batch.validate_moves(
                batch_indices=torch.tensor([0], device=device),
                piece_indices=torch.tensor([piece_idx], device=device),
                orient_indices=torch.tensor([0], device=device),
                rows=torch.tensor([row], device=device),
                cols=torch.tensor([row], device=device),
            )
            # First move must cover starting corner
            cpu_valid = cpu_game.is_valid_move(Move(0, piece_idx, 0, row, row))
            assert valid[0].item() == cpu_valid, \
                f"Validation mismatch for piece {piece_idx} at ({row}, {row})"

    print("Move validation test PASSED")


def test_full_game():
    """Play a complete random game and verify GPU matches CPU at each step."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cpu_game = BlokusGame()
    gpu_batch = BlokusGameBatchGPU.from_cpu_game(cpu_game, device)

    move_count = 0
    while not cpu_game.is_game_over and move_count < 100:
        legal_moves = cpu_game.get_legal_moves()

        if not legal_moves:
            cpu_game.pass_turn()
            gpu_batch.pass_turn(torch.tensor([0], device=device))
            continue

        move = random.choice(legal_moves)

        # Apply to both
        cpu_game.make_move(move)
        gpu_batch.apply_moves(
            batch_indices=torch.tensor([0], device=device),
            piece_indices=torch.tensor([move.piece_idx], device=device),
            orient_indices=torch.tensor([move.orientation_idx], device=device),
            rows=torch.tensor([move.row], device=device),
            cols=torch.tensor([move.col], device=device),
        )

        # Verify match
        gpu_game = gpu_batch.to_cpu_game(0)
        assert np.array_equal(cpu_game.board, gpu_game.board), f"Board mismatch at move {move_count}"
        assert cpu_game.current_player == gpu_game.current_player, f"Player mismatch at move {move_count}"
        assert cpu_game.is_game_over == gpu_game.is_game_over, f"Game over mismatch at move {move_count}"

        move_count += 1

    print(f"Full game test PASSED ({move_count} moves)")


def test_batch_operations():
    """Test batched operations with multiple games."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8

    # Create batch of games
    cpu_games = [BlokusGame() for _ in range(batch_size)]
    gpu_batch = BlokusGameBatchGPU.from_cpu_games(cpu_games, device)

    # Get legal moves for each game
    all_moves = [g.get_legal_moves() for g in cpu_games]

    # Apply random moves to all games simultaneously
    batch_indices = []
    piece_indices = []
    orient_indices = []
    rows = []
    cols = []

    for i, moves in enumerate(all_moves):
        if moves:
            move = random.choice(moves)
            batch_indices.append(i)
            piece_indices.append(move.piece_idx)
            orient_indices.append(move.orientation_idx)
            rows.append(move.row)
            cols.append(move.col)
            cpu_games[i].make_move(move)

    # Apply moves on GPU
    if batch_indices:
        gpu_batch.apply_moves(
            batch_indices=torch.tensor(batch_indices, device=device),
            piece_indices=torch.tensor(piece_indices, device=device),
            orient_indices=torch.tensor(orient_indices, device=device),
            rows=torch.tensor(rows, device=device),
            cols=torch.tensor(cols, device=device),
        )

    # Verify all games match
    for i in range(batch_size):
        gpu_game = gpu_batch.to_cpu_game(i)
        assert np.array_equal(cpu_games[i].board, gpu_game.board), f"Game {i} board mismatch"

    print(f"Batch operations test PASSED ({batch_size} games)")


def test_clone_states():
    """Test cloning game states."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create and advance a game
    cpu_game = BlokusGame()
    for _ in range(5):
        moves = cpu_game.get_legal_moves()
        if moves:
            cpu_game.make_move(random.choice(moves))

    gpu_batch = BlokusGameBatchGPU.from_cpu_game(cpu_game, device)

    # Clone into multiple copies
    indices = torch.tensor([0, 0, 0, 0], device=device)
    cloned_batch = gpu_batch.clone_states(indices)

    # Verify all clones match original
    for i in range(4):
        cloned_game = cloned_batch.to_cpu_game(i)
        assert np.array_equal(cpu_game.board, cloned_game.board), f"Clone {i} board mismatch"
        assert cpu_game.current_player == cloned_game.current_player

    # Modify one clone and verify others unchanged
    moves = cpu_game.get_legal_moves()
    if moves:
        move = moves[0]
        cloned_batch.apply_moves(
            batch_indices=torch.tensor([0], device=device),
            piece_indices=torch.tensor([move.piece_idx], device=device),
            orient_indices=torch.tensor([move.orientation_idx], device=device),
            rows=torch.tensor([move.row], device=device),
            cols=torch.tensor([move.col], device=device),
        )

        # Clone 0 should be modified
        modified_game = cloned_batch.to_cpu_game(0)
        assert not np.array_equal(cpu_game.board, modified_game.board), "Clone 0 should be modified"

        # Clone 1-3 should still match original
        for i in range(1, 4):
            unchanged_game = cloned_batch.to_cpu_game(i)
            assert np.array_equal(cpu_game.board, unchanged_game.board), f"Clone {i} should be unchanged"

    print("Clone states test PASSED")


def test_neural_net_encoding():
    """Test that neural network encoding matches CPU."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cpu_game = BlokusGame()

    # Make a few moves
    for _ in range(3):
        moves = cpu_game.get_legal_moves()
        if moves:
            cpu_game.make_move(random.choice(moves))

    gpu_batch = BlokusGameBatchGPU.from_cpu_game(cpu_game, device)

    # Get encodings
    cpu_board, cpu_pieces = cpu_game.get_state_for_neural_net()
    gpu_board, gpu_pieces = gpu_batch.get_state_for_neural_net()

    # Compare
    gpu_board_np = gpu_board[0].cpu().numpy()
    gpu_pieces_np = gpu_pieces[0].cpu().numpy()

    assert np.allclose(cpu_board, gpu_board_np, atol=1e-6), "Board tensor mismatch"
    assert np.allclose(cpu_pieces, gpu_pieces_np, atol=1e-6), "Pieces vector mismatch"

    print("Neural net encoding test PASSED")


def benchmark_gpu_vs_cpu():
    """Benchmark GPU vs CPU game operations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping benchmark")
        return

    num_games = 100
    moves_per_game = 10

    # CPU benchmark
    print(f"\nBenchmarking {num_games} games with {moves_per_game} moves each...")

    cpu_start = time.time()
    for _ in range(num_games):
        game = BlokusGame()
        for _ in range(moves_per_game):
            moves = game.get_legal_moves()
            if moves:
                game.make_move(random.choice(moves))
    cpu_time = time.time() - cpu_start
    print(f"CPU: {cpu_time*1000:.1f}ms total, {cpu_time/num_games*1000:.2f}ms per game")

    # GPU benchmark (sequential moves within batch)
    gpu_start = time.time()
    torch.cuda.synchronize()

    for _ in range(num_games):
        cpu_game = BlokusGame()
        gpu_batch = BlokusGameBatchGPU.from_cpu_game(cpu_game, device)

        for _ in range(moves_per_game):
            # Need legal moves from CPU for now
            moves = cpu_game.get_legal_moves()
            if moves:
                move = random.choice(moves)
                gpu_batch.apply_moves(
                    batch_indices=torch.tensor([0], device=device),
                    piece_indices=torch.tensor([move.piece_idx], device=device),
                    orient_indices=torch.tensor([move.orientation_idx], device=device),
                    rows=torch.tensor([move.row], device=device),
                    cols=torch.tensor([move.col], device=device),
                )
                cpu_game.make_move(move)

    torch.cuda.synchronize()
    gpu_time = time.time() - gpu_start
    print(f"GPU (sequential): {gpu_time*1000:.1f}ms total, {gpu_time/num_games*1000:.2f}ms per game")

    # GPU benchmark (batched moves)
    batch_size = 32
    gpu_batch_start = time.time()
    torch.cuda.synchronize()

    for batch_idx in range(num_games // batch_size):
        cpu_games = [BlokusGame() for _ in range(batch_size)]
        gpu_batch = BlokusGameBatchGPU.from_cpu_games(cpu_games, device)

        for _ in range(moves_per_game):
            batch_indices = []
            piece_indices = []
            orient_indices = []
            rows_list = []
            cols_list = []

            for i, game in enumerate(cpu_games):
                moves = game.get_legal_moves()
                if moves:
                    move = random.choice(moves)
                    batch_indices.append(i)
                    piece_indices.append(move.piece_idx)
                    orient_indices.append(move.orientation_idx)
                    rows_list.append(move.row)
                    cols_list.append(move.col)
                    game.make_move(move)

            if batch_indices:
                gpu_batch.apply_moves(
                    batch_indices=torch.tensor(batch_indices, device=device),
                    piece_indices=torch.tensor(piece_indices, device=device),
                    orient_indices=torch.tensor(orient_indices, device=device),
                    rows=torch.tensor(rows_list, device=device),
                    cols=torch.tensor(cols_list, device=device),
                )

    torch.cuda.synchronize()
    gpu_batch_time = time.time() - gpu_batch_start
    effective_games = (num_games // batch_size) * batch_size
    print(f"GPU (batched {batch_size}): {gpu_batch_time*1000:.1f}ms for {effective_games} games, "
          f"{gpu_batch_time/effective_games*1000:.2f}ms per game")

    # Clone benchmark
    print("\nClone benchmark:")
    cpu_game = BlokusGame()
    for _ in range(10):
        moves = cpu_game.get_legal_moves()
        if moves:
            cpu_game.make_move(random.choice(moves))

    num_clones = 1000

    cpu_clone_start = time.time()
    for _ in range(num_clones):
        _ = cpu_game.clone()
    cpu_clone_time = time.time() - cpu_clone_start
    print(f"CPU clone: {num_clones} clones in {cpu_clone_time*1000:.1f}ms, "
          f"{cpu_clone_time/num_clones*1000000:.1f}us per clone")

    gpu_batch = BlokusGameBatchGPU.from_cpu_game(cpu_game, device)
    indices = torch.zeros(num_clones, dtype=torch.int64, device=device)

    torch.cuda.synchronize()
    gpu_clone_start = time.time()
    _ = gpu_batch.clone_states(indices)
    torch.cuda.synchronize()
    gpu_clone_time = time.time() - gpu_clone_start
    print(f"GPU clone: {num_clones} clones in {gpu_clone_time*1000:.1f}ms, "
          f"{gpu_clone_time/num_clones*1000000:.1f}us per clone")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running GPU Game Engine Tests")
    print("=" * 60)

    test_initial_state()
    test_move_validation()
    test_full_game()
    test_batch_operations()
    test_clone_states()
    test_neural_net_encoding()
    benchmark_gpu_vs_cpu()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
