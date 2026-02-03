"""
Test the Rust MCTS integration with the actual Blokus game.
"""

import time
import torch
import numpy as np
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.features.alphazero.game import BlokusGame
from app.features.alphazero.network import BlokusNet
from app.features.alphazero.mcts import MCTSConfig, BatchedMCTS
from app.features.alphazero.mcts_rust import HybridMCTS, is_rust_available, encode_move, decode_move


def test_move_encoding():
    """Test that move encoding/decoding is correct."""
    print("\n=== Testing Move Encoding ===")

    from app.features.alphazero.game import Move

    for piece_idx in [0, 10, 20]:
        for orientation in [0, 4, 7]:
            for row in [0, 10, 19]:
                for col in [0, 10, 19]:
                    move = Move(player=0, piece_idx=piece_idx, orientation_idx=orientation, row=row, col=col)
                    encoded = encode_move(move)
                    decoded = decode_move(encoded, player=0)

                    assert decoded.piece_idx == piece_idx, f"piece_idx mismatch"
                    assert decoded.orientation_idx == orientation, f"orientation mismatch"
                    assert decoded.row == row, f"row mismatch"
                    assert decoded.col == col, f"col mismatch"

    print("✓ Move encoding/decoding correct")


def test_rust_mcts_single_game():
    """Test Rust MCTS on a single game."""
    print("\n=== Testing Rust MCTS (Single Game) ===")

    if not is_rust_available():
        print("✗ Rust MCTS not available, skipping")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create network
    network = BlokusNet(num_res_blocks=5, channels=64).to(device)
    network.eval()

    # Create config
    config = MCTSConfig(
        num_simulations=50,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
        temperature_threshold=20,
    )

    # Create MCTS
    rust_mcts = HybridMCTS(network, config, device)

    # Create game
    game = BlokusGame()
    print(f"Initial legal moves: {len(game.get_legal_moves())}")

    # Run MCTS
    with torch.no_grad():
        start = time.time()
        move, policy, visits = rust_mcts.search(game, add_noise=True, batch_size=16)
        elapsed = time.time() - start

    print(f"✓ Rust MCTS completed in {elapsed*1000:.1f}ms")
    print(f"  Selected move: piece={move.piece_idx}, orient={move.orientation_idx}, pos=({move.row}, {move.col})")
    print(f"  Policy sum: {policy.sum():.4f}")
    print(f"  Root visits: {visits}")


def test_rust_vs_python_mcts():
    """Compare Rust and Python MCTS implementations."""
    print("\n=== Comparing Rust vs Python MCTS ===")

    if not is_rust_available():
        print("✗ Rust MCTS not available, skipping")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = BlokusNet(num_res_blocks=5, channels=64).to(device)
    network.eval()

    config = MCTSConfig(
        num_simulations=100,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    rust_mcts = HybridMCTS(network, config, device)
    python_mcts = BatchedMCTS(network, config, device)

    game = BlokusGame()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        python_mcts.select_move(game, batch_size=16)
        rust_mcts.search(game, batch_size=16)

    # Benchmark Python
    print("Benchmarking Python MCTS...")
    with torch.no_grad():
        start = time.time()
        for _ in range(3):
            move_py, probs_py, _ = python_mcts.select_move(game, batch_size=16)
        python_time = (time.time() - start) / 3

    # Benchmark Rust
    print("Benchmarking Rust MCTS...")
    with torch.no_grad():
        start = time.time()
        for _ in range(3):
            move_rs, probs_rs, _ = rust_mcts.search(game, batch_size=16)
        rust_time = (time.time() - start) / 3

    print(f"\nResults (100 simulations, batch_size=16):")
    print(f"  Python MCTS: {python_time*1000:.1f}ms per decision")
    print(f"  Rust MCTS:   {rust_time*1000:.1f}ms per decision")
    print(f"  Speedup:     {python_time/rust_time:.2f}x")


def test_play_game():
    """Test playing a few moves with Rust MCTS."""
    print("\n=== Playing Moves with Rust MCTS ===")

    if not is_rust_available():
        print("✗ Rust MCTS not available, skipping")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = BlokusNet(num_res_blocks=5, channels=64).to(device)
    network.eval()

    config = MCTSConfig(num_simulations=20)
    rust_mcts = HybridMCTS(network, config, device)

    game = BlokusGame()
    moves_played = 0

    print("Playing first 8 moves...")
    with torch.no_grad():
        while not game.is_game_over and moves_played < 8:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                game.pass_turn()
                continue

            move, policy, visits = rust_mcts.search(game, add_noise=moves_played < 4, batch_size=8)
            game.make_move(move)
            moves_played += 1

            print(f"  Move {moves_played}: Player {move.player}, piece {move.piece_idx} at ({move.row}, {move.col})")

    print(f"✓ Played {moves_played} moves successfully")
    print(f"  Scores: {[p.score for p in game.players]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Rust MCTS Integration Tests")
    print("=" * 60)

    test_move_encoding()
    test_rust_mcts_single_game()
    test_rust_vs_python_mcts()
    test_play_game()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
