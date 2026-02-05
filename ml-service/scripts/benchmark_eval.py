#!/usr/bin/env python3
"""
Benchmark script for evaluation games.

Runs evaluation games between two neural networks (or NN vs heuristic)
and reports detailed timing information.

Usage:
    python scripts/benchmark_eval.py [--games 64] [--parallel 16] [--mcts-sims 400]
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.features.alphazero.game import BlokusGame
from app.features.alphazero.network import create_network
from app.features.alphazero.mcts import MCTSConfig
from app.features.alphazero.mcts_rust import HybridMCTS, is_rust_available
from app.features.alphazero.heuristic_player import HeuristicPlayer


@dataclass
class TimingStats:
    """Accumulated timing statistics."""
    total_time: float = 0.0
    game_times: list = None
    move_times: list = None
    total_moves: int = 0
    total_games: int = 0

    def __post_init__(self):
        if self.game_times is None:
            self.game_times = []
        if self.move_times is None:
            self.move_times = []

    def add_game(self, game_time: float, moves: int, move_times: list):
        self.game_times.append(game_time)
        self.move_times.extend(move_times)
        self.total_moves += moves
        self.total_games += 1
        self.total_time += game_time

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            'total_games': self.total_games,
            'total_moves': self.total_moves,
            'total_time_sec': round(self.total_time, 2),
            'avg_game_time_sec': round(statistics.mean(self.game_times), 3) if self.game_times else 0,
            'std_game_time_sec': round(statistics.stdev(self.game_times), 3) if len(self.game_times) > 1 else 0,
            'min_game_time_sec': round(min(self.game_times), 3) if self.game_times else 0,
            'max_game_time_sec': round(max(self.game_times), 3) if self.game_times else 0,
            'avg_move_time_ms': round(statistics.mean(self.move_times) * 1000, 2) if self.move_times else 0,
            'std_move_time_ms': round(statistics.stdev(self.move_times) * 1000, 2) if len(self.move_times) > 1 else 0,
            'moves_per_sec': round(self.total_moves / self.total_time, 1) if self.total_time > 0 else 0,
            'games_per_min': round(self.total_games / self.total_time * 60, 2) if self.total_time > 0 else 0,
        }


def run_single_game(
    network1: torch.nn.Module,
    network2: torch.nn.Module,
    mcts_config: MCTSConfig,
    mcts_batch_size: int,
    device: str,
    player1_idx: int = 0,  # Which player network1 plays as
) -> tuple[int, float, int, list[float]]:
    """
    Run a single evaluation game between two networks.

    Returns:
        (winner, game_time, num_moves, move_times)
    """
    game = BlokusGame()
    mcts1 = HybridMCTS(network1, mcts_config, device)
    mcts2 = HybridMCTS(network2, mcts_config, device)

    move_times = []
    game_start = time.perf_counter()

    while not game.is_game_over:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            game.pass_turn()
            continue

        current = game.current_player
        mcts = mcts1 if current == player1_idx else mcts2

        move_start = time.perf_counter()
        move, _, _ = mcts.search(game, add_noise=False, batch_size=mcts_batch_size)
        move_time = time.perf_counter() - move_start
        move_times.append(move_time)

        game.make_move(move)

    game_time = time.perf_counter() - game_start
    winner = game.get_winner()

    return winner, game_time, len(move_times), move_times


def run_single_game_vs_heuristic(
    network: torch.nn.Module,
    mcts_config: MCTSConfig,
    mcts_batch_size: int,
    device: str,
    nn_player_idx: int = 0,
) -> tuple[int, float, int, list[float]]:
    """
    Run a single evaluation game between network and heuristic player.

    Returns:
        (winner, game_time, num_moves, move_times)
    """
    game = BlokusGame()
    mcts = HybridMCTS(network, mcts_config, device)
    heuristic = HeuristicPlayer()

    move_times = []
    game_start = time.perf_counter()

    while not game.is_game_over:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            game.pass_turn()
            continue

        current = game.current_player

        move_start = time.perf_counter()
        if current == nn_player_idx:
            move, _, _ = mcts.search(game, add_noise=False, batch_size=mcts_batch_size)
        else:
            move = heuristic.select_move(game)

        move_time = time.perf_counter() - move_start
        move_times.append(move_time)

        if move:
            game.make_move(move)
        else:
            game.pass_turn()

    game_time = time.perf_counter() - game_start
    winner = game.get_winner()

    return winner, game_time, len(move_times), move_times


def run_parallel_evaluation(
    network1: torch.nn.Module,
    network2: Optional[torch.nn.Module],
    total_games: int,
    parallel_games: int,
    mcts_config: MCTSConfig,
    mcts_batch_size: int,
    device: str,
    vs_heuristic: bool = False,
) -> tuple[TimingStats, int]:
    """
    Run parallel evaluation games using ThreadPoolExecutor.

    Returns:
        (timing_stats, network1_wins)
    """
    stats = TimingStats()
    network1_wins = 0

    print(f"\nRunning {total_games} evaluation games ({parallel_games} parallel threads)...")
    print(f"MCTS simulations: {mcts_config.num_simulations}")
    print(f"MCTS batch size: {mcts_batch_size}")
    print(f"Mode: {'NN vs Heuristic' if vs_heuristic else 'NN vs NN'}")
    print("-" * 60)

    def run_game(game_idx: int):
        """Run a single game - called from thread pool."""
        player1_idx = game_idx % 2  # Alternate who plays first
        if vs_heuristic:
            return run_single_game_vs_heuristic(
                network1, mcts_config, mcts_batch_size, device, player1_idx
            ) + (player1_idx,)
        else:
            return run_single_game(
                network1, network2, mcts_config, mcts_batch_size, device, player1_idx
            ) + (player1_idx,)

    batch_start = time.perf_counter()
    games_completed = 0

    # Use ThreadPoolExecutor for true parallelism
    # Rust MCTS releases the GIL, so threads can run concurrently
    with ThreadPoolExecutor(max_workers=parallel_games) as executor:
        futures = {executor.submit(run_game, i): i for i in range(total_games)}

        for future in as_completed(futures):
            winner, game_time, num_moves, move_times, player1_idx = future.result()

            if winner == player1_idx:
                network1_wins += 1
            stats.add_game(game_time, num_moves, move_times)
            games_completed += 1

            if games_completed % parallel_games == 0 or games_completed == total_games:
                elapsed = time.perf_counter() - batch_start
                win_rate = network1_wins / games_completed
                games_per_min = games_completed / elapsed * 60
                print(f"Progress: {games_completed}/{total_games} games in {elapsed:.1f}s | "
                      f"{games_per_min:.1f} games/min | "
                      f"Win rate: {win_rate:.1%} ({network1_wins}/{games_completed})")

    return stats, network1_wins


def print_timing_report(stats: TimingStats, network1_wins: int, mode: str):
    """Print detailed timing report."""
    summary = stats.summary()

    print("\n" + "=" * 60)
    print(f"EVALUATION BENCHMARK RESULTS ({mode})")
    print("=" * 60)

    print(f"\nGames Summary:")
    print(f"  Total games:      {summary['total_games']}")
    print(f"  Network 1 wins:   {network1_wins} ({network1_wins/summary['total_games']:.1%})")
    print(f"  Total moves:      {summary['total_moves']}")

    print(f"\nTiming Summary:")
    print(f"  Total time:       {summary['total_time_sec']:.1f}s")
    print(f"  Games per minute: {summary['games_per_min']:.2f}")
    print(f"  Moves per second: {summary['moves_per_sec']:.1f}")

    print(f"\nPer-Game Statistics:")
    print(f"  Mean game time:   {summary['avg_game_time_sec']:.3f}s")
    print(f"  Std dev:          {summary['std_game_time_sec']:.3f}s")
    print(f"  Min game time:    {summary['min_game_time_sec']:.3f}s")
    print(f"  Max game time:    {summary['max_game_time_sec']:.3f}s")

    print(f"\nPer-Move Statistics:")
    print(f"  Mean move time:   {summary['avg_move_time_ms']:.2f}ms")
    print(f"  Std dev:          {summary['std_move_time_ms']:.2f}ms")

    # Estimate for 200 games
    estimated_200_time = summary['avg_game_time_sec'] * 200
    print(f"\nEstimated time for 200 games: {estimated_200_time:.1f}s ({estimated_200_time/60:.1f} min)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Benchmark evaluation games')
    parser.add_argument('--games', type=int, default=64, help='Number of games to run')
    parser.add_argument('--parallel', type=int, default=16, help='Number of parallel games')
    parser.add_argument('--mcts-sims', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--mcts-batch', type=int, default=16, help='MCTS batch size')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--vs-heuristic', action='store_true', help='Evaluate against heuristic player')
    parser.add_argument('--num-res-blocks', type=int, default=15, help='Number of residual blocks')
    parser.add_argument('--channels', type=int, default=256, help='Number of channels')
    args = parser.parse_args()

    # Check Rust MCTS availability
    if not is_rust_available():
        print("ERROR: Rust MCTS not available!")
        print("Build with: cd ml-service/rust_mcts && maturin develop --release")
        sys.exit(1)

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create networks
    print(f"Creating network (res_blocks={args.num_res_blocks}, channels={args.channels})...")
    network1 = create_network(
        num_res_blocks=args.num_res_blocks,
        channels=args.channels,
        device=device,
    )

    # Load model weights if provided
    if args.model:
        print(f"Loading model from: {args.model}")
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            network1.load_state_dict(checkpoint['model_state_dict'])
        else:
            network1.load_state_dict(checkpoint)

    network1.eval()

    # Second network (same architecture, different random init) for NN vs NN
    network2 = None
    if not args.vs_heuristic:
        network2 = create_network(
            num_res_blocks=args.num_res_blocks,
            channels=args.channels,
            device=device,
        )
        network2.eval()

    # MCTS config
    mcts_config = MCTSConfig(
        num_simulations=args.mcts_sims,
        c_puct=1.5,
        temperature_threshold=20,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    # Warmup
    print("\nWarmup run (1 game)...")
    warmup_start = time.perf_counter()
    if args.vs_heuristic:
        run_single_game_vs_heuristic(network1, mcts_config, args.mcts_batch, device, 0)
    else:
        run_single_game(network1, network2, mcts_config, args.mcts_batch, device, 0)
    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup complete in {warmup_time:.1f}s")

    # Run benchmark
    mode = "NN vs Heuristic" if args.vs_heuristic else "NN vs NN"
    stats, wins = run_parallel_evaluation(
        network1=network1,
        network2=network2,
        total_games=args.games,
        parallel_games=args.parallel,
        mcts_config=mcts_config,
        mcts_batch_size=args.mcts_batch,
        device=device,
        vs_heuristic=args.vs_heuristic,
    )

    # Print report
    print_timing_report(stats, wins, mode)


if __name__ == '__main__':
    main()
