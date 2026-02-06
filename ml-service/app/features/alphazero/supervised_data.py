"""
Supervised data generation for bootstrapping AlphaZero training.

Generates training data by playing Expert vs Expert Blokus games,
recording every position with heuristic policy targets, then assigning
value targets based on final game outcomes.

This provides high-quality initial training data without needing
neural network self-play, solving the cold-start problem.
"""

import os
import time
import pickle
import random
import multiprocessing as mp
from typing import List, Optional
from dataclasses import dataclass, field

import numpy as np

from app.logging_config import log
from .game import BlokusGame, Move, NUM_PLAYERS
from .self_play import TrainingExample
from .smart_player import SmartPlayer, SmartPlayerConfig
from .heuristics import compute_heuristic_policy


@dataclass
class GameRecord:
    """Record of a complete game played between expert players."""
    positions: List[dict] = field(default_factory=list)
    # Each position dict contains:
    #   board_tensor: np.ndarray (10, 20, 20)
    #   pieces_vector: np.ndarray (84,)
    #   legal_moves: List[Move]
    #   policy: np.ndarray
    #   current_player: int
    scores: List[int] = field(default_factory=list)
    winner: Optional[int] = None


def generate_expert_games(num_games: int, difficulty: str = "expert") -> List[GameRecord]:
    """
    Generate games where all 4 players are SmartPlayer(Expert).

    Args:
        num_games: Number of games to generate.
        difficulty: Difficulty preset for SmartPlayer.

    Returns:
        List of GameRecord with positions, scores, and winner.
    """
    records = []
    start_time = time.time()

    for game_idx in range(num_games):
        game = BlokusGame()
        players = [SmartPlayer(SmartPlayerConfig.expert()) for _ in range(NUM_PLAYERS)]

        record = GameRecord()

        while not game.is_game_over:
            legal_moves = game.get_legal_moves()

            if not legal_moves:
                game.pass_turn()
                continue

            # Capture state before the move
            board_tensor, pieces_vector = game.get_state_for_neural_net()
            policy = compute_heuristic_policy(game, legal_moves)
            current_player = game.current_player

            record.positions.append({
                "board_tensor": board_tensor.copy(),
                "pieces_vector": pieces_vector.copy(),
                "legal_moves": legal_moves.copy(),
                "policy": policy.copy(),
                "current_player": current_player,
            })

            # Select and make move using the expert player
            move = players[game.current_player].select_move(game)
            if move:
                game.make_move(move)
            else:
                game.pass_turn()

        record.scores = game.get_scores()
        record.winner = game.get_winner()
        records.append(record)

        # Log progress every 100 games
        if (game_idx + 1) % 100 == 0 or game_idx == 0:
            elapsed = time.time() - start_time
            games_per_sec = (game_idx + 1) / elapsed if elapsed > 0 else 0
            eta_sec = (num_games - game_idx - 1) / games_per_sec if games_per_sec > 0 else 0
            log.info(
                f"[supervised] Game {game_idx + 1}/{num_games} "
                f"({len(record.positions)} positions, "
                f"scores={record.scores}, winner={record.winner}, "
                f"{games_per_sec:.2f} games/s, ETA: {eta_sec/60:.1f}min)"
            )

    elapsed = time.time() - start_time
    total_positions = sum(len(r.positions) for r in records)
    log.info(
        f"[supervised] Generated {num_games} games with {total_positions} "
        f"total positions in {elapsed:.1f}s"
    )

    return records


def _play_single_expert_game(_seed: int) -> GameRecord:
    """Play one Expert game. Top-level function for multiprocessing."""
    # Each worker needs its own RNG seed for variety
    random.seed(_seed)
    np.random.seed(_seed % (2**31))

    game = BlokusGame()
    players = [SmartPlayer(SmartPlayerConfig.expert()) for _ in range(NUM_PLAYERS)]

    record = GameRecord()

    while not game.is_game_over:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            game.pass_turn()
            continue

        board_tensor, pieces_vector = game.get_state_for_neural_net()
        policy = compute_heuristic_policy(game, legal_moves)
        current_player = game.current_player

        record.positions.append({
            "board_tensor": board_tensor.copy(),
            "pieces_vector": pieces_vector.copy(),
            "legal_moves": legal_moves.copy(),
            "policy": policy.copy(),
            "current_player": current_player,
        })

        move = players[game.current_player].select_move(game)
        if move:
            game.make_move(move)
        else:
            game.pass_turn()

    record.scores = game.get_scores()
    record.winner = game.get_winner()
    return record


def generate_expert_games_parallel(
    num_games: int,
    num_workers: int = 0,
) -> List[GameRecord]:
    """
    Generate Expert games using multiprocessing for ~Nx speedup.

    Args:
        num_games: Number of games to generate.
        num_workers: Number of parallel workers. 0 = auto (cpu_count - 2).

    Returns:
        List of GameRecord.
    """
    if num_workers <= 0:
        num_workers = max(1, mp.cpu_count() - 2)

    log.info(f"[supervised] Generating {num_games} expert games with {num_workers} workers")
    start_time = time.time()

    # Use unique seeds per game for variety
    base_seed = int(time.time() * 1000) % (2**31)
    seeds = [base_seed + i for i in range(num_games)]

    records = []
    completed = 0

    with mp.Pool(num_workers) as pool:
        for record in pool.imap_unordered(_play_single_expert_game, seeds, chunksize=4):
            records.append(record)
            completed += 1
            if completed % 100 == 0 or completed == 1:
                elapsed = time.time() - start_time
                games_per_sec = completed / elapsed
                eta_sec = (num_games - completed) / games_per_sec if games_per_sec > 0 else 0
                log.info(
                    f"[supervised] Game {completed}/{num_games} "
                    f"({len(record.positions)} pos, scores={record.scores}, "
                    f"{games_per_sec:.2f} games/s, ETA: {eta_sec/60:.1f}min)"
                )

    elapsed = time.time() - start_time
    total_positions = sum(len(r.positions) for r in records)
    log.info(
        f"[supervised] Generated {num_games} games with {total_positions} "
        f"total positions in {elapsed:.1f}s using {num_workers} workers "
        f"({num_games/elapsed:.1f} games/s)"
    )

    return records


def generate_training_examples(num_games: int, parallel: bool = True) -> List[TrainingExample]:
    """
    Generate supervised training examples from expert games.

    Plays expert vs expert games and converts each position into a
    TrainingExample with heuristic policy targets and outcome-based
    value targets.

    Args:
        num_games: Number of games to generate.
        parallel: Use multiprocessing (default True, ~10x faster).

    Returns:
        List of TrainingExample ready for neural network training.
    """
    if parallel:
        records = generate_expert_games_parallel(num_games)
    else:
        records = generate_expert_games(num_games)

    examples = []
    for record in records:
        for position in record.positions:
            player = position["current_player"]

            # Assign value based on game outcome
            if record.winner is not None:
                value = 1.0 if player == record.winner else -1.0
            else:
                # Tie: use relative score
                my_score = record.scores[player]
                max_other = max(
                    record.scores[i] for i in range(NUM_PLAYERS) if i != player
                )
                if my_score > max_other:
                    value = 0.5
                elif my_score < max_other:
                    value = -0.5
                else:
                    value = 0.0

            examples.append(TrainingExample(
                board_tensor=position["board_tensor"],
                pieces_vector=position["pieces_vector"],
                policy=position["policy"],
                legal_moves=position["legal_moves"],
                value=value,
            ))

    log.info(f"[supervised] Created {len(examples)} training examples from {num_games} games")
    return examples


def save_dataset(examples: List[TrainingExample], path: str):
    """Save training examples to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(examples, f)
    log.info(f"[supervised] Saved {len(examples)} examples to {path}")


def load_dataset(path: str) -> List[TrainingExample]:
    """Load training examples from disk."""
    with open(path, "rb") as f:
        examples = pickle.load(f)
    log.info(f"[supervised] Loaded {len(examples)} examples from {path}")
    return examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate supervised training data from expert games")
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/supervised/expert_games.pkl")
    args = parser.parse_args()

    examples = generate_training_examples(args.num_games)
    save_dataset(examples, args.output)
    print(f"Saved {len(examples)} examples to {args.output}")
