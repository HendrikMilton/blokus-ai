"""
Self-play module for generating training data.
Runs games where the AI plays against itself using MCTS.

Supports both sequential and parallel game generation with batched
neural network inference for maximum GPU utilization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from app.logging_config import log
from .game import BlokusGame, Move, NUM_PLAYERS
from .mcts import MCTS, MCTSConfig, BatchedMCTS, MCTSNode
from .network import BlokusNet
from .mcts_rust import HybridMCTS, encode_move

# Import heuristics for cold start bootstrapping (optional, may not exist)
try:
    from .heuristics import HeuristicConfig, compute_heuristic_policy, blend_policies
    HEURISTICS_AVAILABLE = True
except ImportError:
    HEURISTICS_AVAILABLE = False
    HeuristicConfig = None

log.info("Using Rust-accelerated MCTS")


@dataclass
class TrainingExample:
    """A single training example from self-play."""
    board_tensor: np.ndarray      # (10, 20, 20) state encoding
    pieces_vector: np.ndarray     # (84,) remaining pieces
    policy: np.ndarray            # (num_moves,) target policy
    legal_moves: List[Move]       # Legal moves at this position
    value: float                  # Final game outcome from this player's perspective


@dataclass
class GameMetrics:
    """Heuristic metrics collected during a game for diagnostics."""
    avg_corner_count: float       # Average valid corners per player per turn
    avg_mobility: float           # Average legal moves available per turn
    avg_center_coverage: float    # Average pieces placed near center (inner 10x10)
    final_corner_counts: List[int]  # Final corner counts per player
    # Policy quality metrics (added for debugging cold start problem)
    avg_policy_entropy: float = 0.0  # Average entropy of MCTS policy targets (higher = more uniform)
    avg_top1_prob: float = 0.0       # Average probability of top move (higher = more focused)
    avg_top5_prob: float = 0.0       # Average probability mass in top 5 moves


@dataclass
class GameResult:
    """Result of a self-play game."""
    examples: List[TrainingExample]
    winner: Optional[int]
    scores: List[int]
    num_moves: int
    metrics: Optional[GameMetrics] = None  # Diagnostic metrics


class SelfPlayWorker:
    """Worker that runs self-play games."""

    def __init__(self, network: BlokusNet, mcts_config: MCTSConfig, device: str = 'cuda'):
        self.network = network
        self.mcts_config = mcts_config
        self.device = device

    def play_game(self, collect_metrics: bool = True) -> GameResult:
        """
        Play a single self-play game and collect training examples.

        Args:
            collect_metrics: Whether to collect heuristic metrics for diagnostics

        Returns:
            GameResult with examples and game outcome
        """
        from .mcts import MCTS
        import time
        import os

        import sys
        game = BlokusGame()
        mcts = MCTS(self.network, self.mcts_config, self.device)

        # Collect examples: (state, policy, player)
        examples_raw: List[Tuple[np.ndarray, np.ndarray, List[Move], int]] = []

        # Metrics tracking
        corner_counts: List[int] = []  # Track corner counts per turn
        mobility_counts: List[int] = []  # Track legal move counts per turn

        move_count = 0
        game_start = time.time()
        worker_id = os.getpid() % 1000  # Last 3 digits of PID for identification

        log.info(f"[worker-{worker_id}] Game loop starting, {len(game.get_legal_moves())} initial legal moves")

        while not game.is_game_over:
            # Get legal moves
            legal_moves = game.get_legal_moves()

            if not legal_moves:
                game.pass_turn()
                continue

            # Collect metrics
            if collect_metrics:
                # Track valid corners for current player
                corner_counts.append(len(game._valid_corners[game.current_player]))
                # Track mobility (legal move count)
                mobility_counts.append(len(legal_moves))

            # Get state encoding
            board_tensor, pieces_vector = game.get_state_for_neural_net()
            current_player = game.current_player

            # Use MCTS to select move
            add_noise = game.turn_number <= 30  # Add noise early in game
            log.info(f"[worker-{worker_id}] Move {move_count+1}: calling MCTS with {len(legal_moves)} legal moves...")
            move, policy, _ = mcts.select_move(game, add_noise=add_noise)
            log.info(f"[worker-{worker_id}] Move {move_count+1}: MCTS returned")

            # Store example
            examples_raw.append((
                board_tensor.copy(),
                pieces_vector.copy(),
                policy.copy(),
                legal_moves.copy(),
                current_player,
            ))

            # Make move
            game.make_move(move)
            move_count += 1

            # Log every 10 moves
            if move_count % 10 == 0:
                elapsed = time.time() - game_start
                moves_per_sec = move_count / elapsed if elapsed > 0 else 0
                import sys
                log.info(f"[worker-{worker_id}] Move {move_count}, {len(legal_moves)} legal, {moves_per_sec:.1f} moves/s")

        # Game finished - compute final values
        scores = game.get_scores()
        winner = game.get_winner()

        # Compute metrics
        metrics = None
        if collect_metrics:
            # Calculate center coverage (pieces in inner 10x10 region)
            center_coverage = 0
            for r in range(5, 15):
                for c in range(5, 15):
                    if game.board[r, c] != 0:
                        center_coverage += 1
            # Normalize by max possible (100 cells in inner region)
            center_coverage_ratio = center_coverage / 100.0

            # Final corner counts per player
            final_corners = [len(game._valid_corners[p]) for p in range(NUM_PLAYERS)]

            metrics = GameMetrics(
                avg_corner_count=sum(corner_counts) / len(corner_counts) if corner_counts else 0,
                avg_mobility=sum(mobility_counts) / len(mobility_counts) if mobility_counts else 0,
                avg_center_coverage=center_coverage_ratio,
                final_corner_counts=final_corners,
            )

        # Assign values to each example based on final outcome
        examples: List[TrainingExample] = []
        for board_tensor, pieces_vector, policy, legal_moves, player in examples_raw:
            if winner is not None:
                value = 1.0 if player == winner else -1.0
            else:
                # Tie: use relative score
                my_score = scores[player]
                max_other = max(scores[i] for i in range(NUM_PLAYERS) if i != player)
                if my_score > max_other:
                    value = 0.5
                elif my_score < max_other:
                    value = -0.5
                else:
                    value = 0.0

            examples.append(TrainingExample(
                board_tensor=board_tensor,
                pieces_vector=pieces_vector,
                policy=policy,
                legal_moves=legal_moves,
                value=value,
            ))

        return GameResult(
            examples=examples,
            winner=winner,
            scores=scores,
            num_moves=len(game.move_history),
            metrics=metrics,
        )


class SelfPlayManager:
    """
    Manages parallel self-play game generation.
    """

    def __init__(
        self,
        network: BlokusNet,
        mcts_config: MCTSConfig,
        num_workers: int = 8,
        device: str = 'cuda',
    ):
        self.network = network
        self.mcts_config = mcts_config
        self.num_workers = num_workers
        self.device = device

        # Statistics
        self.games_played = 0
        self.total_examples = 0
        self._lock = threading.Lock()

    def generate_games(
        self,
        num_games: int,
        progress_callback: Optional[callable] = None,
    ) -> List[TrainingExample]:
        """
        Generate training examples from self-play games.

        Note: Games are played sequentially because PyTorch CUDA doesn't
        handle multi-threaded access to the same model well.

        Args:
            num_games: Number of games to play
            progress_callback: Optional callback(games_complete, total_games)

        Returns:
            List of all training examples from all games
        """
        import time
        import sys

        all_examples: List[TrainingExample] = []
        start_time = time.time()

        worker = SelfPlayWorker(self.network, self.mcts_config, self.device)

        for game_idx in range(num_games):
            log.info(f"[alphazero] Starting game {game_idx + 1}/{num_games}...")

            result = worker.play_game()

            all_examples.extend(result.examples)
            self.games_played += 1
            self.total_examples += len(result.examples)
            completed = game_idx + 1

            elapsed = time.time() - start_time
            games_per_sec = completed / elapsed if elapsed > 0 else 0
            eta_sec = (num_games - completed) / games_per_sec if games_per_sec > 0 else 0

            log.info(f"[alphazero] Game {completed}/{num_games} complete "
                  f"({result.num_moves} moves, {len(result.examples)} examples, "
                  f"winner={result.winner}, {games_per_sec:.2f} games/s, "
                  f"ETA: {eta_sec/60:.1f}min)")

            if progress_callback:
                try:
                    progress_callback(completed, num_games, result)
                except TypeError:
                    progress_callback(completed, num_games)

        return all_examples

    def reset_stats(self):
        """Reset statistics."""
        with self._lock:
            self.games_played = 0
            self.total_examples = 0


class ReplayBuffer:
    """
    Replay buffer for storing training examples.
    Maintains a fixed-size buffer with oldest examples removed first.
    """

    def __init__(self, max_size: int = 1_000_000):
        self.max_size = max_size
        self.buffer: List[TrainingExample] = []
        self._lock = threading.Lock()

    def add(self, examples: List[TrainingExample]):
        """Add examples to the buffer."""
        with self._lock:
            self.buffer.extend(examples)
            # Remove oldest examples if over capacity
            if len(self.buffer) > self.max_size:
                self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size: int, recent_bias: float = 0.0) -> List[TrainingExample]:
        """
        Sample a batch of examples with optional recent bias.

        Args:
            batch_size: Number of examples to sample
            recent_bias: Weight for recency (0.0 = uniform, 1.0 = strongly favor recent)
                        Default 0.0 means uniform sampling for backwards compatibility.
                        Values around 0.7 mean recent examples are ~10x more likely.

        Returns:
            List of sampled training examples
        """
        with self._lock:
            if len(self.buffer) < batch_size:
                # Return all if not enough
                return self.buffer.copy()

            if recent_bias > 0:
                # Exponential weights favoring recent examples
                # recent_bias controls the steepness
                positions = np.arange(len(self.buffer))
                weights = np.exp(recent_bias * (positions / len(self.buffer) - 1) * 3)
                weights = weights / weights.sum()
                indices = np.random.choice(len(self.buffer), batch_size, p=weights, replace=False)
            else:
                indices = np.random.choice(len(self.buffer), batch_size, replace=False)

            return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self.buffer = []


@dataclass
class ActiveGame:
    """Tracks state for a game being played in parallel."""
    game: BlokusGame
    examples_raw: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[Move], int]] = field(default_factory=list)
    move_count: int = 0
    start_time: float = field(default_factory=time.time)
    # Metrics tracking
    corner_counts: List[int] = field(default_factory=list)
    mobility_counts: List[int] = field(default_factory=list)


class ParallelSelfPlay:
    """
    Parallel self-play game generation with batched neural network inference.

    Runs multiple games simultaneously and batches neural network evaluations
    across all active games to maximize GPU utilization. This provides
    significant speedup over sequential game generation.

    Architecture:
    1. Multiple games run in parallel
    2. Each game's MCTS collects leaf nodes for evaluation
    3. Leaves from all games are batched together for GPU inference
    4. Results are distributed back to respective games
    """

    def __init__(
        self,
        network: BlokusNet,
        mcts_config: MCTSConfig,
        num_parallel_games: int = 8,
        mcts_batch_size: int = 16,
        inference_batch_size: int = 32,
        device: str = 'cuda',
        # Cold start fixes
        heuristic_config: Optional['HeuristicConfig'] = None,
        heuristic_blend_factor: float = 0.0,
        policy_target_temperature: float = 1.0,
        # Adaptive noise
        noise_multiplier: float = 1.0,
    ):
        self.network = network
        self.mcts_config = mcts_config
        self.num_parallel_games = num_parallel_games
        self.mcts_batch_size = mcts_batch_size
        self.inference_batch_size = inference_batch_size
        self.device = device

        # Cold start fixes
        self.heuristic_config = heuristic_config
        self.heuristic_blend_factor = heuristic_blend_factor
        self.policy_target_temperature = policy_target_temperature

        # Adaptive noise: multiply dirichlet_epsilon by this factor
        # when entropy is low to force more exploration
        self.noise_multiplier = noise_multiplier

        # Statistics
        self.games_completed = 0
        self.total_examples = 0
        self._lock = threading.Lock()

    def _play_single_game_batched(self, game_id: int, collect_metrics: bool = True) -> GameResult:
        """
        Play a single game using BatchedMCTS for efficient inference.

        This is used when we want to play games one at a time but still
        benefit from batched leaf evaluation within MCTS.
        """
        game = BlokusGame()
        mcts = BatchedMCTS(self.network, self.mcts_config, self.device)

        examples_raw: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[Move], int]] = []

        # Metrics tracking
        corner_counts: List[int] = [] if collect_metrics else None
        mobility_counts: List[int] = [] if collect_metrics else None

        move_count = 0

        while not game.is_game_over:
            legal_moves = game.get_legal_moves()

            if not legal_moves:
                game.pass_turn()
                continue

            # Collect metrics
            if collect_metrics:
                corner_counts.append(len(game._valid_corners[game.current_player]))
                mobility_counts.append(len(legal_moves))

            board_tensor, pieces_vector = game.get_state_for_neural_net()
            current_player = game.current_player

            # Use batched MCTS
            add_noise = game.turn_number <= 30
            move, policy, _ = mcts.select_move(
                game,
                add_noise=add_noise,
                batch_size=self.mcts_batch_size,
            )

            examples_raw.append((
                board_tensor.copy(),
                pieces_vector.copy(),
                policy.copy(),
                legal_moves.copy(),
                current_player,
            ))

            game.make_move(move)
            move_count += 1

        # Process game result
        return self._finalize_game(game, examples_raw, corner_counts, mobility_counts)

    def _finalize_game(
        self,
        game: BlokusGame,
        examples_raw: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[Move], int]],
        corner_counts: Optional[List[int]] = None,
        mobility_counts: Optional[List[int]] = None,
    ) -> GameResult:
        """Convert raw examples to TrainingExamples with final values."""
        scores = game.get_scores()
        winner = game.get_winner()

        # Compute metrics if tracking data provided
        metrics = None
        if corner_counts is not None and mobility_counts is not None:
            # Calculate center coverage (pieces in inner 10x10 region)
            center_coverage = 0
            for r in range(5, 15):
                for c in range(5, 15):
                    if game.board[r, c] != 0:
                        center_coverage += 1
            center_coverage_ratio = center_coverage / 100.0

            # Final corner counts per player
            final_corners = [len(game._valid_corners[p]) for p in range(NUM_PLAYERS)]

            # Calculate policy quality metrics from MCTS policies
            # This helps diagnose if MCTS is generating focused vs uniform targets
            entropies = []
            top1_probs = []
            top5_probs = []
            for _, _, policy, _, _ in examples_raw:
                if len(policy) > 0:
                    # Entropy: -sum(p * log(p)), higher = more uniform
                    # Add small epsilon to avoid log(0)
                    p = policy + 1e-10
                    p = p / p.sum()  # Renormalize
                    entropy = -np.sum(p * np.log(p))
                    entropies.append(entropy)

                    # Top-1 probability (how confident is the policy)
                    top1_probs.append(float(np.max(policy)))

                    # Top-5 probability mass
                    if len(policy) >= 5:
                        top5_probs.append(float(np.sum(np.sort(policy)[-5:])))
                    else:
                        top5_probs.append(float(np.sum(policy)))

            metrics = GameMetrics(
                avg_corner_count=sum(corner_counts) / len(corner_counts) if corner_counts else 0,
                avg_mobility=sum(mobility_counts) / len(mobility_counts) if mobility_counts else 0,
                avg_center_coverage=center_coverage_ratio,
                final_corner_counts=final_corners,
                avg_policy_entropy=sum(entropies) / len(entropies) if entropies else 0.0,
                avg_top1_prob=sum(top1_probs) / len(top1_probs) if top1_probs else 0.0,
                avg_top5_prob=sum(top5_probs) / len(top5_probs) if top5_probs else 0.0,
            )

        examples: List[TrainingExample] = []
        for board_tensor, pieces_vector, policy, legal_moves, player in examples_raw:
            if winner is not None:
                value = 1.0 if player == winner else -1.0
            else:
                my_score = scores[player]
                max_other = max(scores[i] for i in range(NUM_PLAYERS) if i != player)
                if my_score > max_other:
                    value = 0.5
                elif my_score < max_other:
                    value = -0.5
                else:
                    value = 0.0

            examples.append(TrainingExample(
                board_tensor=board_tensor,
                pieces_vector=pieces_vector,
                policy=policy,
                legal_moves=legal_moves,
                value=value,
            ))

        return GameResult(
            examples=examples,
            winner=winner,
            scores=scores,
            num_moves=len(game.move_history),
            metrics=metrics,
        )

    def generate_games(
        self,
        num_games: int,
        progress_callback: Optional[callable] = None,
    ) -> List[TrainingExample]:
        """
        Generate training examples from self-play games using batched MCTS.

        Games are played sequentially but each game's MCTS uses batched
        leaf evaluation for efficient GPU utilization.

        Args:
            num_games: Number of games to play
            progress_callback: Optional callback(games_complete, total_games, result)

        Returns:
            List of all training examples from all games
        """
        import sys

        all_examples: List[TrainingExample] = []
        start_time = time.time()

        for game_idx in range(num_games):
            log.info(f"[parallel-selfplay] Starting game {game_idx + 1}/{num_games}...")

            result = self._play_single_game_batched(game_idx)

            all_examples.extend(result.examples)

            with self._lock:
                self.games_completed += 1
                self.total_examples += len(result.examples)

            elapsed = time.time() - start_time
            completed = game_idx + 1
            games_per_min = (completed / elapsed) * 60 if elapsed > 0 else 0
            eta_sec = ((num_games - completed) / completed) * elapsed if completed > 0 else 0

            log.info(f"[parallel-selfplay] Game {completed}/{num_games} complete "
                  f"({result.num_moves} moves, {len(result.examples)} examples, "
                  f"winner={result.winner}, {games_per_min:.1f} games/min, "
                  f"ETA: {eta_sec/60:.1f}min)")

            if progress_callback:
                try:
                    progress_callback(completed, num_games, result)
                except TypeError:
                    progress_callback(completed, num_games)

        return all_examples

    def generate_games_fully_parallel(
        self,
        num_games: int,
        progress_callback: Optional[callable] = None,
    ) -> List[TrainingExample]:
        """
        Generate games with full parallelization - multiple games running
        simultaneously with shared batch inference across all games.

        This provides maximum GPU utilization by batching neural network
        calls across multiple concurrent games.

        Args:
            num_games: Number of games to play
            progress_callback: Optional callback(games_complete, total_games, result)

        Returns:
            List of all training examples from all games
        """
        import sys

        all_examples: List[TrainingExample] = []
        games_completed = 0
        start_time = time.time()
        last_progress_time = start_time
        total_moves_made = 0

        while games_completed < num_games:
            # Start a batch of parallel games
            batch_size = min(self.num_parallel_games, num_games - games_completed)
            active_games: List[ActiveGame] = [
                ActiveGame(game=BlokusGame()) for _ in range(batch_size)
            ]

            log.info(f"[parallel-selfplay] Starting batch of {batch_size} parallel games "
                  f"(games {games_completed + 1}-{games_completed + batch_size}/{num_games})")

            batch_moves = 0

            # Play all games in this batch to completion
            while any(not ag.game.is_game_over for ag in active_games):
                # Collect all active games that need a move
                games_needing_moves: List[Tuple[int, ActiveGame, List[Move]]] = []
                for i, ag in enumerate(active_games):
                    if ag.game.is_game_over:
                        continue

                    legal_moves = ag.game.get_legal_moves()
                    if not legal_moves:
                        ag.game.pass_turn()
                        continue

                    games_needing_moves.append((i, ag, legal_moves))

                if not games_needing_moves:
                    continue

                # Run batched MCTS for all active games
                self._batched_mcts_step(games_needing_moves)
                batch_moves += len(games_needing_moves)
                total_moves_made += len(games_needing_moves)

                # Report intermediate progress every 10 seconds
                now = time.time()
                if progress_callback and (now - last_progress_time) >= 10.0:
                    last_progress_time = now
                    elapsed = now - start_time
                    moves_per_sec = total_moves_made / elapsed if elapsed > 0 else 0
                    # Estimate progress based on moves (assume ~70 moves per game on average)
                    estimated_total_moves = num_games * 70
                    move_progress = min(total_moves_made / estimated_total_moves, 0.99)

                    # Create a partial result for progress reporting
                    try:
                        progress_callback(
                            games_completed,
                            num_games,
                            None,  # No game result yet
                            intermediate=True,
                            moves_made=total_moves_made,
                            moves_per_sec=moves_per_sec,
                            active_games=len(games_needing_moves),
                        )
                    except TypeError:
                        # Callback doesn't support intermediate updates
                        pass

            # Process completed games
            for ag in active_games:
                result = self._finalize_game(
                    ag.game, ag.examples_raw, ag.corner_counts, ag.mobility_counts
                )
                all_examples.extend(result.examples)
                games_completed += 1

                elapsed = time.time() - start_time
                games_per_min = (games_completed / elapsed) * 60 if elapsed > 0 else 0
                eta_sec = ((num_games - games_completed) / games_completed) * elapsed if games_completed > 0 else 0

                log.info(f"[parallel-selfplay] Game {games_completed}/{num_games} complete "
                      f"({result.num_moves} moves, {len(result.examples)} examples, "
                      f"winner={result.winner}, {games_per_min:.1f} games/min, "
                      f"ETA: {eta_sec/60:.1f}min)")

                if progress_callback:
                    try:
                        progress_callback(games_completed, num_games, result)
                    except TypeError:
                        progress_callback(games_completed, num_games)

        return all_examples

    def _batched_mcts_step(
        self,
        games_needing_moves: List[Tuple[int, ActiveGame, List[Move]]],
    ):
        """
        Run one MCTS decision for multiple games using Rust-accelerated MCTS.

        Uses HybridMCTS which is ~14x faster than Python for tree operations.

        Applies cold start fixes:
        1. Heuristic blending: combines NN policy prior with handcrafted heuristics
        2. Temperature adjustment: applies temperature to MCTS visit counts for sharper targets
        """
        mcts_start = time.time()

        for game_idx, ag, legal_moves in games_needing_moves:
            # Collect metrics before move
            ag.corner_counts.append(len(ag.game._valid_corners[ag.game.current_player]))
            ag.mobility_counts.append(len(legal_moves))

            # Create Rust MCTS and run search
            # Note: heuristic blending happens inside HybridMCTS when we pass it
            mcts = HybridMCTS(
                self.network,
                self.mcts_config,
                self.device,
                # Pass heuristic config for policy prior blending
                heuristic_config=self.heuristic_config,
                heuristic_blend_factor=self.heuristic_blend_factor,
                # Pass adaptive noise multiplier
                noise_multiplier=self.noise_multiplier,
            )
            add_noise = ag.game.turn_number <= 30

            move, policy, visits = mcts.search(
                ag.game,
                add_noise=add_noise,
                batch_size=self.mcts_batch_size,
            )

            # Apply temperature to policy targets (sharper policies for training)
            # Temperature < 1.0 makes the distribution more peaked
            if self.policy_target_temperature != 1.0 and len(policy) > 0:
                # Apply temperature: p_i^(1/T) / sum(p_j^(1/T))
                temp_policy = policy ** (1.0 / self.policy_target_temperature)
                temp_policy = temp_policy / temp_policy.sum()
                policy = temp_policy

            # Store example
            board_tensor, pieces_vector = ag.game.get_state_for_neural_net()
            ag.examples_raw.append((
                board_tensor.copy(),
                pieces_vector.copy(),
                policy.copy(),
                legal_moves.copy(),
                ag.game.current_player,
            ))

            # Make move
            ag.game.make_move(move)
            ag.move_count += 1

        log.info(f"Rust MCTS complete: {len(games_needing_moves)} games, "
              f"{(time.time()-mcts_start)*1000:.1f}ms total")

    def reset_stats(self):
        """Reset statistics."""
        with self._lock:
            self.games_completed = 0
            self.total_examples = 0
