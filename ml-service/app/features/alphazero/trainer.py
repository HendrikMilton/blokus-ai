"""
Training loop for AlphaZero-style Blokus AI.

Coordinates:
1. Self-play game generation
2. Neural network training
3. Model evaluation
4. Checkpoint management
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np

from app.logging_config import log
from .game import BlokusGame, NUM_PLAYERS
from .network import BlokusNet, create_network
from .mcts import MCTSConfig
from .self_play import SelfPlayManager, ReplayBuffer, TrainingExample, ParallelSelfPlay
from .mcts_rust import HybridMCTS

log.info("Using Rust-accelerated MCTS (~14x faster tree operations)")


@dataclass
class TrainingConfig:
    """Configuration for AlphaZero training."""
    # Self-play
    games_per_generation: int = 100
    mcts_simulations: int = 400
    mcts_c_puct: float = 1.5
    temperature_threshold: int = 20
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    num_parallel_games: int = 8

    # Batching (for GPU efficiency)
    mcts_batch_size: int = 16        # Leaves to batch per MCTS search
    inference_batch_size: int = 32   # Max positions per GPU batch across games
    use_batched_mcts: bool = True    # Use batched MCTS (vs sequential)
    use_parallel_games: bool = True  # Run multiple games in parallel

    # Training
    batch_size: int = 4096
    learning_rate: float = 0.002
    lr_decay: str = 'cosine'
    training_steps_per_gen: int = 2000  # Used if adaptive is disabled
    training_steps_min: int = 500       # Adaptive: min steps (small buffer)
    training_steps_max: int = 2000      # Adaptive: max steps (full buffer)
    adaptive_training_steps: bool = True  # Enable adaptive scaling
    replay_buffer_size: int = 1_000_000
    mixed_precision: bool = True

    # Network
    num_res_blocks: int = 15
    channels: int = 256

    # Evaluation
    eval_games: int = 100
    promotion_threshold: float = 0.55

    # Limits
    max_generations: int = 100

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary (with camelCase keys)."""
        # Map camelCase to snake_case
        mapping = {
            'gamesPerGeneration': 'games_per_generation',
            'mctsSimulations': 'mcts_simulations',
            'mctsCPuct': 'mcts_c_puct',
            'temperatureThreshold': 'temperature_threshold',
            'dirichletAlpha': 'dirichlet_alpha',
            'dirichletEpsilon': 'dirichlet_epsilon',
            'numParallelGames': 'num_parallel_games',
            'mctsBatchSize': 'mcts_batch_size',
            'inferenceBatchSize': 'inference_batch_size',
            'useBatchedMcts': 'use_batched_mcts',
            'useParallelGames': 'use_parallel_games',
            'batchSize': 'batch_size',
            'learningRate': 'learning_rate',
            'lrDecay': 'lr_decay',
            'trainingStepsPerGen': 'training_steps_per_gen',
            'trainingStepsMin': 'training_steps_min',
            'trainingStepsMax': 'training_steps_max',
            'adaptiveTrainingSteps': 'adaptive_training_steps',
            'replayBufferSize': 'replay_buffer_size',
            'mixedPrecision': 'mixed_precision',
            'numResBlocks': 'num_res_blocks',
            'channels': 'channels',
            'evalGames': 'eval_games',
            'promotionThreshold': 'promotion_threshold',
            'maxGenerations': 'max_generations',
        }

        kwargs = {}
        for camel, snake in mapping.items():
            if camel in d:
                kwargs[snake] = d[camel]
            elif snake in d:
                kwargs[snake] = d[snake]

        return cls(**kwargs)


class AlphaZeroTrainer:
    """
    Main training loop for AlphaZero.
    """

    def __init__(
        self,
        run_id: str,
        config: TrainingConfig,
        models_dir: Path,
        api_base_url: str,
        device: str = 'cuda',
    ):
        self.run_id = run_id
        self.config = config
        self.models_dir = models_dir
        self.api_base_url = api_base_url
        self.device = device

        # Create model directory
        self.run_dir = models_dir / 'alphazero' / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize network
        self.network = create_network(
            num_res_blocks=config.num_res_blocks,
            channels=config.channels,
            device=device,
        )

        # Best network (for evaluation)
        self.best_network = create_network(
            num_res_blocks=config.num_res_blocks,
            channels=config.channels,
            device=device,
        )
        # Copy initial weights
        self.best_network.load_state_dict(self.network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

        # Learning rate scheduler
        if config.lr_decay == 'cosine':
            total_steps = config.max_generations * config.training_steps_per_gen
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        else:
            self.scheduler = None

        # Mixed precision
        self.scaler = GradScaler('cuda') if config.mixed_precision else None

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)

        # MCTS config
        self.mcts_config = MCTSConfig(
            num_simulations=config.mcts_simulations,
            c_puct=config.mcts_c_puct,
            temperature_threshold=config.temperature_threshold,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
        )

        # Training state
        self.current_generation = 0
        self.total_games = 0
        self.total_steps = 0
        self.best_win_rate = 0.0
        self._stop_requested = False

        # Timing for ETA calculation
        self.training_start_time: Optional[float] = None
        self.generation_start_time: Optional[float] = None
        self.phase_start_time: Optional[float] = None
        self.games_times: List[float] = []  # Track time per game for better ETA

        # Track actual phase durations from completed generations for accurate ETA
        self.completed_gen_durations: List[float] = []  # Total time per completed generation
        self.phase_durations: Dict[str, List[float]] = {
            'self_play': [],
            'training': [],
            'evaluation': [],
        }
        self.current_phase_durations: Dict[str, float] = {}  # Durations within current generation

    def _report_progress(
        self,
        phase: str,
        phase_progress: float,
        phase_current: int = 0,
        phase_total: int = 0,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        message: str = '',
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Report detailed progress to API with ETA."""
        import httpx

        now = time.time()

        # Calculate phase ETA
        phase_eta_sec = 0.0
        phase_elapsed = 0.0
        if self.phase_start_time and phase_progress > 0:
            phase_elapsed = now - self.phase_start_time
            phase_eta_sec = (phase_elapsed / phase_progress) * (1 - phase_progress)

        # Calculate generation ETA based on actual measured durations
        gen_eta_sec = 0.0
        gen_progress = 0.0
        gen_elapsed = now - self.generation_start_time if self.generation_start_time else 0.0

        if self.generation_start_time:
            # Get phase weights from actual measurements or use defaults
            if self.completed_gen_durations:
                # Use actual measured durations from completed generations
                avg_self_play = sum(self.phase_durations['self_play']) / len(self.phase_durations['self_play']) if self.phase_durations['self_play'] else 0
                avg_training = sum(self.phase_durations['training']) / len(self.phase_durations['training']) if self.phase_durations['training'] else 0
                avg_eval = sum(self.phase_durations['evaluation']) / len(self.phase_durations['evaluation']) if self.phase_durations['evaluation'] else 0
                avg_total = avg_self_play + avg_training + avg_eval

                if avg_total > 0:
                    w_self_play = avg_self_play / avg_total
                    w_training = avg_training / avg_total
                    w_eval = avg_eval / avg_total
                else:
                    w_self_play, w_training, w_eval = 0.6, 0.3, 0.1
            else:
                # Default weights for first generation
                w_self_play, w_training, w_eval = 0.6, 0.3, 0.1

            # Calculate gen_progress based on completed phases + current phase progress
            completed_weight = 0.0
            for p, w in [('self_play', w_self_play), ('training', w_training), ('evaluation', w_eval)]:
                if p in self.current_phase_durations:
                    completed_weight += w
                elif p == phase:
                    completed_weight += phase_progress * w
                    break

            gen_progress = completed_weight

            if gen_progress > 0:
                gen_eta_sec = (gen_elapsed / gen_progress) * (1 - gen_progress)

        # Calculate overall ETA based on completed generation times
        overall_eta_sec = 0.0
        overall_progress = 0.0
        if self.training_start_time and self.current_generation > 0:
            completed_gens = self.current_generation - 1
            total_gens = self.config.max_generations

            if self.completed_gen_durations:
                # Use actual average generation time for remaining generations
                avg_gen_time = sum(self.completed_gen_durations) / len(self.completed_gen_durations)
                remaining_full_gens = total_gens - self.current_generation
                # Time remaining = remaining full generations * avg + current gen remaining
                overall_eta_sec = remaining_full_gens * avg_gen_time + gen_eta_sec
                overall_progress = (completed_gens + gen_progress) / total_gens
            else:
                # First generation: estimate based on gen_eta_sec
                # We don't have completed generations yet, so extrapolate from current gen
                overall_progress = gen_progress / total_gens
                remaining_full_gens = total_gens - 1  # Excluding current generation
                # Estimate: current gen remaining + (remaining gens * estimated gen time)
                if gen_progress > 0:
                    estimated_gen_time = gen_elapsed / gen_progress  # Estimated total time for this gen
                    overall_eta_sec = gen_eta_sec + remaining_full_gens * estimated_gen_time
                else:
                    overall_eta_sec = 0.0

        try:
            url = f"{self.api_base_url}/api/ml/alphazero/runs/{self.run_id}/progress"
            payload = {
                # Current state
                'generation': self.current_generation,
                'maxGenerations': self.config.max_generations,
                'gamesGenerated': self.total_games,
                'trainingSteps': self.total_steps,
                'policyLoss': policy_loss if policy_loss else None,
                'valueLoss': value_loss if value_loss else None,
                # Phase info
                'phase': phase,
                'phaseCurrent': phase_current,
                'phaseTotal': phase_total,
                'phaseProgress': round(phase_progress * 100, 1),
                'phaseEtaSec': round(phase_eta_sec),
                # Generation info
                'generationProgress': round(gen_progress * 100, 1),
                'generationEtaSec': round(gen_eta_sec),
                # Overall info
                'overallProgress': round(overall_progress * 100, 1),
                'overallEtaSec': round(overall_eta_sec),
                # Human-readable message
                'message': message,
                # Extra data (e.g., games per second, etc.)
                **(extra or {}),
            }
            httpx.post(url, json=payload, timeout=5.0)
        except Exception as e:
            log.info(f"Failed to report progress: {e}")

    def _report_checkpoint(
        self,
        generation: int,
        step: int,
        policy_loss: float,
        value_loss: float,
        eval_win_rate: Optional[float],
        eval_games: Optional[int],
        is_best: bool,
        model_path: str,
    ):
        """Report checkpoint to API."""
        import httpx

        try:
            url = f"{self.api_base_url}/api/ml/alphazero/runs/{self.run_id}/checkpoint"
            httpx.post(url, json={
                'generation': generation,
                'step': step,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'eval_win_rate': eval_win_rate,
                'eval_games': eval_games,
                'is_best': is_best,
                'model_path': model_path,
            }, timeout=5.0)
        except Exception as e:
            log.info(f"Failed to report checkpoint: {e}")

    def _report_completion(self, policy_loss: float, value_loss: float, best_win_rate: float):
        """Report training completion to API."""
        import httpx

        try:
            url = f"{self.api_base_url}/api/ml/alphazero/runs/{self.run_id}/complete"
            httpx.post(url, json={
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'best_win_rate': best_win_rate,
            }, timeout=10.0)
        except Exception as e:
            log.info(f"Failed to report completion: {e}")

    def _report_failure(self, error_message: str):
        """Report training failure to API."""
        import httpx

        try:
            url = f"{self.api_base_url}/api/ml/alphazero/runs/{self.run_id}/failed"
            httpx.post(url, json={
                'error_message': error_message,
            }, timeout=10.0)
        except Exception as e:
            log.info(f"Failed to report failure: {e}")

    def _send_metrics_batch(self, metrics: List[Dict[str, Any]]):
        """Send batch of training metrics to API for storage."""
        import httpx

        if not metrics:
            return

        try:
            url = f"{self.api_base_url}/api/ml/alphazero/runs/{self.run_id}/metrics"
            httpx.post(url, json={'metrics': metrics}, timeout=10.0)
        except Exception as e:
            log.info(f"Failed to send metrics batch: {e}")

    def stop(self):
        """Request training to stop."""
        self._stop_requested = True

    def _record_phase_duration(self, phase: str):
        """Record the duration of a completed phase."""
        if self.phase_start_time:
            duration = time.time() - self.phase_start_time
            self.current_phase_durations[phase] = duration
            self.phase_durations[phase].append(duration)
            log.info(f"Phase {phase} completed in {duration:.1f}s")

    def _record_generation_complete(self):
        """Record total duration of completed generation."""
        if self.generation_start_time:
            duration = time.time() - self.generation_start_time
            self.completed_gen_durations.append(duration)
            log.info(f"Generation {self.current_generation} completed in {duration:.1f}s")

    def train(self) -> bool:
        """
        Run the full training loop.

        Returns:
            True if completed successfully, False if stopped/failed
        """
        log.info(f"Starting training run: {self.run_id}")
        log.info(f"Config: {self.config}")

        self.training_start_time = time.time()

        try:
            for gen in range(self.config.max_generations):
                if self._stop_requested:
                    log.info("Stop requested, exiting")
                    return False

                self.current_generation = gen + 1
                self.generation_start_time = time.time()
                self.current_phase_durations = {}  # Reset for new generation
                log.info(f"=== Generation {self.current_generation}/{self.config.max_generations} ===")

                # Phase 1: Self-play (move best_network to CPU to save VRAM)
                log.info("Phase 1: Self-play")
                self.best_network.cpu()
                torch.cuda.empty_cache()
                self._run_self_play()
                self._record_phase_duration('self_play')

                if self._stop_requested:
                    return False

                # Clear GPU memory before training phase
                import gc
                import ctypes

                # Move network to CPU temporarily to force CUDA to release memory
                self.network.cpu()

                # Multiple rounds of cleanup to ensure all references are freed
                for _ in range(3):
                    gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Check memory before continuing
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                log.info(f"GPU memory after initial cleanup: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

                # If still using significant memory, try harder
                if reserved > 1.0:
                    log.info("Memory still high, forcing additional cleanup...")
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Release memory back to OS (Linux glibc malloc_trim)
                    try:
                        libc = ctypes.CDLL("libc.so.6")
                        libc.malloc_trim(0)
                    except Exception:
                        pass

                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    log.info(f"GPU memory after aggressive cleanup: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

                # Move network back to GPU for training
                self.network.to(self.device)

                # Phase 2: Training
                log.info("Phase 2: Training")
                policy_loss, value_loss = self._run_training()
                self._record_phase_duration('training')

                if self._stop_requested:
                    return False

                # Phase 3: Evaluation (move best_network back to GPU)
                log.info("Phase 3: Evaluation")
                self.best_network.to(self.device)
                win_rate, is_best = self._run_evaluation()
                self._record_phase_duration('evaluation')
                self._record_generation_complete()

                # Track best win rate (regardless of promotion)
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate

                # Save checkpoint
                model_path = self._save_checkpoint(policy_loss, value_loss, win_rate, is_best)
                log.info(f"Checkpoint saved: {model_path}")

                if is_best:
                    # Update best network (promoted)
                    self.best_network.load_state_dict(self.network.state_dict())
                    log.info(f"New best model! Win rate: {win_rate:.2%}")

            # Training completed
            log.info("Training completed!")
            self._report_completion(policy_loss, value_loss, self.best_win_rate)
            return True

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            log.info(f"Training failed: {error_msg}")
            self._report_failure(error_msg)
            return False

    def _run_self_play(self):
        """Generate self-play games using batched MCTS for GPU efficiency."""
        import sys
        self.phase_start_time = time.time()
        self.games_times = []

        # Choose between batched parallel or legacy sequential
        use_batched = self.config.use_batched_mcts
        use_parallel = self.config.use_parallel_games

        log.info(f"Starting self-play: {self.config.games_per_generation} games, "
              f"{self.config.mcts_simulations} MCTS sims, "
              f"batched={use_batched}, parallel={use_parallel}, "
              f"mcts_batch={self.config.mcts_batch_size}, "
              f"parallel_games={self.config.num_parallel_games}")

        last_game_time = time.time()
        last_intermediate_report = time.time()

        # Collect metrics from all games for aggregated logging
        game_metrics_list = []

        def progress_callback(completed, total, game_result=None, **kwargs):
            nonlocal last_game_time, last_intermediate_report
            now = time.time()

            # Handle intermediate progress updates (during game, before completion)
            if kwargs.get('intermediate'):
                moves_made = kwargs.get('moves_made', 0)
                moves_per_sec = kwargs.get('moves_per_sec', 0)
                active_games = kwargs.get('active_games', 0)

                # Estimate progress based on moves (assume ~70 moves per game)
                estimated_total_moves = total * 70
                move_progress = min(moves_made / estimated_total_moves, 0.95) if estimated_total_moves > 0 else 0

                message = f"Playing... {moves_made} moves ({moves_per_sec:.1f}/s), {active_games} active games"

                self._report_progress(
                    phase='self_play',
                    phase_progress=move_progress,
                    phase_current=completed,
                    phase_total=total,
                    message=message,
                    extra={
                        'movesPerSec': round(moves_per_sec, 2),
                        'totalMoves': moves_made,
                        'activeGames': active_games,
                        'batchedMcts': use_batched,
                        'parallelGames': use_parallel,
                    }
                )
                return

            # Collect metrics from completed game
            if game_result and game_result.metrics:
                game_metrics_list.append(game_result.metrics)

            # Track game completion time
            game_time = now - last_game_time
            self.games_times.append(game_time)
            last_game_time = now

            self.total_games = (self.current_generation - 1) * self.config.games_per_generation + completed

            # Calculate games per second (average of last 5 games)
            recent_times = self.games_times[-5:] if len(self.games_times) >= 5 else self.games_times
            avg_game_time = sum(recent_times) / len(recent_times) if recent_times else 0
            games_per_min = 60 / avg_game_time if avg_game_time > 0 else 0

            # Build message
            game_info = ""
            if game_result:
                game_info = f" ({game_result.num_moves} moves, winner={game_result.winner})"

            message = f"Game {completed}/{total}{game_info} - {games_per_min:.1f} games/min"

            self._report_progress(
                phase='self_play',
                phase_progress=completed / total,
                phase_current=completed,
                phase_total=total,
                message=message,
                extra={
                    'gamesPerMin': round(games_per_min, 2),
                    'avgGameTimeSec': round(avg_game_time, 1),
                    'lastGameMoves': game_result.num_moves if game_result else None,
                    'lastGameExamples': len(game_result.examples) if game_result else None,
                    'batchedMcts': use_batched,
                    'parallelGames': use_parallel,
                }
            )

        if use_batched:
            # Use new ParallelSelfPlay with batched MCTS
            manager = ParallelSelfPlay(
                network=self.network,
                mcts_config=self.mcts_config,
                num_parallel_games=self.config.num_parallel_games,
                mcts_batch_size=self.config.mcts_batch_size,
                inference_batch_size=self.config.inference_batch_size,
                device=self.device,
            )

            if use_parallel:
                # Full parallelization - multiple games with shared batching
                examples = manager.generate_games_fully_parallel(
                    num_games=self.config.games_per_generation,
                    progress_callback=progress_callback,
                )
            else:
                # Sequential games but batched MCTS within each game
                examples = manager.generate_games(
                    num_games=self.config.games_per_generation,
                    progress_callback=progress_callback,
                )
        else:
            # Legacy sequential MCTS (for comparison/debugging)
            manager = SelfPlayManager(
                network=self.network,
                mcts_config=self.mcts_config,
                num_workers=self.config.num_parallel_games,
                device=self.device,
            )
            examples = manager.generate_games(
                num_games=self.config.games_per_generation,
                progress_callback=progress_callback,
            )

        self.replay_buffer.add(examples)
        elapsed = time.time() - self.phase_start_time
        games_per_min = (self.config.games_per_generation / elapsed) * 60 if elapsed > 0 else 0
        log.info(f"Generated {len(examples)} examples from {self.config.games_per_generation} games "
              f"in {elapsed:.1f}s ({games_per_min:.1f} games/min)")
        log.info(f"Replay buffer size: {len(self.replay_buffer)}")

        # Log aggregated heuristic metrics for diagnostics
        # Store on instance for use in training metrics
        self._last_heuristics = None
        if game_metrics_list:
            avg_corners = sum(m.avg_corner_count for m in game_metrics_list) / len(game_metrics_list)
            avg_mobility = sum(m.avg_mobility for m in game_metrics_list) / len(game_metrics_list)
            avg_center = sum(m.avg_center_coverage for m in game_metrics_list) / len(game_metrics_list)

            # Store for training metrics
            self._last_heuristics = {
                'avg_corners': avg_corners,
                'avg_mobility': avg_mobility,
                'center_coverage': avg_center,
            }

            # Average final corners per player across all games
            avg_final_corners = [
                sum(m.final_corner_counts[p] for m in game_metrics_list) / len(game_metrics_list)
                for p in range(4)
            ]

            log.info(f"[Heuristics] Gen {self.current_generation}: "
                  f"avg_corners={avg_corners:.1f}, avg_mobility={avg_mobility:.1f}, "
                  f"center_coverage={avg_center:.2%}, final_corners={[f'{c:.1f}' for c in avg_final_corners]}")

            # Report metrics via progress API for frontend display
            self._report_progress(
                phase='self_play',
                phase_progress=1.0,
                phase_current=self.config.games_per_generation,
                phase_total=self.config.games_per_generation,
                message='Self-play complete',
                extra={
                    'heuristics': {
                        'avgCorners': round(avg_corners, 2),
                        'avgMobility': round(avg_mobility, 2),
                        'centerCoverage': round(avg_center, 4),
                        'finalCorners': [round(c, 2) for c in avg_final_corners],
                    }
                }
            )

        # Clean up self-play objects to free memory before training
        del examples
        del manager

        # Force cleanup of any cached tensors
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def _get_training_steps(self) -> int:
        """Calculate training steps based on buffer size (adaptive) or fixed config."""
        if not self.config.adaptive_training_steps:
            return self.config.training_steps_per_gen

        # Scale steps linearly with buffer fullness
        # At 0% full -> training_steps_min
        # At 100% full -> training_steps_max
        buffer_ratio = min(len(self.replay_buffer) / self.config.replay_buffer_size, 1.0)
        steps = int(
            self.config.training_steps_min +
            buffer_ratio * (self.config.training_steps_max - self.config.training_steps_min)
        )
        return steps

    def _run_training(self) -> tuple[float, float]:
        """Train the network on replay buffer."""
        self.phase_start_time = time.time()
        self.network.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0

        # Calculate adaptive training steps
        training_steps = self._get_training_steps()
        log.info(f"Training steps this generation: {training_steps} "
                 f"(buffer: {len(self.replay_buffer)}/{self.config.replay_buffer_size})")

        # Collect metrics for batch insert to database
        metrics_batch = []
        last_step_time = time.time()

        for step in range(training_steps):
            if self._stop_requested:
                break

            # Sample batch
            batch = self.replay_buffer.sample(self.config.batch_size)
            if len(batch) < self.config.batch_size // 2 and step == 0:
                log.info(f"Warning: only {len(batch)} examples in buffer (batch_size={self.config.batch_size})")

            # Prepare batch tensors - stack numpy arrays first, then convert once
            board_np = np.stack([ex.board_tensor for ex in batch])
            pieces_np = np.stack([ex.pieces_vector for ex in batch])
            values_np = np.array([ex.value for ex in batch], dtype=np.float32)

            board_tensors = torch.from_numpy(board_np).to(self.device)
            pieces_vectors = torch.from_numpy(pieces_np).to(self.device)
            target_values = torch.from_numpy(values_np).unsqueeze(1).to(self.device)

            # Legal moves list for policy
            legal_moves_list = [ex.legal_moves for ex in batch]

            # Pre-compute target policies - stack numpy then convert
            max_policy_len = max(len(ex.policy) for ex in batch) if batch else 1
            target_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
            mask_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
            for i, ex in enumerate(batch):
                plen = len(ex.policy)
                if plen > 0:
                    target_np[i, :plen] = ex.policy
                    mask_np[i, :plen] = 1.0

            target_policies = torch.from_numpy(target_np).to(self.device)
            policy_mask = torch.from_numpy(mask_np).to(self.device)

            self.optimizer.zero_grad()

            if self.config.mixed_precision:
                with autocast('cuda'):
                    policy_logits, values = self.network(
                        board_tensors, pieces_vectors, legal_moves_list
                    )
                    value_loss = F.mse_loss(values, target_values)

                    # Vectorized policy loss - no Python loop!
                    # Mask out invalid positions with -inf before softmax
                    masked_logits = policy_logits[:, :max_policy_len].clone()
                    masked_logits[policy_mask == 0] = float('-inf')
                    log_probs = F.log_softmax(masked_logits, dim=1)
                    # Replace NaN with 0 (can happen if all logits are -inf)
                    log_probs = torch.nan_to_num(log_probs, nan=0.0)
                    # Cross-entropy: -sum(target * log_probs) per sample, then mean
                    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

                    loss = policy_loss + value_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                policy_logits, values = self.network(
                    board_tensors, pieces_vectors, legal_moves_list
                )
                value_loss = F.mse_loss(values, target_values)

                # Vectorized policy loss - no Python loop!
                masked_logits = policy_logits[:, :max_policy_len].clone()
                masked_logits[policy_mask == 0] = float('-inf')
                log_probs = F.log_softmax(masked_logits, dim=1)
                # Replace NaN with 0 (can happen if all logits are -inf)
                log_probs = torch.nan_to_num(log_probs, nan=0.0)
                policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()

            if self.scheduler and step > 0:
                self.scheduler.step()

            p_item = policy_loss.item()
            v_item = value_loss.item()
            # Skip NaN losses
            if not (np.isnan(p_item) or np.isnan(v_item)):
                total_policy_loss += p_item
                total_value_loss += v_item
            self.total_steps += 1

            # Track step timing
            now = time.time()
            step_time_ms = (now - last_step_time) * 1000
            last_step_time = now

            # Collect metric for this step
            lr = self.optimizer.param_groups[0]['lr']
            metric = {
                'generation': self.current_generation,
                'step': step + 1,
                'policy_loss': p_item if not np.isnan(p_item) else None,
                'value_loss': v_item if not np.isnan(v_item) else None,
                'total_loss': (p_item + v_item) if not (np.isnan(p_item) or np.isnan(v_item)) else None,
                'learning_rate': lr,
                'step_time_ms': step_time_ms,
                'replay_buffer_size': len(self.replay_buffer),
            }
            # Include heuristics from self-play on first step of each generation
            if step == 0 and self._last_heuristics:
                metric['avg_corners'] = self._last_heuristics['avg_corners']
                metric['avg_mobility'] = self._last_heuristics['avg_mobility']
                metric['center_coverage'] = self._last_heuristics['center_coverage']
            metrics_batch.append(metric)

            # Report progress and send metrics every 50 steps
            if (step + 1) % 50 == 0:
                avg_p = total_policy_loss / (step + 1) if step > 0 else 0.0
                avg_v = total_value_loss / (step + 1) if step > 0 else 0.0
                elapsed = time.time() - self.phase_start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

                # Sanitize for JSON
                avg_p = 0.0 if np.isnan(avg_p) or np.isinf(avg_p) else avg_p
                avg_v = 0.0 if np.isnan(avg_v) or np.isinf(avg_v) else avg_v

                message = f"Step {step + 1}/{training_steps} - loss: {avg_p + avg_v:.4f}"

                self._report_progress(
                    phase='training',
                    phase_progress=(step + 1) / training_steps,
                    phase_current=step + 1,
                    phase_total=training_steps,
                    policy_loss=avg_p,
                    value_loss=avg_v,
                    message=message,
                    extra={
                        'stepsPerSec': round(steps_per_sec, 2),
                        'learningRate': lr,
                        'totalLoss': round(avg_p + avg_v, 4),
                        'adaptiveSteps': training_steps,
                    }
                )
                log.info(f"Step {step + 1}/{training_steps} - "
                      f"policy_loss: {avg_p:.4f}, value_loss: {avg_v:.4f}, {steps_per_sec:.1f} steps/s")

                # Send metrics batch to API
                self._send_metrics_batch(metrics_batch)
                metrics_batch = []

        # Send any remaining metrics
        if metrics_batch:
            self._send_metrics_batch(metrics_batch)

        avg_policy_loss = total_policy_loss / training_steps if training_steps > 0 else 0.0
        avg_value_loss = total_value_loss / training_steps if training_steps > 0 else 0.0

        return avg_policy_loss, avg_value_loss

    def _run_evaluation(self) -> tuple[float, bool]:
        """Evaluate new network against best network using batched MCTS."""
        self.phase_start_time = time.time()
        self.network.eval()
        self.best_network.eval()

        new_wins = 0
        games_played = 0

        for game_idx in range(self.config.eval_games):
            if self._stop_requested:
                break

            # Alternate who plays first
            new_player = game_idx % 2

            # Play a game
            game = BlokusGame()

            new_mcts = HybridMCTS(self.network, self.mcts_config, self.device)
            best_mcts = HybridMCTS(self.best_network, self.mcts_config, self.device)

            while not game.is_game_over:
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    game.pass_turn()
                    continue

                # Select which network to use
                current = game.current_player
                if current == new_player:
                    mcts = new_mcts
                else:
                    mcts = best_mcts

                # Select move (no noise during evaluation)
                move, _, _ = mcts.search(game, add_noise=False, batch_size=self.config.mcts_batch_size)
                game.make_move(move)

            # Check winner
            winner = game.get_winner()
            if winner == new_player:
                new_wins += 1

            games_played += 1
            current_win_rate = new_wins / games_played

            # Report progress every game
            message = f"Eval game {games_played}/{self.config.eval_games} - win rate: {current_win_rate:.1%}"

            self._report_progress(
                phase='evaluation',
                phase_progress=games_played / self.config.eval_games,
                phase_current=games_played,
                phase_total=self.config.eval_games,
                message=message,
                extra={
                    'evalWins': new_wins,
                    'evalGames': games_played,
                    'currentWinRate': round(current_win_rate, 3),
                    'promotionThreshold': self.config.promotion_threshold,
                    'rustMcts': True,
                }
            )

            if games_played % 5 == 0:
                log.info(f"Eval {games_played}/{self.config.eval_games}: {new_wins} wins ({current_win_rate:.1%})")

        win_rate = new_wins / games_played if games_played > 0 else 0.0
        is_best = win_rate >= self.config.promotion_threshold

        log.info(f"Evaluation complete: {new_wins}/{games_played} wins ({win_rate:.2%}) - "
              f"{'PROMOTED' if is_best else 'not promoted'}")

        return win_rate, is_best

    def _save_checkpoint(
        self,
        policy_loss: float,
        value_loss: float,
        eval_win_rate: float,
        is_best: bool,
    ) -> str:
        """Save model checkpoint."""
        checkpoint_name = f"gen{self.current_generation:04d}_step{self.total_steps}"
        checkpoint_path = self.run_dir / f"{checkpoint_name}.pt"

        torch.save({
            'generation': self.current_generation,
            'step': self.total_steps,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'eval_win_rate': eval_win_rate,
            'config': self.config.__dict__,
        }, checkpoint_path)

        # Also save as best if it is
        if is_best:
            best_path = self.run_dir / "best.pt"
            torch.save({
                'generation': self.current_generation,
                'step': self.total_steps,
                'model_state_dict': self.network.state_dict(),
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'eval_win_rate': eval_win_rate,
            }, best_path)

        # Report to API
        self._report_checkpoint(
            generation=self.current_generation,
            step=self.total_steps,
            policy_loss=policy_loss,
            value_loss=value_loss,
            eval_win_rate=eval_win_rate,
            eval_games=self.config.eval_games,
            is_best=is_best,
            model_path=str(checkpoint_path),
        )

        return str(checkpoint_path)


# Global stop flags for running trainers
_active_trainers: Dict[str, AlphaZeroTrainer] = {}
_trainer_lock = threading.Lock()


def start_training(run_id: str, config_dict: Dict[str, Any], models_dir: Path, api_base_url: str):
    """Start a new training run in the background."""
    config = TrainingConfig.from_dict(config_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")

    trainer = AlphaZeroTrainer(
        run_id=run_id,
        config=config,
        models_dir=models_dir,
        api_base_url=api_base_url,
        device=device,
    )

    with _trainer_lock:
        _active_trainers[run_id] = trainer

    try:
        trainer.train()
    finally:
        with _trainer_lock:
            _active_trainers.pop(run_id, None)


def stop_training(run_id: str):
    """Stop a running training."""
    with _trainer_lock:
        trainer = _active_trainers.get(run_id)
        if trainer:
            trainer.stop()
