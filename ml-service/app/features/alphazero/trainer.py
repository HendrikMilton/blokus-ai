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
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field

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
from .heuristics import (
    HeuristicConfig,
    compute_heuristic_policy,
    blend_policies,
    get_blend_factor_for_generation,
)

log.info("Using Rust-accelerated MCTS (~14x faster tree operations)")


@dataclass
class ELORating:
    """
    ELO rating system for model evaluation (from ELF OpenGo).

    Advantages over threshold-based promotion:
    - No arbitrary threshold (55% is meaningless)
    - Robust to lucky/unlucky gen 1 evaluation
    - Natural handling of improvement plateaus
    - Compares against multiple recent generations, not just "best"
    """
    ratings: Dict[int, float] = field(default_factory=dict)  # generation -> ELO rating
    k_factor: float = 32.0
    initial_rating: float = 1500.0

    def get_rating(self, generation: int) -> float:
        """Get ELO rating for a generation."""
        return self.ratings.get(generation, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A vs player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_rating(
        self,
        generation: int,
        opponent_generation: int,
        wins: int,
        losses: int,
        draws: int = 0,
    ) -> float:
        """
        Update ELO rating based on game results.

        Args:
            generation: Generation being evaluated
            opponent_generation: Generation played against
            wins: Number of wins for the new generation
            losses: Number of losses
            draws: Number of draws

        Returns:
            New ELO rating for the generation
        """
        total_games = wins + losses + draws
        if total_games == 0:
            return self.get_rating(generation)

        # Get current ratings
        rating_new = self.get_rating(generation)
        rating_opp = self.get_rating(opponent_generation)

        # Calculate actual score (win=1, draw=0.5, loss=0)
        actual_score = (wins + 0.5 * draws) / total_games

        # Calculate expected score
        expected = self.expected_score(rating_new, rating_opp)

        # Update rating
        new_rating = rating_new + self.k_factor * (actual_score - expected) * total_games
        self.ratings[generation] = new_rating

        return new_rating

    def get_recent_generations(self, current_gen: int, num_gens: int) -> List[int]:
        """Get list of recent generations that have ratings (excluding current)."""
        available = [g for g in self.ratings.keys() if g < current_gen]
        available.sort(reverse=True)
        return available[:num_gens]

    def should_promote(
        self,
        generation: int,
        threshold: float = 50.0,
    ) -> Tuple[bool, float]:
        """
        Determine if a generation should be promoted based on ELO improvement.

        Args:
            generation: Generation to evaluate
            threshold: Minimum ELO improvement to promote

        Returns:
            (should_promote, elo_improvement)
        """
        if generation <= 1:
            return True, 0.0  # Always promote gen 1

        current_rating = self.get_rating(generation)
        prev_rating = self.get_rating(generation - 1)
        improvement = current_rating - prev_rating

        return improvement >= threshold, improvement


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

    # Gumbel MCTS (from MuZero paper - guaranteed policy improvement)
    # Much more reliable than PUCT, works with fewer simulations
    use_gumbel_mcts: bool = True     # Use Gumbel MCTS instead of PUCT
    gumbel_c_visit: float = 50.0     # Visit count scale for action selection
    gumbel_c_scale: float = 1.0      # Scale for completed Q-values
    gumbel_max_actions: int = 16     # Top-k actions to consider

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

    # Gradient clipping (prevents explosion)
    max_grad_norm: float = 1.0       # Max gradient norm for clipping

    # Auxiliary losses (KataGo-style for 50x efficiency)
    # These provide denser gradients and faster credit assignment
    use_auxiliary_losses: bool = True
    aux_ownership_weight: float = 0.00375  # 1.5/400 from KataGo paper
    aux_score_weight: float = 0.02          # Score distribution loss
    aux_opponent_policy_weight: float = 0.15  # Predict opponent's response

    # Cold start fixes
    # 1. Heuristic bootstrapping: blend handcrafted policy with NN
    use_heuristic_bootstrap: bool = True
    heuristic_blend_initial: float = 0.5   # Initial blend (0=pure NN, 1=pure heuristic)
    heuristic_blend_decay_gens: int = 30   # Generations to decay to 0

    # 2. Increased MCTS simulations for early training
    mcts_simulations_initial: int = 1600   # Higher sims for first N generations
    mcts_simulations_decay_gens: int = 20  # Decay to base mcts_simulations over this many gens

    # 3. Temperature-adjusted targets for sharper policies
    policy_target_temperature: float = 0.5  # <1.0 makes MCTS targets more peaked

    # 4. Playout cap randomization (from KataGo - 50x efficiency improvement)
    # Randomly choose between full search (recorded) and fast search (not recorded)
    # This gives 4x more games for value training while keeping policy target quality
    use_playout_cap_randomization: bool = True
    playout_cap_fraction: float = 0.25     # Fraction of moves with full search
    playout_full_simulations: int = 600    # Simulations for full search (recorded)
    playout_fast_simulations: int = 100    # Simulations for fast search (not recorded)
    playout_fast_no_noise: bool = True     # No Dirichlet noise for fast moves

    # 5. Policy target pruning (from KataGo)
    # Removes exploration artifacts from MCTS policy targets
    # Subtracts "forced playouts" that were added for exploration, not value
    use_policy_target_pruning: bool = True
    policy_prune_threshold: float = 0.01   # Prune moves with prob < this after subtraction

    # 6. Warm-start rollout blending (from Warm-Start AlphaZero paper)
    # Early in training, blend NN value with heuristic rollout value
    # This provides meaningful signal before NN learns anything
    use_warm_start_rollouts: bool = True
    warm_start_initial_blend: float = 0.7  # Initial rollout weight (high = more rollout)
    warm_start_decay_gens: int = 5         # Generations to decay to pure NN

    # Network
    num_res_blocks: int = 15
    channels: int = 256

    # Evaluation
    eval_games: int = 200  # Reduced from 400 for faster iteration
    eval_parallel_games: int = 16  # Number of eval games to run in parallel
    heuristic_eval_games: int = 100  # Games against heuristic baseline
    promotion_threshold: float = 0.55

    # === Anti-Collapse Fixes (Research-Backed) ===
    # Note: Entropy regularization REMOVED - research shows it fights policy loss
    # and causes oscillation. Instead, we use Gumbel MCTS and auxiliary losses.

    # ELO-based evaluation (replaces threshold-based promotion)
    # From ELF OpenGo: more robust than win-rate threshold
    use_elo_evaluation: bool = True
    elo_initial: float = 1500.0            # Starting ELO for gen 1
    elo_k_factor: float = 32.0             # ELO K-factor for rating updates
    elo_promotion_threshold: float = 50.0  # Promote if ELO increases by this amount
    elo_compare_generations: int = 3       # Number of recent generations to compare against

    # Legacy threshold-based promotion (DEPRECATED - kept for backwards compatibility)
    dynamic_promotion: bool = False  # Disabled by default, use ELO instead
    promotion_threshold_min: float = 0.45   # Minimum threshold
    promotion_decay_per_stuck_gen: float = 0.02  # Lower by 2% per stuck generation

    # 3. Recent-weighted replay sampling
    replay_recent_bias: float = 0.7  # Bias toward recent examples (0=uniform, 1=strong)

    # 4. Sublinear replay window (from KataGo)
    # Prevents early overfitting by limiting how much old data is used
    # Window grows sublinearly: N_window = c(1 + β(N_total/c)^α - 1)/α
    use_sublinear_window: bool = True
    sublinear_window_c: float = 250000.0  # Initial window constant
    sublinear_window_alpha: float = 0.75  # Sublinear exponent (< 1 for sublinear)
    sublinear_window_beta: float = 0.4    # Growth rate factor

    # 4. Adaptive Dirichlet noise
    adaptive_noise: bool = True
    noise_entropy_threshold: float = 0.5  # Increase noise when entropy below this
    noise_multiplier_max: float = 2.0     # Max multiplier for noise when entropy low

    # Limits
    max_generations: int = 100

    # Heuristic pretraining (Phase 0 - before self-play)
    use_heuristic_pretraining: bool = True
    pretrain_positions: int = 10000
    pretrain_steps: int = 1000
    pretrain_batch_size: int = 256
    pretrain_lr: float = 0.001

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
            # Gumbel MCTS
            'useGumbelMcts': 'use_gumbel_mcts',
            'gumbelCVisit': 'gumbel_c_visit',
            'gumbelCScale': 'gumbel_c_scale',
            'gumbelMaxActions': 'gumbel_max_actions',
            'batchSize': 'batch_size',
            'learningRate': 'learning_rate',
            'lrDecay': 'lr_decay',
            'trainingStepsPerGen': 'training_steps_per_gen',
            'trainingStepsMin': 'training_steps_min',
            'trainingStepsMax': 'training_steps_max',
            'adaptiveTrainingSteps': 'adaptive_training_steps',
            'replayBufferSize': 'replay_buffer_size',
            'mixedPrecision': 'mixed_precision',
            'maxGradNorm': 'max_grad_norm',
            # Auxiliary losses
            'useAuxiliaryLosses': 'use_auxiliary_losses',
            'auxOwnershipWeight': 'aux_ownership_weight',
            'auxScoreWeight': 'aux_score_weight',
            'auxOpponentPolicyWeight': 'aux_opponent_policy_weight',
            'useHeuristicBootstrap': 'use_heuristic_bootstrap',
            'heuristicBlendInitial': 'heuristic_blend_initial',
            'heuristicBlendDecayGens': 'heuristic_blend_decay_gens',
            'mctsSimulationsInitial': 'mcts_simulations_initial',
            'mctsSimulationsDecayGens': 'mcts_simulations_decay_gens',
            'policyTargetTemperature': 'policy_target_temperature',
            # Playout cap randomization
            'usePlayoutCapRandomization': 'use_playout_cap_randomization',
            'playoutCapFraction': 'playout_cap_fraction',
            'playoutFullSimulations': 'playout_full_simulations',
            'playoutFastSimulations': 'playout_fast_simulations',
            'playoutFastNoNoise': 'playout_fast_no_noise',
            # Policy target pruning
            'usePolicyTargetPruning': 'use_policy_target_pruning',
            'policyPruneThreshold': 'policy_prune_threshold',
            # Warm-start rollouts
            'useWarmStartRollouts': 'use_warm_start_rollouts',
            'warmStartInitialBlend': 'warm_start_initial_blend',
            'warmStartDecayGens': 'warm_start_decay_gens',
            'numResBlocks': 'num_res_blocks',
            'channels': 'channels',
            'evalGames': 'eval_games',
            'evalParallelGames': 'eval_parallel_games',
            'heuristicEvalGames': 'heuristic_eval_games',
            'promotionThreshold': 'promotion_threshold',
            'maxGenerations': 'max_generations',
            # Heuristic pretraining
            'useHeuristicPretraining': 'use_heuristic_pretraining',
            'pretrainPositions': 'pretrain_positions',
            'pretrainSteps': 'pretrain_steps',
            'pretrainBatchSize': 'pretrain_batch_size',
            'pretrainLr': 'pretrain_lr',
            # ELO-based evaluation
            'useEloEvaluation': 'use_elo_evaluation',
            'eloInitial': 'elo_initial',
            'eloKFactor': 'elo_k_factor',
            'eloPromotionThreshold': 'elo_promotion_threshold',
            'eloCompareGenerations': 'elo_compare_generations',
            # Legacy threshold-based (deprecated)
            'dynamicPromotion': 'dynamic_promotion',
            'promotionThresholdMin': 'promotion_threshold_min',
            'promotionDecayPerStuckGen': 'promotion_decay_per_stuck_gen',
            'replayRecentBias': 'replay_recent_bias',
            # Sublinear replay window
            'useSublinearWindow': 'use_sublinear_window',
            'sublinearWindowC': 'sublinear_window_c',
            'sublinearWindowAlpha': 'sublinear_window_alpha',
            'sublinearWindowBeta': 'sublinear_window_beta',
            'adaptiveNoise': 'adaptive_noise',
            'noiseEntropyThreshold': 'noise_entropy_threshold',
            'noiseMultiplierMax': 'noise_multiplier_max',
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

        # Replay buffer with optional sublinear window (KataGo)
        self.replay_buffer = ReplayBuffer(
            max_size=config.replay_buffer_size,
            use_sublinear_window=config.use_sublinear_window,
            sublinear_c=config.sublinear_window_c,
            sublinear_alpha=config.sublinear_window_alpha,
            sublinear_beta=config.sublinear_window_beta,
        )

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
        self._last_heuristics: Optional[Dict[str, float]] = None  # Populated after each self-play phase

        # Heuristic config for policy bootstrapping
        self.heuristic_config = HeuristicConfig() if config.use_heuristic_bootstrap else None

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

        # Dynamic promotion threshold state (legacy - deprecated)
        self.generations_since_promotion = 0
        self.current_promotion_threshold = config.promotion_threshold

        # ELO rating system (research-backed replacement for threshold)
        self.elo_rating = ELORating(
            k_factor=config.elo_k_factor,
            initial_rating=config.elo_initial,
        )
        # Store network checkpoints for ELO comparison
        self.generation_networks: Dict[int, Dict[str, Any]] = {}  # gen -> state_dict

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
        heuristic_win_rate: Optional[float],
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
                'heuristic_win_rate': heuristic_win_rate,
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

    def _get_mcts_simulations(self) -> int:
        """
        Get MCTS simulations for current generation.

        Starts high (mcts_simulations_initial) and decays to base (mcts_simulations)
        over the first mcts_simulations_decay_gens generations.
        """
        if self.current_generation >= self.config.mcts_simulations_decay_gens:
            return self.config.mcts_simulations

        # Linear decay from initial to base
        progress = self.current_generation / self.config.mcts_simulations_decay_gens
        sims = int(
            self.config.mcts_simulations_initial * (1 - progress) +
            self.config.mcts_simulations * progress
        )
        return sims

    def _get_heuristic_blend_factor(self) -> float:
        """
        Get heuristic blend factor for current generation.

        Returns 0.0 if heuristic bootstrapping is disabled.
        """
        if not self.config.use_heuristic_bootstrap:
            return 0.0

        return get_blend_factor_for_generation(
            generation=self.current_generation,
            initial_blend=self.config.heuristic_blend_initial,
            decay_generations=self.config.heuristic_blend_decay_gens,
            min_blend=0.0,
        )

    def _get_warm_start_blend_factor(self) -> float:
        """
        Get warm-start rollout blend factor for current generation.

        From Warm-Start AlphaZero paper: early in training, blend NN value
        with heuristic rollout value to provide meaningful signal before
        the network learns anything useful.

        Returns:
            Blend factor: 0.0 = pure NN value, 1.0 = pure rollout value
        """
        if not self.config.use_warm_start_rollouts:
            return 0.0

        if self.current_generation >= self.config.warm_start_decay_gens:
            return 0.0

        # Linear decay from initial to 0
        progress = self.current_generation / self.config.warm_start_decay_gens
        return self.config.warm_start_initial_blend * (1 - progress)

    def _update_promotion_threshold(self, was_promoted: bool):
        """Update promotion threshold based on stuck generations."""
        if not self.config.dynamic_promotion:
            return

        if was_promoted:
            # Reset to base threshold after promotion
            self.generations_since_promotion = 0
            self.current_promotion_threshold = self.config.promotion_threshold
        else:
            # Lower threshold if stuck
            self.generations_since_promotion += 1
            if self.generations_since_promotion >= 2:  # After 2 stuck generations
                new_threshold = self.current_promotion_threshold - self.config.promotion_decay_per_stuck_gen
                self.current_promotion_threshold = max(new_threshold, self.config.promotion_threshold_min)
                log.info(f"Lowered promotion threshold to {self.current_promotion_threshold:.2%} "
                        f"(stuck for {self.generations_since_promotion} generations)")

    def _get_noise_multiplier(self) -> float:
        """
        Calculate noise multiplier based on current policy entropy.

        When entropy is low (policy is too confident/collapsed), increase noise
        to force more exploration.
        """
        if not self.config.adaptive_noise or not self._last_heuristics:
            return 1.0

        current_entropy = self._last_heuristics.get('policy_entropy', 2.0)
        if current_entropy < self.config.noise_entropy_threshold:
            noise_mult = min(
                self.config.noise_multiplier_max,
                self.config.noise_entropy_threshold / max(current_entropy, 0.1)
            )
            log.info(f"Adaptive noise: entropy={current_entropy:.3f} < threshold={self.config.noise_entropy_threshold}, "
                    f"noise_multiplier={noise_mult:.2f}")
            return noise_mult
        return 1.0

    def _run_heuristic_pretraining(self):
        """Phase 0: Pretrain policy head on heuristic-scored positions."""
        from .pretraining import pretrain_policy_head

        log.info("=" * 60)
        log.info("PHASE 0: Heuristic Pretraining")
        log.info("=" * 60)

        self._report_progress(
            phase='pretraining',
            phase_progress=0.0,
            message='Generating training positions...',
        )

        def progress_cb(step: int, total: int, loss: float):
            self._report_progress(
                phase='pretraining',
                phase_progress=step / total,
                phase_current=step,
                phase_total=total,
                message=f'Step {step}/{total} - loss: {loss:.4f}',
            )

        result = pretrain_policy_head(
            network=self.network,
            device=self.device,
            num_positions=self.config.pretrain_positions,
            num_steps=self.config.pretrain_steps,
            batch_size=self.config.pretrain_batch_size,
            learning_rate=self.config.pretrain_lr,
            progress_callback=progress_cb,
        )

        log.info(f"Pretraining complete! Avg loss: {result['avg_loss']:.4f}")

        self._report_progress(
            phase='pretraining',
            phase_progress=1.0,
            message='Pretraining complete',
        )

        return result

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
            # Phase 0: Heuristic pretraining (before self-play)
            if self.config.use_heuristic_pretraining:
                self._run_heuristic_pretraining()

                if self._stop_requested:
                    log.info("Stop requested during pretraining, exiting")
                    return False

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

                # Phase 3b: Heuristic baseline check
                log.info("Phase 3b: Heuristic Baseline Check")
                heuristic_win_rate = self._run_heuristic_evaluation()
                self._record_generation_complete()

                # Track best win rate (regardless of promotion)
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate

                # Save checkpoint
                model_path = self._save_checkpoint(policy_loss, value_loss, win_rate, heuristic_win_rate, is_best)
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

        # Get dynamic values for cold start fixes
        current_mcts_sims = self._get_mcts_simulations()
        current_blend_factor = self._get_heuristic_blend_factor()

        log.info(f"Starting self-play: {self.config.games_per_generation} games, "
              f"{current_mcts_sims} MCTS sims (base={self.config.mcts_simulations}), "
              f"heuristic_blend={current_blend_factor:.3f}, "
              f"policy_temp={self.config.policy_target_temperature}, "
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

        # Create generation-specific MCTS config with dynamic simulations
        gen_mcts_config = MCTSConfig(
            num_simulations=current_mcts_sims,
            c_puct=self.config.mcts_c_puct,
            temperature_threshold=self.config.temperature_threshold,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
        )

        # Calculate adaptive noise multiplier based on entropy
        noise_multiplier = self._get_noise_multiplier()

        if use_batched:
            # Use new ParallelSelfPlay with batched MCTS
            manager = ParallelSelfPlay(
                network=self.network,
                mcts_config=gen_mcts_config,
                num_parallel_games=self.config.num_parallel_games,
                mcts_batch_size=self.config.mcts_batch_size,
                inference_batch_size=self.config.inference_batch_size,
                device=self.device,
                # Cold start fixes
                heuristic_config=self.heuristic_config if current_blend_factor > 0 else None,
                heuristic_blend_factor=current_blend_factor,
                policy_target_temperature=self.config.policy_target_temperature,
                # Adaptive noise
                noise_multiplier=noise_multiplier,
                # Playout cap randomization (KataGo - 50x efficiency improvement)
                use_playout_cap_randomization=self.config.use_playout_cap_randomization,
                playout_cap_fraction=self.config.playout_cap_fraction,
                playout_full_simulations=self.config.playout_full_simulations,
                playout_fast_simulations=self.config.playout_fast_simulations,
                playout_fast_no_noise=self.config.playout_fast_no_noise,
            )

            # Enable Gumbel MCTS if configured (guaranteed policy improvement)
            if self.config.use_gumbel_mcts:
                manager.configure_gumbel_mcts(
                    c_visit=self.config.gumbel_c_visit,
                    c_scale=self.config.gumbel_c_scale,
                    max_actions=self.config.gumbel_max_actions,
                )

            # Enable policy target pruning if configured (removes exploration artifacts)
            if self.config.use_policy_target_pruning:
                manager.configure_policy_target_pruning(
                    threshold=self.config.policy_prune_threshold,
                )

            # Enable warm-start rollout blending for early generations
            warm_start_blend = self._get_warm_start_blend_factor()
            if warm_start_blend > 0:
                manager.configure_warm_start(warm_start_blend)

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
                mcts_config=gen_mcts_config,
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

            # Policy quality metrics (for diagnosing cold start problem)
            avg_entropy = sum(m.avg_policy_entropy for m in game_metrics_list) / len(game_metrics_list)
            avg_top1 = sum(m.avg_top1_prob for m in game_metrics_list) / len(game_metrics_list)
            avg_top5 = sum(m.avg_top5_prob for m in game_metrics_list) / len(game_metrics_list)

            # Store for training metrics
            self._last_heuristics = {
                'avg_corners': avg_corners,
                'avg_mobility': avg_mobility,
                'center_coverage': avg_center,
                'policy_entropy': avg_entropy,
                'policy_top1': avg_top1,
                'policy_top5': avg_top5,
            }

            # Average final corners per player across all games
            avg_final_corners = [
                sum(m.final_corner_counts[p] for m in game_metrics_list) / len(game_metrics_list)
                for p in range(4)
            ]

            # Log policy quality - this is key for diagnosing learning issues
            # High entropy (~4.0+) means MCTS is generating nearly uniform targets (bad)
            # Low entropy (<2.0) means MCTS is generating focused targets (good)
            log.info(f"[Heuristics] Gen {self.current_generation}: "
                  f"avg_corners={avg_corners:.1f}, avg_mobility={avg_mobility:.1f}, "
                  f"center_coverage={avg_center:.2%}, final_corners={[f'{c:.1f}' for c in avg_final_corners]}")
            log.info(f"[Policy Quality] Gen {self.current_generation}: "
                  f"entropy={avg_entropy:.3f}, top1={avg_top1:.3f}, top5={avg_top5:.3f} "
                  f"(low entropy + high top1/top5 = focused policies = good learning signal)")

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
                    },
                    'policyQuality': {
                        'entropy': round(avg_entropy, 4),
                        'top1Prob': round(avg_top1, 4),
                        'top5Prob': round(avg_top5, 4),
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
            batch = self.replay_buffer.sample(self.config.batch_size, self.config.replay_recent_bias)
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

            # Forward pass and loss computation
            # Use auxiliary losses if enabled (KataGo-style for 50x efficiency)
            use_aux = self.config.use_auxiliary_losses and hasattr(self.network, 'forward_with_auxiliary')

            if self.config.mixed_precision:
                with autocast('cuda'):
                    if use_aux:
                        # Forward with auxiliary outputs
                        outputs = self.network.forward_with_auxiliary(
                            board_tensors, pieces_vectors, legal_moves_list
                        )
                        policy_logits = outputs['policy_logits']
                        values = outputs['value']
                    else:
                        policy_logits, values = self.network(
                            board_tensors, pieces_vectors, legal_moves_list
                        )

                    # Value loss
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

                    # Combined loss
                    loss = policy_loss + value_loss

                    # Add auxiliary losses (KataGo-style)
                    # These provide denser gradients and faster credit assignment
                    if use_aux and 'ownership_logits' in outputs:
                        # Ownership loss: predict who controls each cell
                        # Target: board state ownership (derived from board tensor)
                        # For now, use current board occupancy as proxy target
                        # Channels 0-3 of board_tensor represent player pieces
                        ownership_target = board_tensors[:, :4, :, :]  # (batch, 4, 20, 20)
                        # Normalize to get ownership probabilities
                        ownership_sum = ownership_target.sum(dim=1, keepdim=True).clamp(min=1e-6)
                        ownership_target = ownership_target / ownership_sum
                        ownership_logits = outputs['ownership_logits']
                        ownership_loss = F.cross_entropy(
                            ownership_logits.permute(0, 2, 3, 1).reshape(-1, 4),
                            ownership_target.permute(0, 2, 3, 1).reshape(-1, 4).argmax(dim=1),
                            reduction='mean'
                        )
                        loss = loss + self.config.aux_ownership_weight * ownership_loss

                        # Score loss: predict final score distribution
                        # Target: convert value to score bucket
                        # Value is in [-1, 1], map to score buckets [0, 160]
                        # Score of 0 = bucket 80
                        score_targets = ((target_values.squeeze() + 1) / 2 * 80).long().clamp(0, 160)
                        score_logits = outputs['score_logits']
                        score_loss = F.cross_entropy(score_logits, score_targets, reduction='mean')
                        loss = loss + self.config.aux_score_weight * score_loss

                        # Opponent policy loss: predict opponent's likely response
                        # For training, use same MCTS policy as target (approximation)
                        # In full KataGo, this would use actual opponent responses
                        if 'opponent_policy_logits' in outputs:
                            opp_logits = outputs['opponent_policy_logits'][:, :max_policy_len].clone()
                            opp_logits[policy_mask == 0] = float('-inf')
                            opp_log_probs = F.log_softmax(opp_logits, dim=1)
                            opp_log_probs = torch.nan_to_num(opp_log_probs, nan=0.0)
                            opp_policy_loss = -(target_policies * opp_log_probs).sum(dim=1).mean()
                            loss = loss + self.config.aux_opponent_policy_weight * opp_policy_loss

                self.scaler.scale(loss).backward()
                # Unscale gradients before clipping (required for mixed precision)
                self.scaler.unscale_(self.optimizer)
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if use_aux:
                    outputs = self.network.forward_with_auxiliary(
                        board_tensors, pieces_vectors, legal_moves_list
                    )
                    policy_logits = outputs['policy_logits']
                    values = outputs['value']
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

                # Combined loss
                loss = policy_loss + value_loss

                # Add auxiliary losses if enabled
                if use_aux and 'ownership_logits' in outputs:
                    ownership_target = board_tensors[:, :4, :, :]
                    ownership_sum = ownership_target.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    ownership_target = ownership_target / ownership_sum
                    ownership_logits = outputs['ownership_logits']
                    ownership_loss = F.cross_entropy(
                        ownership_logits.permute(0, 2, 3, 1).reshape(-1, 4),
                        ownership_target.permute(0, 2, 3, 1).reshape(-1, 4).argmax(dim=1),
                        reduction='mean'
                    )
                    loss = loss + self.config.aux_ownership_weight * ownership_loss

                    score_targets = ((target_values.squeeze() + 1) / 2 * 80).long().clamp(0, 160)
                    score_logits = outputs['score_logits']
                    score_loss = F.cross_entropy(score_logits, score_targets, reduction='mean')
                    loss = loss + self.config.aux_score_weight * score_loss

                    if 'opponent_policy_logits' in outputs:
                        opp_logits = outputs['opponent_policy_logits'][:, :max_policy_len].clone()
                        opp_logits[policy_mask == 0] = float('-inf')
                        opp_log_probs = F.log_softmax(opp_logits, dim=1)
                        opp_log_probs = torch.nan_to_num(opp_log_probs, nan=0.0)
                        opp_policy_loss = -(target_policies * opp_log_probs).sum(dim=1).mean()
                        loss = loss + self.config.aux_opponent_policy_weight * opp_policy_loss

                loss.backward()
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

            if self.scheduler and step > 0:
                self.scheduler.step()

            p_item = policy_loss.item()
            v_item = value_loss.item()

            # Check for NaN/Inf losses - indicates catastrophic training failure
            if np.isnan(p_item) or np.isnan(v_item) or np.isinf(p_item) or np.isinf(v_item):
                log.warning(f"NaN/Inf loss detected at step {step + 1}: "
                           f"policy={p_item}, value={v_item}")
                # Don't accumulate bad losses
            else:
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
                # Policy quality metrics
                if 'policy_entropy' in self._last_heuristics:
                    metric['policy_entropy'] = self._last_heuristics['policy_entropy']
                    metric['policy_top1'] = self._last_heuristics['policy_top1']
                    metric['policy_top5'] = self._last_heuristics['policy_top5']
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
        """
        Evaluate new network using either ELO rating or threshold-based system.

        With ELO (default): Plays against recent generations and updates ELO ratings.
        With threshold (legacy): Plays against "best" network with win-rate threshold.
        """
        if self.config.use_elo_evaluation:
            return self._run_elo_evaluation()
        else:
            return self._run_threshold_evaluation()

    def _play_evaluation_games(
        self,
        network_a: BlokusNet,
        network_b: BlokusNet,
        num_games: int,
        label: str = "eval",
    ) -> Tuple[int, int, int]:
        """
        Play evaluation games between two networks.

        Args:
            network_a: First network
            network_b: Second network
            num_games: Number of games to play
            label: Label for logging

        Returns:
            (wins_a, wins_b, draws) - results from network_a's perspective
        """
        network_a.eval()
        network_b.eval()
        parallel_games = self.config.eval_parallel_games

        @dataclass
        class EvalGame:
            game: BlokusGame
            player_a: int  # Which player index network_a plays as
            mcts_a: HybridMCTS
            mcts_b: HybridMCTS

        wins_a = 0
        wins_b = 0
        draws = 0
        games_completed = 0
        game_start_idx = 0

        while games_completed < num_games:
            if self._stop_requested:
                break

            batch_size = min(parallel_games, num_games - games_completed)
            active_games: list[EvalGame] = []

            for i in range(batch_size):
                game_idx = game_start_idx + i
                player_a = game_idx % 2  # Alternate who plays first
                active_games.append(EvalGame(
                    game=BlokusGame(),
                    player_a=player_a,
                    mcts_a=HybridMCTS(network_a, self.mcts_config, self.device),
                    mcts_b=HybridMCTS(network_b, self.mcts_config, self.device),
                ))

            game_start_idx += batch_size

            # Play all games to completion
            while any(not eg.game.is_game_over for eg in active_games):
                if self._stop_requested:
                    break

                for eg in active_games:
                    if eg.game.is_game_over:
                        continue

                    legal_moves = eg.game.get_legal_moves()
                    if not legal_moves:
                        eg.game.pass_turn()
                        continue

                    current = eg.game.current_player
                    mcts = eg.mcts_a if current == eg.player_a else eg.mcts_b
                    move, _, _ = mcts.search(eg.game, add_noise=False, batch_size=self.config.mcts_batch_size)
                    eg.game.make_move(move)

            # Process completed games
            for eg in active_games:
                winner = eg.game.get_winner()
                if winner == eg.player_a:
                    wins_a += 1
                elif winner is not None:
                    wins_b += 1
                else:
                    draws += 1
                games_completed += 1

        return wins_a, wins_b, draws

    def _run_elo_evaluation(self) -> tuple[float, bool]:
        """
        Evaluate using ELO rating system (from ELF OpenGo).

        Plays against multiple recent generations and updates ELO ratings.
        More robust than threshold-based evaluation.
        """
        self.phase_start_time = time.time()
        self.network.eval()

        log.info(f"Starting ELO evaluation for generation {self.current_generation}")

        # Store current network weights for future comparisons
        self.generation_networks[self.current_generation] = {
            k: v.cpu().clone() for k, v in self.network.state_dict().items()
        }

        # Get recent generations to compare against
        compare_gens = self.elo_rating.get_recent_generations(
            self.current_generation,
            self.config.elo_compare_generations
        )

        # If no previous generations, set initial rating and promote
        # Note: Return 0.0 as win_rate since no games were played yet
        # The actual performance will be measured in heuristic evaluation
        if not compare_gens:
            self.elo_rating.ratings[self.current_generation] = self.config.elo_initial
            log.info(f"Generation {self.current_generation}: Initial ELO = {self.config.elo_initial}")
            return 0.0, True  # Always promote first generation, but don't claim fake 100% win rate

        # Play games against each recent generation
        total_wins = 0
        total_losses = 0
        total_draws = 0
        games_per_opponent = max(20, self.config.eval_games // len(compare_gens))

        for opp_gen in compare_gens:
            if self._stop_requested:
                break

            # Load opponent network
            opp_network = create_network(
                num_res_blocks=self.config.num_res_blocks,
                channels=self.config.channels,
                device=self.device,
            )

            if opp_gen in self.generation_networks:
                # Load from cached state dict
                opp_state = {k: v.to(self.device) for k, v in self.generation_networks[opp_gen].items()}
                opp_network.load_state_dict(opp_state)
            else:
                # Try to load from checkpoint file
                checkpoint_path = self.run_dir / f"gen{opp_gen:04d}*.pt"
                import glob
                matches = glob.glob(str(checkpoint_path))
                if matches:
                    checkpoint = torch.load(matches[0], map_location=self.device)
                    opp_network.load_state_dict(checkpoint['model_state_dict'])
                else:
                    log.warning(f"Could not load generation {opp_gen}, skipping")
                    continue

            log.info(f"Evaluating vs generation {opp_gen} ({games_per_opponent} games)")

            wins, losses, draws = self._play_evaluation_games(
                self.network, opp_network, games_per_opponent, f"vs_gen{opp_gen}"
            )

            # Update ELO rating
            self.elo_rating.update_rating(
                self.current_generation, opp_gen, wins, losses, draws
            )

            total_wins += wins
            total_losses += losses
            total_draws += draws

            opp_elo = self.elo_rating.get_rating(opp_gen)
            current_elo = self.elo_rating.get_rating(self.current_generation)
            log.info(f"  vs gen {opp_gen} (ELO {opp_elo:.0f}): {wins}W/{losses}L/{draws}D -> ELO {current_elo:.0f}")

            # Report progress
            progress = (compare_gens.index(opp_gen) + 1) / len(compare_gens)
            self._report_progress(
                phase='evaluation',
                phase_progress=progress,
                phase_current=compare_gens.index(opp_gen) + 1,
                phase_total=len(compare_gens),
                message=f"ELO eval vs gen {opp_gen}: {wins}W/{losses}L",
                extra={
                    'evalType': 'elo',
                    'currentElo': round(current_elo, 1),
                    'opponentGen': opp_gen,
                    'opponentElo': round(opp_elo, 1),
                }
            )

            # Clean up opponent network
            del opp_network
            torch.cuda.empty_cache()

        # Determine promotion based on ELO improvement
        should_promote, elo_improvement = self.elo_rating.should_promote(
            self.current_generation,
            self.config.elo_promotion_threshold
        )

        current_elo = self.elo_rating.get_rating(self.current_generation)
        total_games = total_wins + total_losses + total_draws
        win_rate = total_wins / total_games if total_games > 0 else 0.0

        log.info(f"ELO evaluation complete: Gen {self.current_generation} ELO={current_elo:.0f} "
                f"(+{elo_improvement:.0f}), {total_wins}W/{total_losses}L/{total_draws}D ({win_rate:.1%})")
        log.info(f"{'PROMOTED' if should_promote else 'not promoted'} "
                f"(threshold: +{self.config.elo_promotion_threshold:.0f} ELO)")

        return win_rate, should_promote

    def _run_threshold_evaluation(self) -> tuple[float, bool]:
        """
        Legacy threshold-based evaluation (DEPRECATED).

        Plays against "best" network with win-rate threshold for promotion.
        Kept for backwards compatibility.
        """
        self.phase_start_time = time.time()
        total_games = self.config.eval_games

        log.info(f"Starting threshold evaluation: {total_games} games vs best network")

        wins, losses, draws = self._play_evaluation_games(
            self.network, self.best_network, total_games, "vs_best"
        )

        total_played = wins + losses + draws
        win_rate = wins / total_played if total_played > 0 else 0.0
        is_best = win_rate >= self.current_promotion_threshold

        # Update promotion threshold based on result (legacy behavior)
        self._update_promotion_threshold(is_best)

        threshold_info = ""
        if self.config.dynamic_promotion:
            threshold_info = f" (threshold: {self.current_promotion_threshold:.1%})"
        log.info(f"Threshold evaluation complete: {wins}/{total_played} wins ({win_rate:.2%}) - "
              f"{'PROMOTED' if is_best else 'not promoted'}{threshold_info}")

        return win_rate, is_best

    def _run_heuristic_evaluation(self) -> float:
        """Evaluate new network against pure heuristic player as sanity check.

        This ensures the trained model never regresses below handcrafted-strategy level.
        A win rate below 70% indicates potential issues with training.
        Uses parallel games for faster evaluation.

        Returns:
            Win rate against heuristic player (0.0 to 1.0)
        """
        from .heuristic_player import HeuristicPlayer

        total_games = self.config.heuristic_eval_games
        parallel_games = self.config.eval_parallel_games
        heuristic_threshold = 0.70  # Warn if below this

        self.network.eval()
        self.phase_start_time = time.time()

        log.info(f"Starting heuristic evaluation: {total_games} games, {parallel_games} in parallel")

        # Track active games
        @dataclass
        class HeuristicEvalGame:
            game: BlokusGame
            nn_player: int  # Which player index the NN plays as
            nn_mcts: HybridMCTS
            heuristic: HeuristicPlayer

        new_wins = 0
        games_completed = 0
        game_start_idx = 0

        while games_completed < total_games:
            if self._stop_requested:
                break

            # Start a batch of parallel games
            batch_size = min(parallel_games, total_games - games_completed)
            active_games: list[HeuristicEvalGame] = []

            for i in range(batch_size):
                game_idx = game_start_idx + i
                nn_player = game_idx % 2  # Alternate who plays first
                active_games.append(HeuristicEvalGame(
                    game=BlokusGame(),
                    nn_player=nn_player,
                    nn_mcts=HybridMCTS(self.network, self.mcts_config, self.device),
                    heuristic=HeuristicPlayer(),
                ))

            game_start_idx += batch_size

            # Play all games in this batch to completion
            while any(not eg.game.is_game_over for eg in active_games):
                if self._stop_requested:
                    break

                for eg in active_games:
                    if eg.game.is_game_over:
                        continue

                    legal_moves = eg.game.get_legal_moves()
                    if not legal_moves:
                        eg.game.pass_turn()
                        continue

                    current = eg.game.current_player
                    if current == eg.nn_player:
                        # Neural network plays
                        move, _, _ = eg.nn_mcts.search(eg.game, add_noise=False,
                                                       batch_size=self.config.mcts_batch_size)
                    else:
                        # Heuristic player plays
                        move = eg.heuristic.select_move(eg.game)

                    if move:
                        eg.game.make_move(move)
                    else:
                        eg.game.pass_turn()

            # Process completed games
            for eg in active_games:
                winner = eg.game.get_winner()
                if winner == eg.nn_player:
                    new_wins += 1
                games_completed += 1

            current_wr = new_wins / games_completed if games_completed > 0 else 0.0

            # Report progress
            message = f"Eval vs heuristic: {games_completed}/{total_games} - win rate: {current_wr:.1%}"
            self._report_progress(
                phase='heuristic_eval',
                phase_progress=games_completed / total_games,
                phase_current=games_completed,
                phase_total=total_games,
                message=message,
                extra={
                    'heuristicWins': new_wins,
                    'heuristicGames': games_completed,
                    'heuristicWinRate': round(current_wr, 3),
                    'evalType': 'vs_heuristic',
                    'parallelGames': parallel_games,
                }
            )

            log.info(f"Eval vs heuristic: {games_completed}/{total_games}: {new_wins} wins ({current_wr:.1%})")

        win_rate = new_wins / games_completed if games_completed > 0 else 0.0

        # Warn if below threshold
        if win_rate < heuristic_threshold:
            log.warning(f"Heuristic baseline check FAILED: {win_rate:.1%} < {heuristic_threshold:.0%}")
            log.warning("Model may have regressed below handcrafted strategy level!")
        else:
            log.info(f"Heuristic baseline check PASSED: {win_rate:.1%} >= {heuristic_threshold:.0%}")

        return win_rate

    def _save_checkpoint(
        self,
        policy_loss: float,
        value_loss: float,
        eval_win_rate: float,
        heuristic_win_rate: float,
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
            'heuristic_win_rate': heuristic_win_rate,
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
                'heuristic_win_rate': heuristic_win_rate,
            }, best_path)

        # Report to API
        self._report_checkpoint(
            generation=self.current_generation,
            step=self.total_steps,
            policy_loss=policy_loss,
            value_loss=value_loss,
            eval_win_rate=eval_win_rate,
            eval_games=self.config.eval_games,
            heuristic_win_rate=heuristic_win_rate,
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
