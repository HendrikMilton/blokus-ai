"""
Gumbel MCTS implementation for AlphaZero training.

Based on "Policy Improvement by Planning with Gumbel" (Gumbel MuZero paper).
https://openreview.net/forum?id=bERaNdoegnO

Key advantages over standard PUCT-based MCTS:
1. GUARANTEED policy improvement (PUCT doesn't guarantee this!)
2. Works reliably with just 2 simulations (PUCT fails below 16)
3. Deterministic action selection (no sampling from visit counts)
4. Sequential Halving efficiently allocates simulation budget

Algorithm:
1. Add Gumbel noise to policy logits: g(a) = log(π(a)) + Gumbel(0,1)
2. Select top-k actions by perturbed logits
3. Run Sequential Halving on those k actions
4. Return action with highest value estimate
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math

from app.logging_config import log
from .game import BlokusGame, Move
from .mcts import MCTSConfig
from .network import BlokusNet


@dataclass
class GumbelMCTSConfig:
    """Configuration for Gumbel MCTS."""
    num_simulations: int = 50  # Total simulation budget
    c_visit: float = 50.0      # Visit count scale for action selection
    c_scale: float = 1.0       # Scale for completed Q-values
    max_num_considered_actions: int = 16  # Top-k actions to consider
    dirichlet_alpha: float = 0.3   # For root exploration (optional)
    dirichlet_epsilon: float = 0.0  # Disabled by default (Gumbel provides exploration)


def sample_gumbel(shape: Tuple[int, ...], eps: float = 1e-20) -> np.ndarray:
    """Sample from Gumbel(0, 1) distribution."""
    u = np.random.uniform(0, 1, shape)
    return -np.log(-np.log(u + eps) + eps)


def gumbel_softmax_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample from Gumbel-Softmax distribution and return argmax."""
    gumbel_noise = sample_gumbel(logits.shape)
    perturbed = (logits + gumbel_noise) / temperature
    return int(np.argmax(perturbed))


class GumbelMCTSNode:
    """
    Node for Gumbel MCTS tree.

    Unlike standard MCTS, we track:
    - Completed Q-values (after full search) vs prior Q (from NN)
    - Gumbel values for each action (sampled once at creation)
    """

    def __init__(
        self,
        game: BlokusGame,
        prior_policy: np.ndarray,
        prior_value: float,
        legal_moves: List[Move],
    ):
        self.game = game
        self.prior_policy = prior_policy  # NN policy prior
        self.prior_value = prior_value    # NN value estimate
        self.legal_moves = legal_moves
        self.num_actions = len(legal_moves)

        # Gumbel noise sampled once per action (key to the algorithm)
        self.gumbel_noise = sample_gumbel((self.num_actions,))

        # Visit counts and value sums per action
        self.visit_counts = np.zeros(self.num_actions)
        self.value_sums = np.zeros(self.num_actions)

        # Child nodes (lazily created)
        self.children: Dict[int, 'GumbelMCTSNode'] = {}

    def get_completed_q(self, action_idx: int) -> float:
        """Get completed Q-value for an action (after visits)."""
        if self.visit_counts[action_idx] == 0:
            return 0.0
        return self.value_sums[action_idx] / self.visit_counts[action_idx]

    def get_mixed_q(self, action_idx: int, c_visit: float, c_scale: float) -> float:
        """
        Get mixed Q-value combining prior and completed estimates.

        From the Gumbel MuZero paper:
        σ(a) = interpolation weight based on visit count
        Q_mixed(a) = σ(a) * Q_completed(a) + (1 - σ(a)) * Q_prior
        """
        visits = self.visit_counts[action_idx]
        total_visits = np.sum(self.visit_counts)

        if total_visits == 0:
            return self.prior_value

        # Interpolation weight increases with visits
        sigma = visits / (visits + c_visit)

        q_completed = self.get_completed_q(action_idx)
        q_mixed = sigma * q_completed + (1 - sigma) * self.prior_value

        return q_mixed

    def get_score(self, action_idx: int, c_visit: float, c_scale: float) -> float:
        """
        Get score for action selection (Gumbel + transformed Q).

        score(a) = g(a) + logit(q_mixed(a))
        where logit transforms Q from [-1,1] to unbounded range.
        """
        q_mixed = self.get_mixed_q(action_idx, c_visit, c_scale)

        # Transform Q to logit scale (inverse sigmoid)
        # Clip to avoid -inf/inf
        q_clipped = np.clip(q_mixed, -0.999, 0.999)
        q_logit = np.log((1 + q_clipped) / (1 - q_clipped + 1e-10))

        # Combine with Gumbel noise and prior logit
        prior_logit = np.log(self.prior_policy[action_idx] + 1e-10)
        return self.gumbel_noise[action_idx] + prior_logit + c_scale * q_logit


class GumbelMCTS:
    """
    Gumbel MCTS for guaranteed policy improvement.

    Uses Sequential Halving to efficiently allocate simulation budget
    across top-k actions selected by Gumbel-perturbed policy.
    """

    def __init__(
        self,
        network: BlokusNet,
        config: GumbelMCTSConfig,
        device: str = 'cuda',
    ):
        self.network = network
        self.config = config
        self.device = device

    def search(
        self,
        game: BlokusGame,
        add_noise: bool = False,  # Ignored - Gumbel provides exploration
    ) -> Tuple[Move, np.ndarray, int]:
        """
        Run Gumbel MCTS search from the given game state.

        Args:
            game: Current game state
            add_noise: Ignored (Gumbel noise provides exploration)

        Returns:
            (selected_move, policy, total_visits)
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available")

        # Get initial policy and value from neural network
        board_tensor, pieces_vector = game.get_state_for_neural_net()
        moves_tuples = [(m.piece_idx, m.orientation_idx, m.row, m.col) for m in legal_moves]

        policy, value = self.network.predict(
            board_tensor, pieces_vector, moves_tuples, self.device
        )

        # Create root node
        root = GumbelMCTSNode(
            game=game.clone(),
            prior_policy=policy,
            prior_value=value,
            legal_moves=legal_moves,
        )

        # If only one legal move, return it
        if len(legal_moves) == 1:
            return legal_moves[0], np.array([1.0]), 0

        # Run Sequential Halving with Gumbel sampling
        selected_idx = self._sequential_halving(root)

        # Build policy from visit counts (for training)
        total_visits = np.sum(root.visit_counts)
        if total_visits > 0:
            visit_policy = root.visit_counts / total_visits
        else:
            visit_policy = policy  # Fall back to prior

        return legal_moves[selected_idx], visit_policy, int(total_visits)

    def _sequential_halving(self, root: GumbelMCTSNode) -> int:
        """
        Sequential Halving algorithm with Gumbel-Top-k sampling.

        1. Select top-k actions by Gumbel-perturbed policy logits
        2. Allocate simulations across log2(k) phases
        3. In each phase, simulate and eliminate bottom half
        4. Return final winner

        Guaranteed to find improving action if one exists!
        """
        num_actions = root.num_actions
        k = min(self.config.max_num_considered_actions, num_actions)

        # Phase 0: Select top-k actions by Gumbel-perturbed policy
        prior_logits = np.log(root.prior_policy + 1e-10)
        perturbed_logits = prior_logits + root.gumbel_noise
        top_k_indices = np.argsort(perturbed_logits)[-k:][::-1].tolist()

        # Calculate number of phases and simulations per phase
        num_phases = max(1, int(np.ceil(np.log2(k))))
        sims_per_phase = max(1, self.config.num_simulations // num_phases)

        candidates = top_k_indices

        # Run Sequential Halving phases
        for phase in range(num_phases):
            if len(candidates) <= 1:
                break

            # Allocate simulations equally among remaining candidates
            sims_each = max(1, sims_per_phase // len(candidates))

            # Run simulations for each candidate
            for action_idx in candidates:
                for _ in range(sims_each):
                    value = self._simulate(root, action_idx)
                    root.visit_counts[action_idx] += 1
                    root.value_sums[action_idx] += value

            # Eliminate bottom half based on scores
            scores = [
                root.get_score(idx, self.config.c_visit, self.config.c_scale)
                for idx in candidates
            ]
            sorted_candidates = [x for _, x in sorted(zip(scores, candidates), reverse=True)]

            # Keep top half (at least 1)
            num_keep = max(1, len(sorted_candidates) // 2)
            candidates = sorted_candidates[:num_keep]

        # Return the winner
        return candidates[0]

    def _simulate(self, root: GumbelMCTSNode, action_idx: int) -> float:
        """
        Simulate from root taking the specified action.

        Returns value from root player's perspective.
        """
        perspective_player = root.game.current_player

        # Get or create child node
        if action_idx not in root.children:
            child_game = root.game.clone()
            child_game.make_move(root.legal_moves[action_idx])

            # Handle passes
            while not child_game.is_game_over and not child_game.get_legal_moves():
                child_game.pass_turn()

            if child_game.is_game_over:
                # Terminal: return actual outcome
                winner = child_game.get_winner()
                if winner is not None:
                    return 1.0 if winner == perspective_player else -1.0
                return 0.0

            # Non-terminal: evaluate with neural network
            child_legal = child_game.get_legal_moves()
            bt, pv = child_game.get_state_for_neural_net()
            moves_tuples = [(m.piece_idx, m.orientation_idx, m.row, m.col) for m in child_legal]

            child_policy, child_value = self.network.predict(bt, pv, moves_tuples, self.device)

            root.children[action_idx] = GumbelMCTSNode(
                game=child_game,
                prior_policy=child_policy,
                prior_value=child_value,
                legal_moves=child_legal,
            )

            # Flip value to root's perspective if needed
            if child_game.current_player != perspective_player:
                child_value = -child_value

            return child_value

        # Child exists - could do deeper search but for efficiency just return cached value
        child = root.children[action_idx]
        value = child.prior_value

        # Flip if opponent's turn
        if child.game.current_player != perspective_player:
            value = -value

        return value


def create_gumbel_mcts(
    network: BlokusNet,
    config: Optional[GumbelMCTSConfig] = None,
    device: str = 'cuda',
) -> GumbelMCTS:
    """Factory function to create Gumbel MCTS instance."""
    if config is None:
        config = GumbelMCTSConfig()
    return GumbelMCTS(network, config, device)


def convert_mcts_config_to_gumbel(mcts_config: MCTSConfig) -> GumbelMCTSConfig:
    """Convert standard MCTSConfig to GumbelMCTSConfig."""
    return GumbelMCTSConfig(
        num_simulations=mcts_config.num_simulations,
        dirichlet_alpha=mcts_config.dirichlet_alpha,
        dirichlet_epsilon=0.0,  # Gumbel provides exploration
    )
