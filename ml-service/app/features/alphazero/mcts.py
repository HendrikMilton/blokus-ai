"""
Monte Carlo Tree Search implementation for AlphaZero-style Blokus AI.

Uses PUCT formula for node selection and neural network for policy/value evaluation.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .game import BlokusGame, Move


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 400
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 20  # Use temperature=0.1 after this many moves


class MCTSNode:
    """
    A node in the MCTS tree with lazy child expansion.

    Key optimization: Children are NOT fully created during expand().
    Instead, we store policy priors and only create the actual child
    game state when that child is selected for traversal.

    This avoids cloning game states for moves that are never explored.
    """

    def __init__(self, game: BlokusGame, parent: Optional['MCTSNode'] = None,
                 move: Optional[Move] = None, prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.move = move  # Move that led to this state
        self.prior = prior  # P(s, a) from neural network

        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Move, 'MCTSNode'] = {}

        # Legal moves at this state (cached)
        self._legal_moves: Optional[List[Move]] = None
        self._is_expanded = False

        # Lazy expansion: store priors without creating child game states
        self._child_priors: Optional[Dict[Move, float]] = None

    @property
    def legal_moves(self) -> List[Move]:
        """Get legal moves (cached)."""
        if self._legal_moves is None:
            self._legal_moves = self.game.get_legal_moves()
        return self._legal_moves

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.game.is_game_over

    @property
    def q_value(self) -> float:
        """Get average value (Q) for this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """
        Calculate UCB score for node selection.

        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))
        """
        if self.parent is None:
            return 0.0

        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration

    def _get_unexpanded_ucb(self, move: Move, c_puct: float) -> float:
        """Calculate UCB score for a move that hasn't been expanded yet."""
        prior = self._child_priors.get(move, 0.0)
        # Unexpanded nodes have visit_count=0, value_sum=0, so Q=0
        exploration = c_puct * prior * math.sqrt(self.visit_count) / 1
        return exploration

    def select_child(self, c_puct: float) -> 'MCTSNode':
        """
        Select best child according to UCB score.

        With lazy expansion, we consider both:
        1. Already-expanded children (have game states)
        2. Unexpanded moves (only have priors)

        If an unexpanded move has the best UCB, we expand it on-demand.
        """
        best_score = float('-inf')
        best_child = None
        best_unexpanded_move = None

        # Check expanded children
        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
                best_unexpanded_move = None

        # Check unexpanded moves (if any)
        if self._child_priors:
            for move, prior in self._child_priors.items():
                if move not in self.children:
                    score = self._get_unexpanded_ucb(move, c_puct)
                    if score > best_score:
                        best_score = score
                        best_child = None
                        best_unexpanded_move = move

        # If best is unexpanded, expand it now (lazy expansion)
        if best_unexpanded_move is not None:
            best_child = self._expand_child(best_unexpanded_move)

        assert best_child is not None, "No children to select from"
        return best_child

    def _expand_child(self, move: Move) -> 'MCTSNode':
        """Lazily expand a single child when it's actually selected."""
        prior = self._child_priors.get(move, 0.0)

        # Clone game and apply move
        child_game = self.game.clone()
        child_game.make_move(move)

        # Auto-pass if needed
        while not child_game.is_game_over and not child_game.get_legal_moves():
            child_game.pass_turn()

        child = MCTSNode(
            game=child_game,
            parent=self,
            move=move,
            prior=prior,
        )
        self.children[move] = child
        return child

    def expand(self, policy: np.ndarray, legal_moves: List[Move]):
        """
        Expand node by storing policy priors (lazy expansion).

        Unlike eager expansion, this does NOT create child game states.
        Children are created on-demand in select_child() when selected.

        This is a massive speedup: O(1) instead of O(num_legal_moves) game clones.
        """
        assert len(policy) == len(legal_moves)

        # Store priors for lazy expansion
        self._child_priors = {move: policy[i] for i, move in enumerate(legal_moves)}
        self._is_expanded = True

    def expand_eager(self, policy: np.ndarray, legal_moves: List[Move]):
        """
        Eager expansion (legacy) - creates all children immediately.
        Use expand() for lazy expansion which is much faster.
        """
        assert len(policy) == len(legal_moves)

        for i, move in enumerate(legal_moves):
            if move not in self.children:
                child_game = self.game.clone()
                child_game.make_move(move)

                while not child_game.is_game_over and not child_game.get_legal_moves():
                    child_game.pass_turn()

                child = MCTSNode(
                    game=child_game,
                    parent=self,
                    move=move,
                    prior=policy[i],
                )
                self.children[move] = child

        self._child_priors = {move: policy[i] for i, move in enumerate(legal_moves)}
        self._is_expanded = True

    def backpropagate(self, value: float, perspective_player: int):
        """
        Backpropagate value up the tree.

        Value is from the perspective of perspective_player.
        Nodes flip the sign based on whose turn it is.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            # Flip value if it's the opponent's perspective
            if node.parent is not None:
                parent_player = node.parent.game.current_player
                if parent_player == perspective_player:
                    node.value_sum += value
                else:
                    node.value_sum -= value
            node = node.parent


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    """

    def __init__(self, network, config: MCTSConfig, device: str = 'cuda'):
        self.network = network
        self.config = config
        self.device = device

    def _get_policy_value(self, game: BlokusGame, legal_moves: List[Move]) -> Tuple[np.ndarray, float]:
        """Get policy and value from neural network."""
        board_tensor, pieces_vector = game.get_state_for_neural_net()
        policy, value = self.network.predict(board_tensor, pieces_vector, legal_moves, self.device)
        return policy, value

    def search(self, root_game: BlokusGame, add_noise: bool = True) -> MCTSNode:
        """
        Run MCTS from the given game state.

        Args:
            root_game: Current game state
            add_noise: Whether to add Dirichlet noise at root (for exploration during training)

        Returns:
            Root node after search
        """
        root = MCTSNode(root_game.clone())

        # Check terminal
        if root.is_terminal:
            return root

        # Get initial policy and expand root
        legal_moves = root.legal_moves
        if not legal_moves:
            return root

        policy, value = self._get_policy_value(root.game, legal_moves)

        # Add Dirichlet noise at root for exploration
        if add_noise and len(policy) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(policy))
            policy = (1 - self.config.dirichlet_epsilon) * policy + self.config.dirichlet_epsilon * noise

        root.expand(policy, legal_moves)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            perspective_player = root.game.current_player

            # Selection: traverse tree to leaf
            while node._is_expanded and not node.is_terminal:
                node = node.select_child(self.config.c_puct)

            # Evaluation and expansion
            if node.is_terminal:
                # Terminal node: use actual game result
                scores = node.game.get_scores()
                winner = node.game.get_winner()
                if winner is not None:
                    value = 1.0 if winner == perspective_player else -1.0
                else:
                    value = 0.0  # Tie
            else:
                # Leaf node: expand and evaluate
                legal_moves = node.legal_moves
                if legal_moves:
                    policy, value = self._get_policy_value(node.game, legal_moves)
                    node.expand(policy, legal_moves)
                else:
                    # No legal moves - pass turn
                    value = 0.0

            # Backpropagation
            node.backpropagate(value, perspective_player)

        return root

    def get_action_probs(self, root: MCTSNode, temperature: float = 1.0) -> Tuple[List[Move], np.ndarray]:
        """
        Get action probabilities from MCTS visit counts.

        Args:
            root: Root node after search
            temperature: Temperature for action selection
                        1.0 = proportional to visit counts
                        0 (or very small) = argmax

        Returns:
            moves: List of legal moves
            probs: Action probabilities
        """
        moves = []
        visits = []

        for move, child in root.children.items():
            moves.append(move)
            visits.append(child.visit_count)

        if not moves:
            return [], np.array([])

        visits = np.array(visits, dtype=np.float32)

        if temperature < 0.01:
            # Greedy selection
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Temperature-weighted
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()

        return moves, probs

    def select_move(self, game: BlokusGame, add_noise: bool = True,
                    temperature: Optional[float] = None) -> Tuple[Move, np.ndarray, MCTSNode]:
        """
        Select a move using MCTS.

        Args:
            game: Current game state
            add_noise: Whether to add exploration noise
            temperature: Override temperature (None = use config based on turn)

        Returns:
            selected_move: The chosen move
            policy: Action probabilities (for training)
            root: Root node (for analysis)
        """
        root = self.search(game, add_noise=add_noise)

        # Determine temperature
        if temperature is None:
            if game.turn_number <= self.config.temperature_threshold:
                temperature = self.config.temperature
            else:
                temperature = 0.1

        moves, probs = self.get_action_probs(root, temperature)

        if not moves:
            raise ValueError("No legal moves available")

        # Sample move according to probabilities
        idx = np.random.choice(len(moves), p=probs)
        selected_move = moves[idx]

        # Create full policy vector for training (over all legal moves, not just children)
        full_policy = np.zeros(len(moves))
        for i, m in enumerate(moves):
            full_policy[i] = probs[i]

        return selected_move, full_policy, root


class BatchedMCTS:
    """
    Monte Carlo Tree Search with batched neural network evaluation.

    Instead of evaluating one leaf at a time, this implementation collects
    multiple leaf nodes during the selection phase and evaluates them in
    a single GPU batch. This dramatically improves GPU utilization.

    Uses "virtual loss" to encourage exploration of different paths when
    selecting multiple leaves in parallel.
    """

    def __init__(self, network, config: MCTSConfig, device: str = 'cuda'):
        self.network = network
        self.config = config
        self.device = device
        self.virtual_loss = 3.0  # Pessimistic value during selection

    def _select_leaf(self, root: MCTSNode, c_puct: float) -> Tuple[MCTSNode, List[MCTSNode]]:
        """
        Select a leaf node by traversing down the tree using UCB scores.

        Returns:
            leaf: The selected leaf node
            path: List of nodes from root to leaf (inclusive)
        """
        node = root
        path = [node]

        while node._is_expanded and not node.is_terminal:
            node = node.select_child(c_puct)
            path.append(node)

        return node, path

    def _apply_virtual_loss(self, path: List[MCTSNode]):
        """
        Apply virtual loss to path to discourage other simulations from
        selecting the same path. This is key for parallel leaf collection.
        """
        for node in path[1:]:  # Skip root
            node.visit_count += 1
            node.value_sum -= self.virtual_loss

    def _remove_virtual_loss(self, path: List[MCTSNode]):
        """Remove virtual loss after real evaluation."""
        for node in path[1:]:  # Skip root
            node.visit_count -= 1
            node.value_sum += self.virtual_loss

    def _get_batch_policy_value(
        self,
        games: List[BlokusGame],
        legal_moves_list: List[List[Move]],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Get policies and values for a batch of positions."""
        board_tensors = []
        pieces_vectors = []

        for game in games:
            bt, pv = game.get_state_for_neural_net()
            board_tensors.append(bt)
            pieces_vectors.append(pv)

        return self.network.predict_batch(
            board_tensors, pieces_vectors, legal_moves_list, self.device
        )

    def search(self, root_game: BlokusGame, add_noise: bool = True,
               batch_size: int = 16) -> MCTSNode:
        """
        Run batched MCTS from the given game state.

        Args:
            root_game: Current game state
            add_noise: Whether to add Dirichlet noise at root
            batch_size: Number of leaves to evaluate per batch

        Returns:
            Root node after search
        """
        root = MCTSNode(root_game.clone())

        if root.is_terminal:
            return root

        legal_moves = root.legal_moves
        if not legal_moves:
            return root

        # Get initial policy and expand root
        board_tensor, pieces_vector = root.game.get_state_for_neural_net()
        policy, value = self.network.predict(
            board_tensor, pieces_vector, legal_moves, self.device
        )

        # Add Dirichlet noise at root for exploration
        if add_noise and len(policy) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(policy))
            policy = (1 - self.config.dirichlet_epsilon) * policy + self.config.dirichlet_epsilon * noise

        root.expand(policy, legal_moves)
        perspective_player = root.game.current_player

        # Run simulations in batches
        simulations_done = 0
        while simulations_done < self.config.num_simulations:
            current_batch_size = min(batch_size, self.config.num_simulations - simulations_done)

            # Phase 1: Selection - collect leaves with virtual loss
            leaves = []
            paths = []
            leaf_legal_moves = []

            for _ in range(current_batch_size):
                leaf, path = self._select_leaf(root, self.config.c_puct)
                leaves.append(leaf)
                paths.append(path)

                if not leaf.is_terminal:
                    self._apply_virtual_loss(path)
                    leaf_legal_moves.append(leaf.legal_moves)
                else:
                    leaf_legal_moves.append([])

            # Phase 2: Batch evaluation
            # Separate terminal and non-terminal leaves
            to_evaluate_indices = []
            to_evaluate_games = []
            to_evaluate_moves = []

            for i, (leaf, moves) in enumerate(zip(leaves, leaf_legal_moves)):
                if not leaf.is_terminal and moves:
                    to_evaluate_indices.append(i)
                    to_evaluate_games.append(leaf.game)
                    to_evaluate_moves.append(moves)

            # Batch evaluate non-terminal leaves
            if to_evaluate_games:
                policies, values = self._get_batch_policy_value(
                    to_evaluate_games, to_evaluate_moves
                )
            else:
                policies, values = [], []

            # Phase 3: Expansion and Backpropagation
            eval_idx = 0
            for i, (leaf, path, moves) in enumerate(zip(leaves, paths, leaf_legal_moves)):
                if leaf.is_terminal:
                    # Terminal: use actual game result
                    winner = leaf.game.get_winner()
                    if winner is not None:
                        value = 1.0 if winner == perspective_player else -1.0
                    else:
                        value = 0.0
                elif moves:
                    # Non-terminal with moves: expand with NN policy
                    policy = policies[eval_idx]
                    value = values[eval_idx]
                    leaf.expand(policy, moves)
                    self._remove_virtual_loss(path)
                    eval_idx += 1
                else:
                    # No legal moves
                    value = 0.0
                    self._remove_virtual_loss(path)

                # Backpropagation
                leaf.backpropagate(value, perspective_player)

            simulations_done += current_batch_size

        return root

    def get_action_probs(self, root: MCTSNode, temperature: float = 1.0) -> Tuple[List[Move], np.ndarray]:
        """Get action probabilities from MCTS visit counts."""
        moves = []
        visits = []

        for move, child in root.children.items():
            moves.append(move)
            visits.append(child.visit_count)

        if not moves:
            return [], np.array([])

        visits = np.array(visits, dtype=np.float32)

        if temperature < 0.01:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()

        return moves, probs

    def select_move(self, game: BlokusGame, add_noise: bool = True,
                    temperature: Optional[float] = None,
                    batch_size: int = 16) -> Tuple[Move, np.ndarray, MCTSNode]:
        """
        Select a move using batched MCTS.

        Args:
            game: Current game state
            add_noise: Whether to add exploration noise
            temperature: Override temperature (None = use config based on turn)
            batch_size: Number of leaves to evaluate per batch

        Returns:
            selected_move: The chosen move
            policy: Action probabilities (for training)
            root: Root node (for analysis)
        """
        root = self.search(game, add_noise=add_noise, batch_size=batch_size)

        if temperature is None:
            if game.turn_number <= self.config.temperature_threshold:
                temperature = self.config.temperature
            else:
                temperature = 0.1

        moves, probs = self.get_action_probs(root, temperature)

        if not moves:
            raise ValueError("No legal moves available")

        idx = np.random.choice(len(moves), p=probs)
        selected_move = moves[idx]

        full_policy = np.zeros(len(moves))
        for i, m in enumerate(moves):
            full_policy[i] = probs[i]

        return selected_move, full_policy, root


def create_mcts(network, config: Optional[MCTSConfig] = None, device: str = 'cuda') -> MCTS:
    """Create MCTS instance."""
    if config is None:
        config = MCTSConfig()
    return MCTS(network, config, device)


def create_batched_mcts(network, config: Optional[MCTSConfig] = None, device: str = 'cuda') -> BatchedMCTS:
    """Create BatchedMCTS instance for efficient GPU inference."""
    if config is None:
        config = MCTSConfig()
    return BatchedMCTS(network, config, device)
