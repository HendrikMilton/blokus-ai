"""
GPU-accelerated MCTS implementation for AlphaZero Blokus.

Key optimization: Instead of cloning game states in Python one at a time,
we maintain a pool of GPU game states and use batched tensor operations
for tree traversal.

Architecture:
1. Game states are stored in a BlokusGameBatchGPU tensor pool
2. MCTS nodes reference indices into this pool instead of holding game objects
3. Clone and move operations are batched across all MCTS simulations
4. Neural network evaluations are batched as before

This achieves high GPU utilization by replacing O(N) Python game.clone()
calls with O(1) GPU tensor indexing.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from .game import BlokusGame, Move, NUM_PLAYERS
from .game_gpu import BlokusGameBatchGPU, PieceDataGPU
from .mcts import MCTSConfig, MCTSNode


@dataclass
class GPUMCTSStats:
    """Statistics for GPU MCTS performance monitoring."""
    total_simulations: int = 0
    total_clones: int = 0
    total_moves_applied: int = 0
    clone_time_ms: float = 0.0
    move_time_ms: float = 0.0
    nn_time_ms: float = 0.0
    selection_time_ms: float = 0.0
    backprop_time_ms: float = 0.0


class MCTSNodeGPU:
    """
    MCTS node that references a game state by index in a GPU batch.

    Instead of storing a BlokusGame object, stores an index into
    the GPU game state pool. This enables batched cloning.
    """

    def __init__(
        self,
        game_idx: int,
        parent: Optional['MCTSNodeGPU'] = None,
        move: Optional[Move] = None,
        prior: float = 0.0,
    ):
        self.game_idx = game_idx  # Index into GPU game batch
        self.parent = parent
        self.move = move
        self.prior = prior

        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Move, 'MCTSNodeGPU'] = {}

        # Lazy expansion
        self._child_priors: Optional[Dict[Move, float]] = None
        self._is_expanded = False
        self._legal_moves: Optional[List[Move]] = None
        self._is_terminal: Optional[bool] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration

    def get_unexpanded_ucb(self, move: Move, c_puct: float) -> float:
        prior = self._child_priors.get(move, 0.0)
        exploration = c_puct * prior * np.sqrt(self.visit_count) / 1
        return exploration

    def select_best_unexpanded(self, c_puct: float) -> Optional[Move]:
        """Select the unexpanded move with highest UCB score."""
        best_score = float('-inf')
        best_move = None

        # Check expanded children
        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_move = None  # Expanded child wins

        # Check unexpanded moves
        if self._child_priors:
            for move, prior in self._child_priors.items():
                if move not in self.children:
                    score = self.get_unexpanded_ucb(move, c_puct)
                    if score > best_score:
                        best_score = score
                        best_move = move

        return best_move


class GPUGamePool:
    """
    Pool of GPU game states for MCTS.

    Manages allocation and deallocation of game state slots in a GPU batch.
    Supports efficient batched cloning and move application.
    """

    def __init__(self, initial_size: int = 1024, device: str = 'cuda'):
        self.device = device
        self.capacity = initial_size
        self.batch = BlokusGameBatchGPU(initial_size, device)

        # Track which slots are in use
        self.allocated = torch.zeros(initial_size, dtype=torch.bool, device='cpu')
        self.num_allocated = 0

        # Cache for legal moves (computed on CPU)
        self._legal_moves_cache: Dict[int, List[Move]] = {}
        self._is_terminal_cache: Dict[int, bool] = {}

    def allocate(self) -> int:
        """Allocate a new game state slot. Returns the index."""
        if self.num_allocated >= self.capacity:
            self._grow()

        # Find first free slot
        free_idx = (~self.allocated).nonzero(as_tuple=True)[0][0].item()
        self.allocated[free_idx] = True
        self.num_allocated += 1
        return free_idx

    def free(self, idx: int):
        """Free a game state slot."""
        self.allocated[idx] = False
        self.num_allocated -= 1
        self._legal_moves_cache.pop(idx, None)
        self._is_terminal_cache.pop(idx, None)

    def _grow(self):
        """Double the pool capacity."""
        new_capacity = self.capacity * 2
        new_batch = BlokusGameBatchGPU(new_capacity, self.device)

        # Copy existing data
        new_batch.boards[:self.capacity] = self.batch.boards
        new_batch.remaining_pieces[:self.capacity] = self.batch.remaining_pieces
        new_batch.current_player[:self.capacity] = self.batch.current_player
        new_batch.has_played[:self.capacity] = self.batch.has_played
        new_batch.is_eliminated[:self.capacity] = self.batch.is_eliminated
        new_batch.valid_corners[:self.capacity] = self.batch.valid_corners
        new_batch.turn_number[:self.capacity] = self.batch.turn_number
        new_batch.consecutive_passes[:self.capacity] = self.batch.consecutive_passes
        new_batch.is_game_over[:self.capacity] = self.batch.is_game_over
        new_batch.scores[:self.capacity] = self.batch.scores

        # Expand allocated tracking
        new_allocated = torch.zeros(new_capacity, dtype=torch.bool, device='cpu')
        new_allocated[:self.capacity] = self.allocated

        self.batch = new_batch
        self.allocated = new_allocated
        self.capacity = new_capacity

    def clone_to_new(self, src_idx: int) -> int:
        """Clone a game state to a new slot."""
        dst_idx = self.allocate()
        indices = torch.tensor([src_idx], device=self.device)

        # Copy all state tensors
        self.batch.boards[dst_idx] = self.batch.boards[src_idx].clone()
        self.batch.remaining_pieces[dst_idx] = self.batch.remaining_pieces[src_idx].clone()
        self.batch.current_player[dst_idx] = self.batch.current_player[src_idx]
        self.batch.has_played[dst_idx] = self.batch.has_played[src_idx].clone()
        self.batch.is_eliminated[dst_idx] = self.batch.is_eliminated[src_idx].clone()
        self.batch.valid_corners[dst_idx] = self.batch.valid_corners[src_idx].clone()
        self.batch.turn_number[dst_idx] = self.batch.turn_number[src_idx]
        self.batch.consecutive_passes[dst_idx] = self.batch.consecutive_passes[src_idx]
        self.batch.is_game_over[dst_idx] = self.batch.is_game_over[src_idx]
        self.batch.scores[dst_idx] = self.batch.scores[src_idx].clone()

        return dst_idx

    def batch_clone(self, src_indices: List[int]) -> List[int]:
        """Clone multiple game states at once."""
        n = len(src_indices)
        dst_indices = []

        for _ in range(n):
            dst_indices.append(self.allocate())

        src_tensor = torch.tensor(src_indices, device=self.device)
        dst_tensor = torch.tensor(dst_indices, device=self.device)

        # Batch copy
        self.batch.boards[dst_tensor] = self.batch.boards[src_tensor].clone()
        self.batch.remaining_pieces[dst_tensor] = self.batch.remaining_pieces[src_tensor].clone()
        self.batch.current_player[dst_tensor] = self.batch.current_player[src_tensor].clone()
        self.batch.has_played[dst_tensor] = self.batch.has_played[src_tensor].clone()
        self.batch.is_eliminated[dst_tensor] = self.batch.is_eliminated[src_tensor].clone()
        self.batch.valid_corners[dst_tensor] = self.batch.valid_corners[src_tensor].clone()
        self.batch.turn_number[dst_tensor] = self.batch.turn_number[src_tensor].clone()
        self.batch.consecutive_passes[dst_tensor] = self.batch.consecutive_passes[src_tensor].clone()
        self.batch.is_game_over[dst_tensor] = self.batch.is_game_over[src_tensor].clone()
        self.batch.scores[dst_tensor] = self.batch.scores[src_tensor].clone()

        return dst_indices

    def apply_move(self, idx: int, move: Move):
        """Apply a move to a game state."""
        self.batch.apply_moves(
            batch_indices=torch.tensor([idx], device=self.device),
            piece_indices=torch.tensor([move.piece_idx], device=self.device),
            orient_indices=torch.tensor([move.orientation_idx], device=self.device),
            rows=torch.tensor([move.row], device=self.device),
            cols=torch.tensor([move.col], device=self.device),
        )
        # Invalidate cache
        self._legal_moves_cache.pop(idx, None)
        self._is_terminal_cache.pop(idx, None)

    def batch_apply_moves(self, indices: List[int], moves: List[Move]):
        """Apply moves to multiple game states at once."""
        if not indices:
            return

        self.batch.apply_moves(
            batch_indices=torch.tensor(indices, device=self.device),
            piece_indices=torch.tensor([m.piece_idx for m in moves], device=self.device),
            orient_indices=torch.tensor([m.orientation_idx for m in moves], device=self.device),
            rows=torch.tensor([m.row for m in moves], device=self.device),
            cols=torch.tensor([m.col for m in moves], device=self.device),
        )

        # Invalidate caches
        for idx in indices:
            self._legal_moves_cache.pop(idx, None)
            self._is_terminal_cache.pop(idx, None)

    def pass_turn(self, idx: int):
        """Pass turn for a game state."""
        self.batch.pass_turn(torch.tensor([idx], device=self.device))
        self._legal_moves_cache.pop(idx, None)
        self._is_terminal_cache.pop(idx, None)

    def get_legal_moves(self, idx: int) -> List[Move]:
        """Get legal moves for a game state (computed on CPU)."""
        if idx in self._legal_moves_cache:
            return self._legal_moves_cache[idx]

        # Convert to CPU game and get moves
        cpu_game = self.batch.to_cpu_game(idx)
        moves = cpu_game.get_legal_moves()
        self._legal_moves_cache[idx] = moves
        return moves

    def is_terminal(self, idx: int) -> bool:
        """Check if game is over."""
        if idx in self._is_terminal_cache:
            return self._is_terminal_cache[idx]

        is_over = self.batch.is_game_over[idx].item()
        self._is_terminal_cache[idx] = is_over
        return is_over

    def get_current_player(self, idx: int) -> int:
        """Get current player for a game state."""
        return self.batch.current_player[idx].item()

    def get_winner(self, idx: int) -> Optional[int]:
        """Get winner of a finished game."""
        cpu_game = self.batch.to_cpu_game(idx)
        return cpu_game.get_winner()

    def get_state_for_nn(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get neural network input for a game state."""
        cpu_game = self.batch.to_cpu_game(idx)
        return cpu_game.get_state_for_neural_net()

    def batch_get_state_for_nn(self, indices: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get neural network inputs for multiple game states."""
        board_tensors = []
        pieces_vectors = []

        for idx in indices:
            bt, pv = self.get_state_for_nn(idx)
            board_tensors.append(bt)
            pieces_vectors.append(pv)

        return board_tensors, pieces_vectors

    def from_cpu_game(self, game: BlokusGame) -> int:
        """Import a CPU game into the pool."""
        idx = self.allocate()

        # Copy state
        self.batch.boards[idx] = torch.from_numpy(game.board.astype(np.int8)).to(self.device)

        for player in range(NUM_PLAYERS):
            ps = game.players[player]
            for piece_idx in range(21):
                self.batch.remaining_pieces[idx, player, piece_idx] = piece_idx in ps.remaining_pieces
            self.batch.has_played[idx, player] = ps.has_played
            self.batch.is_eliminated[idx, player] = ps.is_eliminated
            self.batch.scores[idx, player] = ps.score

            self.batch.valid_corners[idx, player] = False
            for r, c in game._valid_corners[player]:
                self.batch.valid_corners[idx, player, r, c] = True

        self.batch.current_player[idx] = game.current_player
        self.batch.turn_number[idx] = game.turn_number
        self.batch.consecutive_passes[idx] = game.consecutive_passes
        self.batch.is_game_over[idx] = game.is_game_over

        return idx


class BatchedMCTSGPU:
    """
    MCTS with GPU-accelerated game state management.

    Uses GPUGamePool for efficient batched game operations.
    """

    def __init__(
        self,
        network,
        config: MCTSConfig,
        device: str = 'cuda',
        pool_size: int = 2048,
    ):
        self.network = network
        self.config = config
        self.device = device
        self.pool = GPUGamePool(pool_size, device)
        self.virtual_loss = 3.0
        self.stats = GPUMCTSStats()

    def search(
        self,
        root_game: BlokusGame,
        add_noise: bool = True,
        batch_size: int = 16,
    ) -> Tuple[MCTSNodeGPU, GPUGamePool]:
        """
        Run batched MCTS from the given game state.

        Returns:
            root: Root node after search
            pool: Game pool containing all game states (caller should clean up)
        """
        # Import root game into pool
        root_idx = self.pool.from_cpu_game(root_game)
        root = MCTSNodeGPU(root_idx)

        if self.pool.is_terminal(root_idx):
            return root, self.pool

        legal_moves = self.pool.get_legal_moves(root_idx)
        if not legal_moves:
            return root, self.pool

        # Get initial policy
        board_tensor, pieces_vector = self.pool.get_state_for_nn(root_idx)
        policy, value = self.network.predict(
            board_tensor, pieces_vector, legal_moves, self.device
        )

        # Add Dirichlet noise
        if add_noise and len(policy) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(policy))
            policy = (1 - self.config.dirichlet_epsilon) * policy + \
                     self.config.dirichlet_epsilon * noise

        # Expand root
        root._child_priors = {move: policy[i] for i, move in enumerate(legal_moves)}
        root._is_expanded = True
        root._legal_moves = legal_moves

        perspective_player = self.pool.get_current_player(root_idx)

        # Run simulations
        simulations_done = 0
        while simulations_done < self.config.num_simulations:
            current_batch = min(batch_size, self.config.num_simulations - simulations_done)

            # Collect leaves
            leaves_data = []
            for _ in range(current_batch):
                leaf, path = self._select_leaf(root)
                leaves_data.append((leaf, path))

                if not self.pool.is_terminal(leaf.game_idx):
                    self._apply_virtual_loss(path)

            # Batch expand unexpanded leaves
            to_expand = []
            to_expand_indices = []

            for i, (leaf, path) in enumerate(leaves_data):
                if not self.pool.is_terminal(leaf.game_idx) and not leaf._is_expanded:
                    moves = self.pool.get_legal_moves(leaf.game_idx)
                    if moves:
                        leaf._legal_moves = moves
                        to_expand.append((i, leaf, moves))
                        to_expand_indices.append(i)

            # Batch NN evaluation
            if to_expand:
                board_tensors = []
                pieces_vectors = []
                moves_list = []

                for i, leaf, moves in to_expand:
                    bt, pv = self.pool.get_state_for_nn(leaf.game_idx)
                    board_tensors.append(bt)
                    pieces_vectors.append(pv)
                    moves_list.append(moves)

                policies, values = self.network.predict_batch(
                    board_tensors, pieces_vectors, moves_list, self.device
                )

                # Expand leaves
                for j, (i, leaf, moves) in enumerate(to_expand):
                    leaf._child_priors = {m: policies[j][k] for k, m in enumerate(moves)}
                    leaf._is_expanded = True

            # Backpropagation
            for i, (leaf, path) in enumerate(leaves_data):
                if self.pool.is_terminal(leaf.game_idx):
                    winner = self.pool.get_winner(leaf.game_idx)
                    if winner is not None:
                        value = 1.0 if winner == perspective_player else -1.0
                    else:
                        value = 0.0
                elif i in to_expand_indices:
                    j = to_expand_indices.index(i)
                    value = values[j]
                else:
                    value = leaf.q_value

                # Remove virtual loss
                if not self.pool.is_terminal(leaf.game_idx):
                    self._remove_virtual_loss(path)

                # Backpropagate
                self._backpropagate(leaf, value, perspective_player)

            simulations_done += current_batch

        return root, self.pool

    def _select_leaf(self, root: MCTSNodeGPU) -> Tuple[MCTSNodeGPU, List[MCTSNodeGPU]]:
        """Select a leaf node by traversing down the tree."""
        node = root
        path = [node]

        while node._is_expanded and not self.pool.is_terminal(node.game_idx):
            # Find best move (expanded or unexpanded)
            best_unexpanded = node.select_best_unexpanded(self.config.c_puct)

            if best_unexpanded is not None:
                # Need to expand this move
                child = self._create_child(node, best_unexpanded)
                path.append(child)
                return child, path
            else:
                # Select best expanded child
                best_child = max(node.children.values(), key=lambda c: c.ucb_score(self.config.c_puct))
                path.append(best_child)
                node = best_child

        return node, path

    def _create_child(self, parent: MCTSNodeGPU, move: Move) -> MCTSNodeGPU:
        """Create a child node by cloning parent state and applying move."""
        prior = parent._child_priors.get(move, 0.0)

        # Clone parent game state
        child_idx = self.pool.clone_to_new(parent.game_idx)

        # Apply move
        self.pool.apply_move(child_idx, move)

        # Auto-pass if needed
        while not self.pool.is_terminal(child_idx):
            moves = self.pool.get_legal_moves(child_idx)
            if moves:
                break
            self.pool.pass_turn(child_idx)

        child = MCTSNodeGPU(
            game_idx=child_idx,
            parent=parent,
            move=move,
            prior=prior,
        )
        parent.children[move] = child
        return child

    def _apply_virtual_loss(self, path: List[MCTSNodeGPU]):
        """Apply virtual loss to path."""
        for node in path[1:]:
            node.visit_count += 1
            node.value_sum -= self.virtual_loss

    def _remove_virtual_loss(self, path: List[MCTSNodeGPU]):
        """Remove virtual loss from path."""
        for node in path[1:]:
            node.visit_count -= 1
            node.value_sum += self.virtual_loss

    def _backpropagate(self, node: MCTSNodeGPU, value: float, perspective_player: int):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            if current.parent is not None:
                parent_player = self.pool.get_current_player(current.parent.game_idx)
                if parent_player == perspective_player:
                    current.value_sum += value
                else:
                    current.value_sum -= value
            current = current.parent

    def get_action_probs(
        self,
        root: MCTSNodeGPU,
        temperature: float = 1.0,
    ) -> Tuple[List[Move], np.ndarray]:
        """Get action probabilities from visit counts."""
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

    def select_move(
        self,
        game: BlokusGame,
        add_noise: bool = True,
        temperature: Optional[float] = None,
        batch_size: int = 16,
    ) -> Tuple[Move, np.ndarray, MCTSNodeGPU]:
        """
        Select a move using batched MCTS.

        Returns:
            selected_move: The chosen move
            policy: Action probabilities
            root: Root node
        """
        # Create fresh pool for this search
        self.pool = GPUGamePool(2048, self.device)

        root, pool = self.search(game, add_noise=add_noise, batch_size=batch_size)

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

        return selected_move, probs, root

    def cleanup(self):
        """Clean up GPU resources."""
        self.pool = GPUGamePool(2048, self.device)


class MultiGameMCTSGPU:
    """
    Run MCTS for multiple games simultaneously with shared GPU resources.

    This is the key optimization: instead of running MCTS for each game
    sequentially, we interleave MCTS simulations across multiple games
    and batch all neural network evaluations together.

    This achieves high GPU utilization by:
    1. Batching NN evaluations across all concurrent games
    2. Using GPU game pool for efficient state management
    3. Minimizing Python overhead through batched operations
    """

    def __init__(
        self,
        network,
        config: MCTSConfig,
        device: str = 'cuda',
    ):
        self.network = network
        self.config = config
        self.device = device
        self.virtual_loss = 3.0

    def run_mcts_step_for_games(
        self,
        games: List[BlokusGame],
        add_noise: bool = True,
    ) -> List[Tuple[Move, np.ndarray]]:
        """
        Run MCTS and select moves for multiple games simultaneously.

        Args:
            games: List of game states to evaluate
            add_noise: Whether to add Dirichlet noise at root

        Returns:
            List of (selected_move, policy) tuples, one per game
        """
        n_games = len(games)
        if n_games == 0:
            return []

        device = self.device

        # Create shared game pool
        pool = GPUGamePool(n_games * 100, device)  # Reserve space for tree nodes

        # Initialize roots for all games
        roots = []
        root_indices = []
        legal_moves_per_game = []
        active_mask = []  # Which games have legal moves

        for game in games:
            root_idx = pool.from_cpu_game(game)
            root = MCTSNodeGPU(root_idx)
            roots.append(root)
            root_indices.append(root_idx)

            if pool.is_terminal(root_idx):
                legal_moves_per_game.append([])
                active_mask.append(False)
            else:
                moves = pool.get_legal_moves(root_idx)
                legal_moves_per_game.append(moves)
                active_mask.append(len(moves) > 0)

        # Batch evaluate all roots
        active_indices = [i for i, active in enumerate(active_mask) if active]

        if active_indices:
            board_tensors = []
            pieces_vectors = []
            moves_list = []

            for i in active_indices:
                bt, pv = pool.get_state_for_nn(root_indices[i])
                board_tensors.append(bt)
                pieces_vectors.append(pv)
                moves_list.append(legal_moves_per_game[i])

            policies, values = self.network.predict_batch(
                board_tensors, pieces_vectors, moves_list, device
            )

            # Expand roots
            for j, i in enumerate(active_indices):
                policy = policies[j]
                moves = legal_moves_per_game[i]

                # Add Dirichlet noise
                if add_noise and len(policy) > 0:
                    noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(policy))
                    policy = (1 - self.config.dirichlet_epsilon) * policy + \
                             self.config.dirichlet_epsilon * noise

                roots[i]._child_priors = {m: policy[k] for k, m in enumerate(moves)}
                roots[i]._is_expanded = True
                roots[i]._legal_moves = moves

        # Get perspective players
        perspective_players = [pool.get_current_player(idx) for idx in root_indices]

        # Run MCTS simulations
        simulations_done = [0] * n_games
        batch_size = 16

        while any(s < self.config.num_simulations for s, active in zip(simulations_done, active_mask) if active):
            # Collect leaves from all active games
            all_leaves = []  # (game_idx, leaf, path)

            for game_idx in range(n_games):
                if not active_mask[game_idx]:
                    continue
                if simulations_done[game_idx] >= self.config.num_simulations:
                    continue

                sims_needed = min(batch_size, self.config.num_simulations - simulations_done[game_idx])

                for _ in range(sims_needed):
                    leaf, path = self._select_leaf(roots[game_idx], pool)
                    all_leaves.append((game_idx, leaf, path))

                    if not pool.is_terminal(leaf.game_idx):
                        self._apply_virtual_loss(path)

            if not all_leaves:
                break

            # Batch evaluate unexpanded leaves
            to_evaluate = []
            for i, (game_idx, leaf, path) in enumerate(all_leaves):
                if not pool.is_terminal(leaf.game_idx) and not leaf._is_expanded:
                    moves = pool.get_legal_moves(leaf.game_idx)
                    if moves:
                        leaf._legal_moves = moves
                        to_evaluate.append((i, leaf, moves))

            if to_evaluate:
                eval_board_tensors = []
                eval_pieces_vectors = []
                eval_moves_list = []

                for i, leaf, moves in to_evaluate:
                    bt, pv = pool.get_state_for_nn(leaf.game_idx)
                    eval_board_tensors.append(bt)
                    eval_pieces_vectors.append(pv)
                    eval_moves_list.append(moves)

                eval_policies, eval_values = self.network.predict_batch(
                    eval_board_tensors, eval_pieces_vectors, eval_moves_list, device
                )

                # Expand leaves
                for j, (i, leaf, moves) in enumerate(to_evaluate):
                    leaf._child_priors = {m: eval_policies[j][k] for k, m in enumerate(moves)}
                    leaf._is_expanded = True

            # Backpropagate
            eval_idx_map = {to_evaluate[j][0]: j for j in range(len(to_evaluate))}

            for i, (game_idx, leaf, path) in enumerate(all_leaves):
                if pool.is_terminal(leaf.game_idx):
                    winner = pool.get_winner(leaf.game_idx)
                    if winner is not None:
                        value = 1.0 if winner == perspective_players[game_idx] else -1.0
                    else:
                        value = 0.0
                elif i in eval_idx_map:
                    value = eval_values[eval_idx_map[i]]
                else:
                    value = leaf.q_value

                # Remove virtual loss
                if not pool.is_terminal(leaf.game_idx):
                    self._remove_virtual_loss(path)

                # Backpropagate
                self._backpropagate(leaf, value, perspective_players[game_idx], pool)
                simulations_done[game_idx] += 1

        # Select moves for all games
        results = []
        for game_idx in range(n_games):
            if not active_mask[game_idx]:
                results.append((None, np.array([])))
                continue

            root = roots[game_idx]
            game = games[game_idx]

            # Determine temperature
            if game.turn_number <= self.config.temperature_threshold:
                temperature = self.config.temperature
            else:
                temperature = 0.1

            # Get action probs
            moves = []
            visits = []
            for move, child in root.children.items():
                moves.append(move)
                visits.append(child.visit_count)

            if not moves:
                results.append((None, np.array([])))
                continue

            visits = np.array(visits, dtype=np.float32)
            if temperature < 0.01:
                probs = np.zeros_like(visits)
                probs[np.argmax(visits)] = 1.0
            else:
                visits = visits ** (1.0 / temperature)
                probs = visits / visits.sum()

            # Sample move
            idx = np.random.choice(len(moves), p=probs)
            selected_move = moves[idx]

            results.append((selected_move, probs))

        return results

    def _select_leaf(self, root: MCTSNodeGPU, pool: GPUGamePool) -> Tuple[MCTSNodeGPU, List[MCTSNodeGPU]]:
        """Select a leaf node by traversing down the tree."""
        node = root
        path = [node]

        while node._is_expanded and not pool.is_terminal(node.game_idx):
            best_unexpanded = node.select_best_unexpanded(self.config.c_puct)

            if best_unexpanded is not None:
                child = self._create_child(node, best_unexpanded, pool)
                path.append(child)
                return child, path
            else:
                best_child = max(node.children.values(), key=lambda c: c.ucb_score(self.config.c_puct))
                path.append(best_child)
                node = best_child

        return node, path

    def _create_child(self, parent: MCTSNodeGPU, move: Move, pool: GPUGamePool) -> MCTSNodeGPU:
        """Create a child node."""
        prior = parent._child_priors.get(move, 0.0)
        child_idx = pool.clone_to_new(parent.game_idx)
        pool.apply_move(child_idx, move)

        # Auto-pass
        while not pool.is_terminal(child_idx):
            moves = pool.get_legal_moves(child_idx)
            if moves:
                break
            pool.pass_turn(child_idx)

        child = MCTSNodeGPU(game_idx=child_idx, parent=parent, move=move, prior=prior)
        parent.children[move] = child
        return child

    def _apply_virtual_loss(self, path: List[MCTSNodeGPU]):
        for node in path[1:]:
            node.visit_count += 1
            node.value_sum -= self.virtual_loss

    def _remove_virtual_loss(self, path: List[MCTSNodeGPU]):
        for node in path[1:]:
            node.visit_count -= 1
            node.value_sum += self.virtual_loss

    def _backpropagate(self, node: MCTSNodeGPU, value: float, perspective_player: int, pool: GPUGamePool):
        current = node
        while current is not None:
            current.visit_count += 1
            if current.parent is not None:
                parent_player = pool.get_current_player(current.parent.game_idx)
                if parent_player == perspective_player:
                    current.value_sum += value
                else:
                    current.value_sum -= value
            current = current.parent


def benchmark_gpu_mcts():
    """Benchmark GPU MCTS vs standard MCTS."""
    from .network import BlokusNet
    from .mcts import BatchedMCTS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on device: {device}")

    # Create network
    network = BlokusNet().to(device)
    network.eval()

    config = MCTSConfig(num_simulations=100)

    # Test game
    game = BlokusGame()

    # Standard MCTS
    print("\nStandard BatchedMCTS:")
    standard_mcts = BatchedMCTS(network, config, device)

    start = time.time()
    torch.cuda.synchronize() if device == 'cuda' else None

    for _ in range(3):
        move, probs, root = standard_mcts.select_move(game, batch_size=16)

    torch.cuda.synchronize() if device == 'cuda' else None
    standard_time = (time.time() - start) / 3
    print(f"  Time per move: {standard_time*1000:.1f}ms")

    # GPU MCTS - single game
    print("\nGPU BatchedMCTS (single game):")
    gpu_mcts = BatchedMCTSGPU(network, config, device)

    start = time.time()
    torch.cuda.synchronize() if device == 'cuda' else None

    for _ in range(3):
        move, probs, root = gpu_mcts.select_move(game, batch_size=16)

    torch.cuda.synchronize() if device == 'cuda' else None
    gpu_time = (time.time() - start) / 3
    print(f"  Time per move: {gpu_time*1000:.1f}ms")

    # Multi-game GPU MCTS
    print("\nMultiGame GPU MCTS (8 games parallel):")
    multi_mcts = MultiGameMCTSGPU(network, config, device)
    games = [BlokusGame() for _ in range(8)]

    start = time.time()
    torch.cuda.synchronize() if device == 'cuda' else None

    for _ in range(3):
        results = multi_mcts.run_mcts_step_for_games(games)

    torch.cuda.synchronize() if device == 'cuda' else None
    multi_time = (time.time() - start) / 3 / 8  # Per game
    print(f"  Time per game: {multi_time*1000:.1f}ms")
    print(f"  Total time for 8 games: {multi_time*8*1000:.1f}ms")

    print(f"\nSpeedup vs standard (per game): {standard_time/multi_time:.2f}x")


if __name__ == "__main__":
    benchmark_gpu_mcts()
