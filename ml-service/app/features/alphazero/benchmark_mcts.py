"""
Benchmark MCTS with original vs optimized game implementation.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from .network import BlokusNet
from .game import BlokusGame, Move
from .game_optimized import BlokusGameOptimized, Move as OptMove


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 80
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 20


class MCTSNodeOpt:
    """MCTS node using optimized game."""

    def __init__(self, game: BlokusGameOptimized, parent=None, move=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict = {}
        self._legal_moves = None
        self._is_expanded = False
        self._child_priors = None

    @property
    def legal_moves(self):
        if self._legal_moves is None:
            self._legal_moves = self.game.get_legal_moves()
        return self._legal_moves

    @property
    def is_terminal(self):
        return self.game.is_game_over

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct):
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + exploration

    def _get_unexpanded_ucb(self, move, c_puct):
        prior = self._child_priors.get(move, 0.0)
        return c_puct * prior * math.sqrt(self.visit_count) / 1

    def select_child(self, c_puct):
        best_score = float('-inf')
        best_child = None
        best_unexpanded_move = None

        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
                best_unexpanded_move = None

        if self._child_priors:
            for move, prior in self._child_priors.items():
                if move not in self.children:
                    score = self._get_unexpanded_ucb(move, c_puct)
                    if score > best_score:
                        best_score = score
                        best_child = None
                        best_unexpanded_move = move

        if best_unexpanded_move is not None:
            best_child = self._expand_child(best_unexpanded_move)

        return best_child

    def _expand_child(self, move):
        prior = self._child_priors.get(move, 0.0)
        child_game = self.game.clone()
        child_game.make_move(move)

        while not child_game.is_game_over and not child_game.get_legal_moves():
            child_game.pass_turn()

        child = MCTSNodeOpt(child_game, parent=self, move=move, prior=prior)
        self.children[move] = child
        return child

    def expand(self, policy, legal_moves):
        self._child_priors = {move: policy[i] for i, move in enumerate(legal_moves)}
        self._is_expanded = True

    def backpropagate(self, value, perspective_player):
        node = self
        while node is not None:
            node.visit_count += 1
            if node.parent is not None:
                parent_player = node.parent.game.current_player
                if parent_player == perspective_player:
                    node.value_sum += value
                else:
                    node.value_sum -= value
            node = node.parent


class BatchedMCTSOpt:
    """Batched MCTS using optimized game."""

    def __init__(self, network, config, device='cuda'):
        self.network = network
        self.config = config
        self.device = device
        self.virtual_loss = 3.0

    def _select_leaf(self, root, c_puct):
        node = root
        path = [node]
        while node._is_expanded and not node.is_terminal:
            node = node.select_child(c_puct)
            path.append(node)
        return node, path

    def _apply_virtual_loss(self, path):
        for node in path[1:]:
            node.visit_count += 1
            node.value_sum -= self.virtual_loss

    def _remove_virtual_loss(self, path):
        for node in path[1:]:
            node.visit_count -= 1
            node.value_sum += self.virtual_loss

    def _get_batch_policy_value(self, games, legal_moves_list):
        board_tensors = []
        pieces_vectors = []
        for game in games:
            bt, pv = game.get_state_for_neural_net()
            board_tensors.append(bt)
            pieces_vectors.append(pv)

        # Convert moves to tuples for network
        moves_tuples = []
        for moves in legal_moves_list:
            moves_tuples.append([
                (m.piece_idx, m.orientation_idx, m.row, m.col) for m in moves
            ])

        return self.network.predict_batch(
            board_tensors, pieces_vectors, moves_tuples, self.device
        )

    def search(self, root_game, add_noise=True, batch_size=16):
        root = MCTSNodeOpt(root_game.clone())

        if root.is_terminal:
            return root

        legal_moves = root.legal_moves
        if not legal_moves:
            return root

        # Initial policy
        board_tensor, pieces_vector = root.game.get_state_for_neural_net()
        moves_tuples = [(m.piece_idx, m.orientation_idx, m.row, m.col) for m in legal_moves]
        policy, value = self.network.predict(
            board_tensor, pieces_vector, moves_tuples, self.device
        )

        if add_noise and len(policy) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(policy))
            policy = (1 - self.config.dirichlet_epsilon) * policy + self.config.dirichlet_epsilon * noise

        root.expand(policy, legal_moves)
        perspective_player = root.game.current_player

        simulations_done = 0
        while simulations_done < self.config.num_simulations:
            current_batch = min(batch_size, self.config.num_simulations - simulations_done)

            leaves = []
            paths = []
            leaf_legal_moves = []

            for _ in range(current_batch):
                leaf, path = self._select_leaf(root, self.config.c_puct)
                leaves.append(leaf)
                paths.append(path)

                if not leaf.is_terminal:
                    self._apply_virtual_loss(path)
                    leaf_legal_moves.append(leaf.legal_moves)
                else:
                    leaf_legal_moves.append([])

            # Batch evaluate
            to_evaluate_indices = []
            to_evaluate_games = []
            to_evaluate_moves = []

            for i, (leaf, moves) in enumerate(zip(leaves, leaf_legal_moves)):
                if not leaf.is_terminal and moves and not leaf._is_expanded:
                    to_evaluate_indices.append(i)
                    to_evaluate_games.append(leaf.game)
                    to_evaluate_moves.append(moves)

            if to_evaluate_games:
                policies, values = self._get_batch_policy_value(
                    to_evaluate_games, to_evaluate_moves
                )
            else:
                policies, values = [], []

            # Expand and backprop
            eval_idx = 0
            for i, (leaf, path, moves) in enumerate(zip(leaves, paths, leaf_legal_moves)):
                if leaf.is_terminal:
                    winner = leaf.game.get_winner()
                    if winner is not None:
                        value = 1.0 if winner == perspective_player else -1.0
                    else:
                        value = 0.0
                elif i in to_evaluate_indices:
                    policy = policies[eval_idx]
                    value = values[eval_idx]
                    leaf.expand(policy, moves)
                    self._remove_virtual_loss(path)
                    eval_idx += 1
                elif moves:
                    value = leaf.q_value
                    self._remove_virtual_loss(path)
                else:
                    value = 0.0
                    self._remove_virtual_loss(path)

                leaf.backpropagate(value, perspective_player)

            simulations_done += current_batch

        return root

    def get_action_probs(self, root, temperature=1.0):
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

    def select_move(self, game, add_noise=True, temperature=None, batch_size=16):
        # Convert to optimized game if needed
        if isinstance(game, BlokusGame):
            opt_game = BlokusGameOptimized()
            opt_game.board = game.board.copy()
            opt_game.players = [p.clone() for p in game.players]
            opt_game.current_player = game.current_player
            opt_game.turn_number = game.turn_number
            opt_game.consecutive_passes = game.consecutive_passes
            opt_game.is_game_over = game.is_game_over
            opt_game._valid_corners = [s.copy() for s in game._valid_corners]
        else:
            opt_game = game

        root = self.search(opt_game, add_noise=add_noise, batch_size=batch_size)

        if temperature is None:
            if opt_game.turn_number <= self.config.temperature_threshold:
                temperature = self.config.temperature
            else:
                temperature = 0.1

        moves, probs = self.get_action_probs(root, temperature)

        if not moves:
            raise ValueError("No legal moves available")

        idx = np.random.choice(len(moves), p=probs)
        selected_move = moves[idx]

        # Convert back to original Move type if needed
        from .game import Move as OrigMove
        orig_move = OrigMove(
            selected_move.player,
            selected_move.piece_idx,
            selected_move.orientation_idx,
            selected_move.row,
            selected_move.col
        )

        return orig_move, probs, root


def benchmark():
    """Run benchmark comparing original vs optimized MCTS."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    network = BlokusNet().to(device)
    network.eval()

    config = MCTSConfig(num_simulations=80)

    # Original MCTS
    print("=== Original MCTS (BlokusGame) ===")
    from .mcts import BatchedMCTS
    original_mcts = BatchedMCTS(network, config, device)
    game = BlokusGame()

    # Warmup
    with torch.no_grad():
        original_mcts.select_move(game, batch_size=16)

    start = time.time()
    torch.cuda.synchronize()
    with torch.no_grad():
        for i in range(5):
            move, probs, root = original_mcts.select_move(game, batch_size=16)
    torch.cuda.synchronize()
    original_time = (time.time() - start) / 5
    print(f"  Time per MCTS decision: {original_time*1000:.0f}ms")

    # Optimized MCTS
    print()
    print("=== Optimized MCTS (BlokusGameOptimized) ===")
    optimized_mcts = BatchedMCTSOpt(network, config, device)

    # Warmup (includes numba compilation)
    opt_game = BlokusGameOptimized()
    _ = opt_game.get_legal_moves()  # Compile numba

    with torch.no_grad():
        optimized_mcts.select_move(game, batch_size=16)

    start = time.time()
    torch.cuda.synchronize()
    with torch.no_grad():
        for i in range(5):
            move, probs, root = optimized_mcts.select_move(game, batch_size=16)
    torch.cuda.synchronize()
    optimized_time = (time.time() - start) / 5
    print(f"  Time per MCTS decision: {optimized_time*1000:.0f}ms")

    print()
    print(f"=== Speedup: {original_time/optimized_time:.1f}x ===")

    # Profile optimized
    print()
    print("=== Profiling Optimized MCTS ===")
    import cProfile
    import pstats
    from io import StringIO

    pr = cProfile.Profile()
    pr.enable()

    with torch.no_grad():
        optimized_mcts.select_move(game, batch_size=16)

    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    benchmark()
