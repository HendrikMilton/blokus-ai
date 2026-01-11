"""
Monte Carlo Tree Search implementation for Blokus.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from app.schemas import PlayerColor

from .game_state import GameState, Move
from .heuristics import evaluate_move, evaluate_state
from .move_generator import generate_moves_fast


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: GameState
    color: PlayerColor  # The player who will move from this node
    parent: Optional["MCTSNode"] = None
    move: Optional[Move] = None  # Move that led to this node
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_moves: Optional[list[Move]] = None

    def __post_init__(self):
        if self.untried_moves is None:
            self.untried_moves = generate_moves_fast(self.state, self.color, max_moves=30)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.state.is_terminal() or (
            self.is_fully_expanded() and len(self.children) == 0
        )

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float("inf")

        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration


class MCTS:
    """Monte Carlo Tree Search for Blokus."""

    def __init__(
        self,
        iterations: int = 1000,
        time_limit_ms: int = 3000,
        exploration_constant: float = 1.414,
    ):
        self.iterations = iterations
        self.time_limit_ms = time_limit_ms
        self.c = exploration_constant

    def search(self, state: GameState, color: PlayerColor) -> Optional[Move]:
        """
        Run MCTS and return the best move.
        """
        root = MCTSNode(state=state, color=color)

        # Check if any moves available
        if not root.untried_moves and not root.children:
            return None

        deadline = time.perf_counter() + (self.time_limit_ms / 1000)

        for iteration in range(self.iterations):
            if time.perf_counter() > deadline:
                break

            # Selection
            node = self._select(root)

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)

            # Simulation
            result = self._simulate(node, color)

            # Backpropagation
            self._backpropagate(node, result)

        # Return most visited child's move
        return self._best_move(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select most promising node using UCB1."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node

            if not node.children:
                return node

            node = max(node.children, key=lambda n: n.ucb1(self.c))

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand one child node."""
        if not node.untried_moves:
            return node

        # Pick a move (prefer higher-scored moves)
        moves_with_scores = [
            (move, evaluate_move(node.state, move, node.color))
            for move in node.untried_moves
        ]
        moves_with_scores.sort(key=lambda x: -x[1])

        # Weighted random selection favoring better moves
        if len(moves_with_scores) > 1 and random.random() < 0.7:
            move = moves_with_scores[0][0]
        else:
            move = random.choice(node.untried_moves)

        node.untried_moves.remove(move)

        # Create new state
        new_state = node.state.apply_move(move, node.color)
        next_color = new_state.current_player

        child = MCTSNode(
            state=new_state,
            color=next_color,
            parent=node,
            move=move,
        )
        node.children.append(child)

        return child

    def _simulate(self, node: MCTSNode, original_color: PlayerColor) -> float:
        """
        Simulate game from node to terminal state.
        Returns score from perspective of original_color.
        """
        state = node.state.copy()
        current_color = node.color
        max_depth = 20  # Limit simulation depth

        for _ in range(max_depth):
            if state.is_terminal():
                break

            moves = generate_moves_fast(state, current_color, max_moves=10)

            if not moves:
                state = state.pass_turn()
            else:
                # Heuristic-guided move selection (70% best, 30% random)
                if random.random() < 0.7:
                    move = max(moves, key=lambda m: evaluate_move(state, m, current_color))
                else:
                    move = random.choice(moves)

                state = state.apply_move(move, current_color)

            current_color = state.current_player

        # Evaluate final state for original player
        return evaluate_state(state, original_color)

    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate result up the tree."""
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def _best_move(self, root: MCTSNode) -> Optional[Move]:
        """Return the move with most visits."""
        if not root.children:
            # No children, try to return any untried move
            if root.untried_moves:
                return root.untried_moves[0]
            return None

        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.move
