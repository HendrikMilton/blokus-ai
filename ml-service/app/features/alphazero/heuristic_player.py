"""Pure heuristic player for baseline evaluation.

Used to verify that the trained neural network doesn't regress below
handcrafted-strategy level. The model should consistently beat this
baseline (>70% win rate) as a sanity check.
"""

from typing import Optional
import numpy as np

from .game import BlokusGame, Move
from .heuristics import HeuristicConfig, compute_heuristic_policy


class HeuristicPlayer:
    """
    A player that selects moves using only handcrafted heuristics.
    Used as a baseline opponent to ensure NN doesn't regress.
    """

    def __init__(self, config: Optional[HeuristicConfig] = None):
        self.config = config or HeuristicConfig()

    def select_move(self, game: BlokusGame) -> Optional[Move]:
        """Select the best move according to heuristic scoring."""
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None

        # Get heuristic policy (probability distribution)
        policy = compute_heuristic_policy(game, legal_moves, self.config)

        # Select move with highest probability (greedy)
        best_idx = int(np.argmax(policy))
        return legal_moves[best_idx]
