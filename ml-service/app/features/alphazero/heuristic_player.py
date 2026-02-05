"""Pure heuristic player for baseline evaluation.

Used to verify that the trained neural network doesn't regress below
handcrafted-strategy level. The model should consistently beat this
baseline (>70% win rate) as a sanity check.

This module provides two player classes:
- HeuristicPlayer: Simple greedy player (original implementation)
- SmartPlayer: Advanced player with lookahead (from smart_player.py)

The HeuristicPlayer now supports difficulty levels that map to SmartPlayer
configurations for stronger play.
"""

from typing import Optional
from enum import Enum
import numpy as np

from .game import BlokusGame, Move
from .heuristics import HeuristicConfig, compute_heuristic_policy


class Difficulty(Enum):
    """AI difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class HeuristicPlayer:
    """
    A player that selects moves using only handcrafted heuristics.
    Used as a baseline opponent to ensure NN doesn't regress.

    Supports multiple difficulty levels:
    - EASY: Greedy with basic heuristics
    - MEDIUM: Lookahead with moderate candidate evaluation
    - HARD: Full lookahead with all features
    - EXPERT: Deeper search for strongest play
    """

    def __init__(
        self,
        config: Optional[HeuristicConfig] = None,
        difficulty: Difficulty = Difficulty.EASY,
        use_smart_player: bool = False,
    ):
        """
        Initialize HeuristicPlayer.

        Args:
            config: Heuristic configuration (used for EASY mode or when use_smart_player=False)
            difficulty: AI difficulty level
            use_smart_player: If True, use SmartPlayer for MEDIUM+ difficulties
        """
        self.config = config or HeuristicConfig()
        self.difficulty = difficulty
        self.use_smart_player = use_smart_player

        # Lazy-load SmartPlayer to avoid circular imports
        self._smart_player = None

    def _get_smart_player(self):
        """Get or create SmartPlayer instance based on difficulty."""
        if self._smart_player is None and self.use_smart_player:
            from .smart_player import SmartPlayer, SmartPlayerConfig

            if self.difficulty == Difficulty.EASY:
                self._smart_player = SmartPlayer(SmartPlayerConfig.easy())
            elif self.difficulty == Difficulty.MEDIUM:
                self._smart_player = SmartPlayer(SmartPlayerConfig.medium())
            elif self.difficulty == Difficulty.HARD:
                self._smart_player = SmartPlayer(SmartPlayerConfig.hard())
            elif self.difficulty == Difficulty.EXPERT:
                self._smart_player = SmartPlayer(SmartPlayerConfig.expert())
            else:
                self._smart_player = SmartPlayer(SmartPlayerConfig.medium())

        return self._smart_player

    def select_move(self, game: BlokusGame) -> Optional[Move]:
        """Select the best move according to heuristic scoring."""
        # Use SmartPlayer for higher difficulties if enabled
        if self.use_smart_player and self.difficulty != Difficulty.EASY:
            smart_player = self._get_smart_player()
            if smart_player:
                return smart_player.select_move(game)

        # Fallback to simple greedy selection
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None

        # Get heuristic policy (probability distribution)
        policy = compute_heuristic_policy(game, legal_moves, self.config)

        # Select move with highest probability (greedy)
        best_idx = int(np.argmax(policy))
        return legal_moves[best_idx]

    def select_move_with_temperature(
        self,
        game: BlokusGame,
        temperature: float = 1.0,
    ) -> Optional[Move]:
        """
        Select a move with temperature-based sampling.

        Args:
            game: Current game state
            temperature: Sampling temperature (0 = greedy, higher = more random)

        Returns:
            Selected move
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None

        # Get heuristic policy
        policy = compute_heuristic_policy(game, legal_moves, self.config)

        if temperature == 0:
            # Greedy selection
            best_idx = int(np.argmax(policy))
        else:
            # Temperature-based sampling
            policy = policy ** (1.0 / temperature)
            policy = policy / policy.sum()
            best_idx = np.random.choice(len(legal_moves), p=policy)

        return legal_moves[best_idx]


def create_player(difficulty: str = "medium") -> HeuristicPlayer:
    """
    Factory function to create a heuristic player with specified difficulty.

    Args:
        difficulty: "easy", "medium", "hard", or "expert"

    Returns:
        HeuristicPlayer instance
    """
    difficulty_map = {
        "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM,
        "hard": Difficulty.HARD,
        "expert": Difficulty.EXPERT,
    }

    diff = difficulty_map.get(difficulty.lower(), Difficulty.MEDIUM)

    return HeuristicPlayer(
        difficulty=diff,
        use_smart_player=True,
    )
