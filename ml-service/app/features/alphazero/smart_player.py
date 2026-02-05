"""
Strong heuristic player with lookahead for Blokus.

SmartPlayer implements a two-level search:
1. Fast filter: Score all legal moves with basic heuristics, keep top N
2. Deep evaluation: Evaluate candidates with mobility lookahead and
   minimax-lite opponent simulation

This provides much stronger play than the greedy HeuristicPlayer without
requiring neural network training.

Performance targets:
- Depth 0: ~50ms per move (fast heuristic only)
- Depth 1: ~200-500ms per move (with mobility + opponent sim)
- Depth 2: ~2-5s per move (deeper search, for analysis)
"""

import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from app.logging_config import log
from .game import BlokusGame, Move
from .heuristics import (
    HeuristicConfig,
    ScoringContext,
    compute_move_heuristic_score,
    compute_move_heuristic_score_deep,
    compute_heuristic_policy,
    count_moves_after,
    corner_differential,
    get_game_phase,
    get_phase_weights,
)
from .pieces import PIECE_BY_INDEX


@dataclass
class SmartPlayerConfig:
    """Configuration for SmartPlayer."""

    # Search parameters
    lookahead_depth: int = 1          # 0 = greedy, 1 = single lookahead, 2 = deeper
    candidate_count: int = 15         # Number of top candidates for deep evaluation
    opponent_candidate_count: int = 8  # Candidates to consider for opponent response

    # Feature toggles (inherited from HeuristicConfig)
    use_phase_weights: bool = True
    use_mobility_lookahead: bool = True
    use_corner_differential: bool = True
    use_opponent_blocking: bool = True
    use_piece_timing: bool = True

    # Performance settings
    max_time_ms: Optional[int] = None  # Optional time limit per move

    # Difficulty presets
    @classmethod
    def easy(cls) -> 'SmartPlayerConfig':
        """Easy difficulty - greedy with basic heuristics."""
        return cls(
            lookahead_depth=0,
            candidate_count=5,
            use_mobility_lookahead=False,
            use_corner_differential=False,
        )

    @classmethod
    def medium(cls) -> 'SmartPlayerConfig':
        """Medium difficulty - single lookahead."""
        return cls(
            lookahead_depth=1,
            candidate_count=12,
            opponent_candidate_count=5,
        )

    @classmethod
    def hard(cls) -> 'SmartPlayerConfig':
        """Hard difficulty - greedy with improved heuristics.

        Lookahead is disabled as testing showed it hurts performance.
        The exponential piece size scoring provides strong play without
        needing to look ahead.
        """
        return cls(
            lookahead_depth=0,
            candidate_count=20,
            use_mobility_lookahead=False,
            use_corner_differential=False,
            use_opponent_blocking=True,
        )

    @classmethod
    def hard_lookahead(cls) -> 'SmartPlayerConfig':
        """Hard difficulty with lookahead (for comparison testing).

        Uses lookahead but with mobility disabled to prevent small piece
        selection in endgame.
        """
        return cls(
            lookahead_depth=1,
            candidate_count=15,
            opponent_candidate_count=8,
            use_mobility_lookahead=False,  # Disabled - was causing issues
            use_corner_differential=True,
            use_opponent_blocking=True,
        )

    @classmethod
    def expert(cls) -> 'SmartPlayerConfig':
        """Expert difficulty - deeper search."""
        return cls(
            lookahead_depth=2,
            candidate_count=20,
            opponent_candidate_count=10,
        )


class SmartPlayer:
    """
    Strong heuristic player with lookahead.

    Implements a two-level search strategy:
    1. Fast scoring of all legal moves to get top candidates
    2. Deep evaluation of candidates with lookahead

    The depth parameter controls how far ahead we look:
    - depth=0: Pure heuristic (like HeuristicPlayer but with better heuristics)
    - depth=1: Consider mobility after move + basic opponent response
    - depth=2: Consider opponent's best response and our counter
    """

    def __init__(self, config: Optional[SmartPlayerConfig] = None):
        self.config = config or SmartPlayerConfig()
        self._heuristic_config = self._build_heuristic_config()

        # Statistics
        self.last_move_time_ms: float = 0
        self.last_candidates_evaluated: int = 0

    def _build_heuristic_config(self) -> HeuristicConfig:
        """Build HeuristicConfig from SmartPlayerConfig."""
        return HeuristicConfig(
            use_phase_weights=self.config.use_phase_weights,
            use_mobility_lookahead=self.config.use_mobility_lookahead,
            use_corner_differential=self.config.use_corner_differential,
            use_opponent_blocking=self.config.use_opponent_blocking,
            use_piece_timing=self.config.use_piece_timing,
        )

    def select_move(self, game: BlokusGame, verbose: bool = False) -> Optional[Move]:
        """
        Select the best move for the current player.

        Returns:
            Best move according to heuristic evaluation, or None if no legal moves.
        """
        start_time = time.time()

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None

        player = game.current_player
        turn = game.turn_number

        # If only one legal move, return it immediately
        if len(legal_moves) == 1:
            self.last_move_time_ms = (time.time() - start_time) * 1000
            self.last_candidates_evaluated = 1
            if verbose:
                move = legal_moves[0]
                piece = PIECE_BY_INDEX[move.piece_idx]
                log.info(f"P{player} T{turn}: Only 1 legal move: {piece.id} at ({move.row},{move.col})")
            return legal_moves[0]

        # Phase 1: Get top candidates using fast scoring
        candidates = self._get_top_candidates(game, legal_moves)
        self.last_candidates_evaluated = len(candidates)

        # Phase 2: Deep evaluation based on depth
        if self.config.lookahead_depth == 0:
            # Depth 0: Just return best candidate from phase 1
            best_move = candidates[0][1]  # (score, move) tuple
            best_score = candidates[0][0]
        else:
            # Depth 1+: Deep evaluation with lookahead
            best_move, best_score = self._evaluate_candidates_deep(game, candidates)

        self.last_move_time_ms = (time.time() - start_time) * 1000

        if verbose:
            piece = PIECE_BY_INDEX[best_move.piece_idx]
            log.info(
                f"P{player} T{turn}: {len(legal_moves)} moves, {len(candidates)} candidates, "
                f"chose {piece.id} (size={piece.size}) at ({best_move.row},{best_move.col}) "
                f"score={best_score:.2f} in {self.last_move_time_ms:.0f}ms"
            )
            # Log top 3 candidates
            for i, (score, move) in enumerate(candidates[:3]):
                p = PIECE_BY_INDEX[move.piece_idx]
                marker = " <-- CHOSEN" if move == best_move else ""
                log.info(f"  #{i+1}: {p.id} (size={p.size}) at ({move.row},{move.col}) fast_score={score:.2f}{marker}")

        return best_move

    def _get_top_candidates(
        self,
        game: BlokusGame,
        legal_moves: List[Move],
    ) -> List[Tuple[float, Move]]:
        """
        Get top N candidates by fast heuristic scoring.

        Returns:
            List of (score, move) tuples, sorted by score descending.
        """
        # Create scoring context for batch evaluation (caches expensive computations)
        ctx = ScoringContext.create(game)

        # Score all moves with basic (fast) heuristics using cached context
        scored_moves = []
        for move in legal_moves:
            score = compute_move_heuristic_score(move, game, self._heuristic_config, ctx)
            scored_moves.append((score, move))

        # Sort by score descending and take top N
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        n = min(self.config.candidate_count, len(scored_moves))
        return scored_moves[:n]

    def _evaluate_candidates_deep(
        self,
        game: BlokusGame,
        candidates: List[Tuple[float, Move]],
    ) -> Tuple[Move, float]:
        """
        Evaluate candidates with lookahead.

        Returns:
            (best_move, best_score)
        """
        best_move = candidates[0][1]
        best_score = float('-inf')

        for base_score, move in candidates:
            score = self._evaluate_move_deep(game, move, self.config.lookahead_depth)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move, best_score

    def _evaluate_move_deep(
        self,
        game: BlokusGame,
        move: Move,
        depth: int,
    ) -> float:
        """
        Evaluate a move with lookahead.

        Uses binary survival check for mobility instead of continuous bonus
        to prevent mobility from overriding piece size selection.

        Args:
            game: Current game state
            move: Move to evaluate
            depth: Remaining search depth

        Returns:
            Heuristic score for the move
        """
        # Get deep heuristic score (includes mobility and corner diff)
        score = compute_move_heuristic_score_deep(
            move, game, self._heuristic_config,
            include_mobility=self.config.use_mobility_lookahead,
            include_corner_diff=self.config.use_corner_differential,
        )

        if depth <= 0:
            return score

        # Simulate the move
        game_copy = game.clone()
        game_copy.make_move(move)

        # Skip eliminated players
        while not game_copy.is_game_over and game_copy.players[game_copy.current_player].is_eliminated:
            game_copy.pass_turn()

        if game_copy.is_game_over:
            # Game ended - evaluate final position
            scores = game_copy.get_scores()
            my_score = scores[move.player]
            max_opp = max(scores[p] for p in range(4) if p != move.player)
            return score + (my_score - max_opp) * 0.5

        # Binary survival check: penalize if we'll have few/no moves
        my_future_moves = len(game_copy.get_legal_moves(move.player))
        if my_future_moves == 0:
            score -= 10.0  # Can't play = terrible
        elif my_future_moves < 5:
            score -= 1.0   # Cornered = bad
        # Otherwise no bonus - don't let mobility override piece size

        # Evaluate opponent's best response
        opponent_value = self._simulate_opponent_turn(game_copy, move.player, depth - 1)

        # Discount opponent's gains
        score -= opponent_value * 0.3

        return score

    def _simulate_opponent_turn(
        self,
        game: BlokusGame,
        our_player: int,
        remaining_depth: int,
    ) -> float:
        """
        Simulate opponent's turn and return the value of their best move.

        We use a minimax-lite approach: assume opponent plays their best move
        according to our heuristics, and return how much that hurts us.

        Args:
            game: Game state after our move
            our_player: Our player index (to calculate damage to us)
            remaining_depth: Remaining search depth

        Returns:
            Value of opponent's best move (higher = worse for us)
        """
        opponent = game.current_player
        opponent_moves = game.get_legal_moves()

        if not opponent_moves:
            return 0.0

        # Score opponent moves quickly
        scored_moves = []
        for move in opponent_moves:
            score = compute_move_heuristic_score(move, game, self._heuristic_config)
            scored_moves.append((score, move))

        # Take top candidates
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        n = min(self.config.opponent_candidate_count, len(scored_moves))
        candidates = scored_moves[:n]

        # Find opponent's best move
        best_opponent_value = 0.0

        for base_score, opp_move in candidates:
            # Simulate opponent move
            game_copy = game.clone()
            game_copy.make_move(opp_move)

            # Calculate how much this hurts us
            # Check if opponent blocked our corners
            our_corners_before = len(game._valid_corners[our_player])
            our_corners_after = len(game_copy._valid_corners[our_player])
            corners_lost = max(0, our_corners_before - our_corners_after)

            # Calculate our mobility loss
            our_moves_before = len(game.get_legal_moves(our_player))
            our_moves_after = len(game_copy.get_legal_moves(our_player))
            mobility_loss = max(0, our_moves_before - our_moves_after)

            # Opponent value = their gain + damage to us
            opponent_value = base_score * 0.3 + corners_lost * 2.0 + mobility_loss * 0.1

            if remaining_depth > 0:
                # Recurse for deeper search (limited)
                # Skip to our next turn
                sim_game = game_copy.clone()
                while not sim_game.is_game_over and sim_game.current_player != our_player:
                    if not sim_game.get_legal_moves():
                        sim_game.pass_turn()
                    else:
                        # Other opponents play greedily
                        opp_moves = sim_game.get_legal_moves()
                        best_opp = max(opp_moves, key=lambda m: compute_move_heuristic_score(
                            m, sim_game, self._heuristic_config
                        ))
                        sim_game.make_move(best_opp)

                if not sim_game.is_game_over and sim_game.current_player == our_player:
                    # Evaluate our position
                    our_moves = sim_game.get_legal_moves()
                    if our_moves:
                        # Get our best response value
                        our_best = max(
                            compute_move_heuristic_score(m, sim_game, self._heuristic_config)
                            for m in our_moves[:10]  # Limit for performance
                        )
                        # Reduce opponent value if we have a good response
                        opponent_value -= our_best * 0.1

            best_opponent_value = max(best_opponent_value, opponent_value)

        return best_opponent_value

    def get_move_analysis(self, game: BlokusGame, top_n: int = 5) -> List[dict]:
        """
        Analyze legal moves and return detailed scoring breakdown.

        Useful for debugging and understanding move selection.

        Args:
            game: Current game state
            top_n: Number of top moves to analyze

        Returns:
            List of move analysis dictionaries
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return []

        analyses = []
        for move in legal_moves:
            piece = PIECE_BY_INDEX[move.piece_idx]

            # Basic score
            basic_score = compute_move_heuristic_score(move, game, self._heuristic_config)

            # Deep features (computed separately for analysis)
            mobility = count_moves_after(game, move) if self.config.use_mobility_lookahead else 0
            corner_diff = corner_differential(game, move, move.player) if self.config.use_corner_differential else 0

            analyses.append({
                'move': move,
                'piece_id': piece.id,
                'piece_size': piece.size,
                'position': (move.row, move.col),
                'orientation': move.orientation_idx,
                'basic_score': basic_score,
                'mobility': mobility,
                'corner_diff': corner_diff,
                'total_score': basic_score + mobility * 0.04 + corner_diff * 0.5,
            })

        # Sort by total score
        analyses.sort(key=lambda x: x['total_score'], reverse=True)
        return analyses[:top_n]


def benchmark_smart_player():
    """Benchmark SmartPlayer performance."""
    import random

    print("SmartPlayer Benchmark")
    print("=" * 50)

    configs = [
        ("Easy (depth=0)", SmartPlayerConfig.easy()),
        ("Medium (depth=1)", SmartPlayerConfig.medium()),
        ("Hard (depth=1)", SmartPlayerConfig.hard()),
    ]

    for name, config in configs:
        print(f"\n{name}:")
        player = SmartPlayer(config)

        times = []
        for _ in range(5):
            game = BlokusGame()
            # Play a few random moves to get to mid-game
            for _ in range(8):
                moves = game.get_legal_moves()
                if moves:
                    game.make_move(random.choice(moves))

            # Measure time to select move
            move = player.select_move(game)
            times.append(player.last_move_time_ms)

        avg_time = sum(times) / len(times)
        print(f"  Avg time: {avg_time:.1f}ms")
        print(f"  Candidates: {player.last_candidates_evaluated}")


def play_match(
    player1: SmartPlayer,
    player2: SmartPlayer,
    player1_name: str = "Player1",
    player2_name: str = "Player2",
    num_games: int = 10,
    verbose: bool = False,
) -> dict:
    """
    Play a match between two SmartPlayers (true 1v1).

    Each player controls 2 positions (teams):
    - Player1: positions 0 and 2
    - Player2: positions 1 and 3

    Players swap starting positions each game for fairness.
    Returns win statistics based on combined team scores.
    """
    results = {player1_name: 0, player2_name: 0, 'ties': 0}

    for game_idx in range(num_games):
        game = BlokusGame()

        # Alternate which player gets positions 0,2 vs 1,3
        if game_idx % 2 == 0:
            # Player1 at 0,2 - Player2 at 1,3
            players = [player1, player2, player1, player2]
            player_names = [player1_name, player2_name, player1_name, player2_name]
            p1_positions = [0, 2]
            p2_positions = [1, 3]
        else:
            # Player2 at 0,2 - Player1 at 1,3
            players = [player2, player1, player2, player1]
            player_names = [player2_name, player1_name, player2_name, player1_name]
            p1_positions = [1, 3]
            p2_positions = [0, 2]

        if verbose:
            log.info(f"=== Game {game_idx + 1} === Positions: {player_names}")

        while not game.is_game_over:
            current = game.current_player
            player = players[current]

            if isinstance(player, SmartPlayer):
                move = player.select_move(game, verbose=verbose)
            else:
                move = player.select_move(game)

            if move:
                game.make_move(move)
            else:
                game.pass_turn()

        # Determine winner by team score (combined scores of both positions)
        scores = game.get_scores()
        p1_total = sum(scores[i] for i in p1_positions)
        p2_total = sum(scores[i] for i in p2_positions)

        if p1_total > p2_total:
            winner_name = player1_name
            results[player1_name] += 1
        elif p2_total > p1_total:
            winner_name = player2_name
            results[player2_name] += 1
        else:
            winner_name = "Tie"
            results['ties'] += 1

        # Format scores with player names and team totals
        score_str = ", ".join(f"{player_names[i]}={scores[i]}" for i in range(4))
        print(f"Game {game_idx + 1}: Winner={winner_name}, Scores=[{score_str}], Teams: {player1_name}={p1_total}, {player2_name}={p2_total}")

    return results


if __name__ == "__main__":
    import sys
    import time
    from .heuristic_player import HeuristicPlayer

    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    print("=" * 60)
    print("=== Testing Heuristic Improvements (True 1v1) ===")
    print(f"=== Running {num_games} games per match ===")
    print("=== Each player controls 2 positions (team game) ===")
    print("=" * 60)

    # Timing debug for a single game
    print("\n=== Debug: Timing Analysis for 1 Game ===")
    from .game import BlokusGame
    from .pieces import PIECE_BY_INDEX

    hard = SmartPlayer(SmartPlayerConfig.hard())
    easy = SmartPlayer(SmartPlayerConfig.easy())

    game = BlokusGame()
    players = [hard, easy, hard, easy]
    player_names = ["Hard", "Easy", "Hard", "Easy"]

    move_times = {0: [], 1: [], 2: [], 3: []}
    total_start = time.time()

    while not game.is_game_over:
        current = game.current_player
        player = players[current]

        start = time.time()
        move = player.select_move(game, verbose=False)
        elapsed = (time.time() - start) * 1000

        move_times[current].append(elapsed)

        if move:
            game.make_move(move)
        else:
            game.pass_turn()

    total_elapsed = time.time() - total_start

    print(f"\nTotal game time: {total_elapsed:.2f}s")
    print(f"Total moves: {sum(len(t) for t in move_times.values())}")
    print("\nPer-position timing:")
    for pos in range(4):
        times = move_times[pos]
        if times:
            print(f"  P{pos} ({player_names[pos]}): {len(times)} moves, "
                  f"avg={sum(times)/len(times):.1f}ms, "
                  f"max={max(times):.1f}ms, "
                  f"total={sum(times):.0f}ms")

    print(f"\nHard total: {sum(move_times[0]) + sum(move_times[2]):.0f}ms")
    print(f"Easy total: {sum(move_times[1]) + sum(move_times[3]):.0f}ms")

    print("\n" + "=" * 60)
    print("=== Now running full test suite ===")
    print("=" * 60)

    easy = SmartPlayer(SmartPlayerConfig.easy())
    hard = SmartPlayer(SmartPlayerConfig.hard())
    filler = HeuristicPlayer()

    # Wrap filler in a simple adapter so play_match works
    class FillerAdapter:
        def __init__(self, player):
            self._player = player
        def select_move(self, game, verbose=False):
            return self._player.select_move(game)

    filler_adapter = FillerAdapter(filler)

    print("\nTest 1: Hard vs Easy")
    print("-" * 40)
    results1 = play_match(hard, easy, "Hard", "Easy", num_games=num_games, verbose=False)
    print(f"Results: Hard={results1['Hard']}, Easy={results1['Easy']}, Ties={results1['ties']}")

    print("\n" + "=" * 60)
    print("\nTest 2: Hard vs Filler")
    print("-" * 40)
    results2 = play_match(hard, filler_adapter, "Hard", "Filler", num_games=num_games, verbose=False)
    print(f"Results: Hard={results2['Hard']}, Filler={results2['Filler']}, Ties={results2['ties']}")

    print("\n" + "=" * 60)
    print("\nTest 3: Easy vs Filler")
    print("-" * 40)
    results3 = play_match(easy, filler_adapter, "Easy", "Filler", num_games=num_games, verbose=False)
    print(f"Results: Easy={results3['Easy']}, Filler={results3['Filler']}, Ties={results3['ties']}")

    print("\n" + "=" * 60)
    print("=== Summary ===")
    print(f"Test 1 - Hard vs Easy:   Hard={results1['Hard']}, Easy={results1['Easy']}, Ties={results1['ties']}")
    print(f"Test 2 - Hard vs Filler: Hard={results2['Hard']}, Filler={results2['Filler']}, Ties={results2['ties']}")
    print(f"Test 3 - Easy vs Filler: Easy={results3['Easy']}, Filler={results3['Filler']}, Ties={results3['ties']}")
