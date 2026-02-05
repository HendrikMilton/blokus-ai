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
from .game import BlokusGame, Move, BOARD_SIZE
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

    # Tiebreaker settings (for when piece sizes are equal)
    use_strategic_tiebreaker: bool = False  # Enable advanced tiebreaking
    tiebreaker_top_n: int = 5              # Apply tiebreaker to top N same-size moves

    # Position lookahead (Expert only)
    use_position_lookahead: bool = False   # Evaluate position quality after move

    # Opponent tracking (Expert only)
    use_opponent_large_piece_blocking: bool = False  # Block spots where opponents could place large pieces

    # Team mode (Expert only) - for 2v2 games
    # In team mode: players 0 & 2 are a team, players 1 & 3 are a team
    # None = free-for-all (treat all other players as opponents)
    # True = team mode (don't block teammate, help teammate expand)
    team_mode: Optional[bool] = None

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
        """Hard difficulty - greedy with improved heuristics and tiebreaker.

        Uses strategic tiebreaker to pick best move among same-size pieces.
        This differentiates Hard from Easy/Filler without costly lookahead.
        """
        return cls(
            lookahead_depth=0,
            candidate_count=20,
            use_mobility_lookahead=False,
            use_corner_differential=False,
            use_opponent_blocking=True,
            use_strategic_tiebreaker=True,  # Key differentiator from Easy
            tiebreaker_top_n=8,
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
    def expert(cls, team_mode: Optional[bool] = None) -> 'SmartPlayerConfig':
        """Expert difficulty - Hard + opponent large piece blocking.

        Uses same tiebreaker as Hard but also considers blocking spots
        where opponents could place their remaining large pieces.

        Args:
            team_mode: None = free-for-all, True = 2v2 team mode
                       In team mode, won't block teammate and will help them expand.
        """
        return cls(
            lookahead_depth=0,
            candidate_count=30,
            use_mobility_lookahead=False,
            use_corner_differential=False,
            use_opponent_blocking=True,
            use_strategic_tiebreaker=True,
            tiebreaker_top_n=8,
            use_position_lookahead=False,
            use_opponent_large_piece_blocking=True,  # Expert-only feature
            team_mode=team_mode,
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

    def _get_teammate(self, player: int) -> Optional[int]:
        """Get teammate for a player in team mode.

        In 2v2: players 0 & 2 are a team, players 1 & 3 are a team.
        Returns None if not in team mode.
        """
        if not self.config.team_mode:
            return None
        return (player + 2) % 4

    def _is_opponent(self, player: int, other: int) -> bool:
        """Check if other player is an opponent (not self or teammate)."""
        if other == player:
            return False
        if self.config.team_mode:
            return other != self._get_teammate(player)
        return True

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

        # Sort by score descending
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        # Apply strategic tiebreaker if enabled
        if self.config.use_strategic_tiebreaker and len(scored_moves) > 1:
            scored_moves = self._apply_tiebreaker(game, scored_moves, ctx)

        n = min(self.config.candidate_count, len(scored_moves))
        return scored_moves[:n]

    def _apply_tiebreaker(
        self,
        game: BlokusGame,
        scored_moves: List[Tuple[float, Move]],
        ctx: ScoringContext,
    ) -> List[Tuple[float, Move]]:
        """
        Apply strategic tiebreaker to moves with the same piece size.

        When multiple moves use pieces of the same size, use secondary
        criteria to pick the best position:
        1. Corners created (future flexibility)
        2. Blocking opponent corners
        3. Distance toward center (control)
        """
        if not scored_moves:
            return scored_moves

        # Group top moves by piece size
        top_size = PIECE_BY_INDEX[scored_moves[0][1].piece_idx].size
        same_size_moves = []
        other_moves = []

        for score, move in scored_moves:
            piece = PIECE_BY_INDEX[move.piece_idx]
            if piece.size == top_size:
                same_size_moves.append((score, move))
            else:
                other_moves.append((score, move))

        # Only apply tiebreaker if there are multiple same-size moves
        if len(same_size_moves) <= 1:
            return scored_moves

        # Limit tiebreaker evaluation to top N for performance
        n = min(self.config.tiebreaker_top_n, len(same_size_moves))
        candidates = same_size_moves[:n]
        rest = same_size_moves[n:]

        # Calculate tiebreaker scores
        tiebroken = []
        for base_score, move in candidates:
            tb_score = self._compute_tiebreaker_score(game, move, ctx)
            # Add small tiebreaker score (won't override piece size)
            tiebroken.append((base_score + tb_score * 0.2, move))

        # Sort by new score
        tiebroken.sort(key=lambda x: x[0], reverse=True)

        # Reconstruct full list
        return tiebroken + rest + other_moves

    def _compute_tiebreaker_score(
        self,
        game: BlokusGame,
        move: Move,
        ctx: ScoringContext,
    ) -> float:
        """
        Compute tiebreaker score for a move.

        Higher = better position among same-size pieces.
        """
        from .heuristics import blocks_opponent_corners, DIAGONAL_OFFSETS, EDGE_OFFSETS

        piece = PIECE_BY_INDEX[move.piece_idx]
        cells = piece.get_cells(move.orientation_idx, move.row, move.col)
        cell_set = set(map(tuple, cells))
        player_color = move.player + 1

        score = 0.0

        # 1. Corners created (very important - future options)
        corners_created = 0
        corner_positions = []  # Track for Expert's advanced features
        for r, c in cells:
            for dr, dc in DIAGONAL_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if game.board[nr, nc] == 0:
                        is_valid = True
                        for edr, edc in EDGE_OFFSETS:
                            er, ec = nr + edr, nc + edc
                            if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
                                if game.board[er, ec] == player_color:
                                    is_valid = False
                                    break
                            if (er, ec) in cell_set:
                                is_valid = False
                                break
                        if is_valid:
                            corners_created += 1
                            corner_positions.append((nr, nc))
        score += corners_created * 3.0  # Weight corners heavily

        # 2. Blocking opponent corners
        blocking = blocks_opponent_corners(game, move, ctx)
        score += blocking * 2.0

        # 3. Prefer moves toward center (more flexibility)
        avg_row = cells[:, 0].mean()
        avg_col = cells[:, 1].mean()
        center_dist = np.sqrt((avg_row - BOARD_SIZE/2)**2 + (avg_col - BOARD_SIZE/2)**2)
        max_dist = np.sqrt(2) * BOARD_SIZE / 2
        center_score = (1.0 - center_dist / max_dist) * 2.0
        score += center_score

        # 4. Avoid edges (less trapped)
        edge_cells = sum(1 for r, c in cells
                        if r <= 1 or r >= BOARD_SIZE-2 or c <= 1 or c >= BOARD_SIZE-2)
        score -= edge_cells * 0.5

        # === Expert-only features (use_opponent_large_piece_blocking) ===
        if self.config.use_opponent_large_piece_blocking:
            # 5. Breakout potential - corners in multiple quadrants
            if corner_positions:
                quadrants_reached = set()
                for r, c in corner_positions:
                    q = (0 if r < BOARD_SIZE//2 else 1, 0 if c < BOARD_SIZE//2 else 1)
                    quadrants_reached.add(q)
                score += len(quadrants_reached) * 1.5

            # 6. Expansion direction - prefer corners closer to center
            if corner_positions:
                best_corner_centrality = 0.0
                for r, c in corner_positions:
                    dist = np.sqrt((r - BOARD_SIZE/2)**2 + (c - BOARD_SIZE/2)**2)
                    centrality = 1.0 - dist / max_dist
                    best_corner_centrality = max(best_corner_centrality, centrality)
                score += best_corner_centrality * 2.0

            # 7. Block opponent large pieces (skip teammate in team mode)
            large_piece_block_score = self._score_large_piece_blocking(game, move, cell_set)
            score += large_piece_block_score

            # 8. Team mode: bonus for helping teammate expand
            if self.config.team_mode:
                teammate_help_score = self._score_teammate_help(game, move, cell_set, corner_positions)
                score += teammate_help_score

        return score

    def _score_large_piece_blocking(
        self,
        game: BlokusGame,
        move: Move,
        cell_set: set,
    ) -> float:
        """
        Score bonus for blocking spots where opponents could place large pieces.

        If an opponent still has 5-cell pieces and our move blocks a spot
        where they could have placed one, that's valuable.

        In team mode, skips teammate (don't penalize blocking teammate).
        """
        score = 0.0

        for opp in range(4):
            # Skip self
            if opp == move.player:
                continue

            # Skip teammate in team mode
            if not self._is_opponent(move.player, opp):
                continue

            opp_state = game.players[opp]
            if opp_state.is_eliminated:
                continue

            # Check if opponent still has large pieces (size 4-5)
            large_pieces_remaining = sum(
                1 for pid in opp_state.remaining_pieces
                if PIECE_BY_INDEX[pid].size >= 4
            )

            if large_pieces_remaining == 0:
                continue

            # Check if our move blocks any of opponent's corners
            # (making it harder for them to place large pieces)
            opp_corners = game._valid_corners[opp]
            blocked_corners = 0

            for corner in opp_corners:
                # Direct occupation
                if corner in cell_set:
                    blocked_corners += 1
                    continue

                # Edge-adjacent blocking
                for r, c in cell_set:
                    if abs(corner[0] - r) + abs(corner[1] - c) == 1:
                        blocked_corners += 0.5
                        break

            # More valuable to block opponents with more large pieces
            score += blocked_corners * (large_pieces_remaining / 10.0)

        return score

    def _score_teammate_help(
        self,
        game: BlokusGame,
        move: Move,
        cell_set: set,
        corner_positions: List[Tuple[int, int]],
    ) -> float:
        """
        Score bonus for moves that help teammate in team mode.

        Rewards:
        - Not blocking teammate's corners (penalty avoided)
        - Creating corners near teammate's corners (expansion synergy)
        - Blocking opponents that are threatening teammate
        """
        if not self.config.team_mode:
            return 0.0

        teammate = self._get_teammate(move.player)
        if teammate is None:
            return 0.0

        teammate_state = game.players[teammate]
        if teammate_state.is_eliminated:
            return 0.0

        score = 0.0

        # 1. Penalty for blocking teammate's corners
        teammate_corners = game._valid_corners[teammate]
        blocked_teammate = 0
        for corner in teammate_corners:
            if corner in cell_set:
                blocked_teammate += 2.0  # Direct block is bad
            else:
                for r, c in cell_set:
                    if abs(corner[0] - r) + abs(corner[1] - c) == 1:
                        blocked_teammate += 1.0  # Edge-adjacent is also bad
                        break
        score -= blocked_teammate * 1.5  # Penalty for blocking teammate

        # 2. Bonus for creating corners near teammate's territory
        # (allows teammate to potentially connect/expand)
        if corner_positions and teammate_corners:
            teammate_color = teammate + 1
            # Check if any new corners are near teammate's existing pieces
            for cr, cc in corner_positions:
                # Check if corner is diagonally adjacent to teammate's piece
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if game.board[nr, nc] == teammate_color:
                            score += 0.5  # Creates synergy with teammate

        # 3. Bonus for blocking opponents that are near teammate
        # (defensive help)
        for opp in range(4):
            if not self._is_opponent(move.player, opp):
                continue

            opp_corners = game._valid_corners[opp]
            # Check if opponent corners are near teammate's corners (threat)
            for opp_corner in opp_corners:
                for tm_corner in teammate_corners:
                    dist = abs(opp_corner[0] - tm_corner[0]) + abs(opp_corner[1] - tm_corner[1])
                    if dist <= 3:  # Opponent is threatening teammate
                        # Check if our move blocks this opponent corner
                        if opp_corner in cell_set:
                            score += 1.0  # Defended teammate
                        else:
                            for r, c in cell_set:
                                if abs(opp_corner[0] - r) + abs(opp_corner[1] - c) == 1:
                                    score += 0.5
                                    break

        return score

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

        # Position lookahead (Expert only): evaluate position quality after move
        # Use very small weight (0.001) so it only acts as tiebreaker
        if self.config.use_position_lookahead:
            position_score = self._evaluate_position(game_copy, move.player)
            score += position_score * 0.001  # Tiny weight - tiebreaker only

        # Evaluate opponent's best response
        opponent_value = self._simulate_opponent_turn(game_copy, move.player, depth - 1)

        # Discount opponent's gains
        score -= opponent_value * 0.3

        return score

    def _evaluate_position(self, game: BlokusGame, player: int) -> float:
        """
        Evaluate position quality for Expert lookahead.

        Considers:
        1. Corner advantage (my corners vs opponents)
        2. Board control (cells in each quadrant)
        3. Opponent blocking potential
        """
        score = 0.0

        # 1. Corner advantage: more corners = more options
        my_corners = len(game._valid_corners[player])
        opp_corners = sum(len(game._valid_corners[p]) for p in range(4) if p != player)
        avg_opp_corners = opp_corners / 3.0
        corner_advantage = my_corners - avg_opp_corners
        score += corner_advantage * 2.0

        # 2. Board control: presence in multiple quadrants
        player_color = player + 1
        quadrant_presence = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r, c] == player_color:
                    q = (0 if r < 10 else 1, 0 if c < 10 else 1)
                    quadrant_presence[q] += 1

        # Bonus for presence in multiple quadrants
        quadrants_with_presence = sum(1 for v in quadrant_presence.values() if v > 0)
        score += quadrants_with_presence * 1.5

        # 3. Centrality of corners (corners closer to center = better)
        if my_corners > 0:
            corner_centrality = 0.0
            for corner in game._valid_corners[player]:
                r, c = corner
                dist = np.sqrt((r - BOARD_SIZE/2)**2 + (c - BOARD_SIZE/2)**2)
                max_dist = np.sqrt(2) * BOARD_SIZE / 2
                corner_centrality += (1.0 - dist / max_dist)
            score += corner_centrality / my_corners * 2.0

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

    # === Old tests (commented out) ===
    # print("\n" + "=" * 60)
    # print("=== Debug: Comparing Expert vs Easy move selection ===")
    # print("=" * 60)
    #
    # expert = SmartPlayer(SmartPlayerConfig.expert())
    # easy = SmartPlayer(SmartPlayerConfig.easy())
    # filler = HeuristicPlayer()
    #
    # # Wrap filler in a simple adapter so play_match works
    # class FillerAdapter:
    #     def __init__(self, player):
    #         self._player = player
    #     def select_move(self, game, verbose=False):
    #         return self._player.select_move(game)
    #
    # filler_adapter = FillerAdapter(filler)
    #
    # # Compare what Expert and Easy would pick in same positions
    # game = BlokusGame()
    #
    # print("\nComparing move selection at each turn (first 16 moves):\n")
    #
    # for turn in range(16):
    #     current = game.current_player
    #
    #     # Get what both would pick
    #     expert_move = expert.select_move(game, verbose=False)
    #     easy_move = easy.select_move(game, verbose=False)
    #
    #     if expert_move and easy_move:
    #         ep = PIECE_BY_INDEX[expert_move.piece_idx]
    #         eap = PIECE_BY_INDEX[easy_move.piece_idx]
    #
    #         same = "SAME" if (expert_move.piece_idx == easy_move.piece_idx and
    #                          expert_move.row == easy_move.row and
    #                          expert_move.col == easy_move.col) else "DIFF"
    #
    #         print(f"T{game.turn_number:2d} P{current}: Expert={ep.id:3s} at ({expert_move.row:2d},{expert_move.col:2d}), "
    #               f"Easy={eap.id:3s} at ({easy_move.row:2d},{easy_move.col:2d}) [{same}]")
    #
    #         # Play Expert's move to continue
    #         game.make_move(expert_move)
    #     else:
    #         print(f"T{game.turn_number:2d} P{current}: No legal moves")
    #         game.pass_turn()
    #
    # print("\n" + "=" * 60)
    # print("=== Tests: 8 games each ===")
    # print("=" * 60)
    #
    # hard = SmartPlayer(SmartPlayerConfig.hard())
    #
    # print("\nExpert vs Easy:")
    # results1 = play_match(expert, easy, "Expert", "Easy", num_games=8, verbose=False)
    # print(f"Results: Expert={results1['Expert']}, Easy={results1['Easy']}")
    #
    # print("\nExpert vs Hard:")
    # results2 = play_match(expert, hard, "Expert", "Hard", num_games=8, verbose=False)
    # print(f"Results: Expert={results2['Expert']}, Hard={results2['Hard']}")
    #
    # print("\nHard vs Easy:")
    # results3 = play_match(hard, easy, "Hard", "Easy", num_games=8, verbose=False)
    # print(f"Results: Hard={results3['Hard']}, Easy={results3['Easy']}")
    #
    # print("\nExpert vs Filler:")
    # results4 = play_match(expert, filler_adapter, "Expert", "Filler", num_games=8, verbose=False)
    # print(f"Results: Expert={results4['Expert']}, Filler={results4['Filler']}")
    #
    # print("\nHard vs Filler:")
    # results5 = play_match(hard, filler_adapter, "Hard", "Filler", num_games=8, verbose=False)
    # print(f"Results: Hard={results5['Hard']}, Filler={results5['Filler']}")
    #
    # print("\n" + "=" * 60)
    # print("=== Summary ===")
    # print(f"Expert vs Easy:   Expert={results1['Expert']}, Easy={results1['Easy']}")
    # print(f"Expert vs Hard:   Expert={results2['Expert']}, Hard={results2['Hard']}")
    # print(f"Hard vs Easy:     Hard={results3['Hard']}, Easy={results3['Easy']}")
    # print(f"Expert vs Filler: Expert={results4['Expert']}, Filler={results4['Filler']}")
    # print(f"Hard vs Filler:   Hard={results5['Hard']}, Filler={results5['Filler']}")

    # =========================================================================
    # === NEW TESTS: Team Mode vs Free-For-All ===
    # =========================================================================

    print("\n" + "=" * 60)
    print("=== Team Mode Tests ===")
    print("=== Expert(team) vs Expert(ffa) vs Hard vs Easy ===")
    print("=" * 60)

    from .game import BlokusGame
    from .pieces import PIECE_BY_INDEX

    # Create players
    expert_team = SmartPlayer(SmartPlayerConfig.expert(team_mode=True))
    expert_ffa = SmartPlayer(SmartPlayerConfig.expert(team_mode=False))
    hard = SmartPlayer(SmartPlayerConfig.hard())
    easy = SmartPlayer(SmartPlayerConfig.easy())

    def play_4way_match(
        players: List[SmartPlayer],
        names: List[str],
        num_games: int = 8,
    ) -> dict:
        """
        Play a 4-way match where each player controls one position.

        Unlike play_match (2v2 team format), this is true free-for-all
        or can test team dynamics when team-aware players are used.
        """
        # Track individual wins (highest score wins)
        wins = {name: 0 for name in names}
        total_scores = {name: 0 for name in names}

        for game_idx in range(num_games):
            game = BlokusGame()

            # Rotate starting positions for fairness
            rotation = game_idx % 4
            rotated_players = players[rotation:] + players[:rotation]
            rotated_names = names[rotation:] + names[:rotation]

            while not game.is_game_over:
                current = game.current_player
                player = rotated_players[current]

                move = player.select_move(game, verbose=False)
                if move:
                    game.make_move(move)
                else:
                    game.pass_turn()

            scores = game.get_scores()
            score_str = ", ".join(f"{rotated_names[i]}={scores[i]}" for i in range(4))

            # Winner is player with highest score
            max_score = max(scores)
            winners = [rotated_names[i] for i in range(4) if scores[i] == max_score]

            for winner in winners:
                wins[winner] += 1.0 / len(winners)  # Split ties

            for i in range(4):
                total_scores[rotated_names[i]] += scores[i]

            print(f"Game {game_idx + 1}: Scores=[{score_str}], Winner={winners[0] if len(winners) == 1 else 'Tie'}")

        return {'wins': wins, 'avg_scores': {n: total_scores[n] / num_games for n in names}}

    # Test 1: Expert(team) vs Hard - 2v2 format
    print("\n" + "-" * 50)
    print("Test 1: Expert(team) vs Hard [2v2 format]")
    print("Expert(team) at positions 0,2 - Hard at positions 1,3")
    print("-" * 50)
    results1 = play_match(expert_team, hard, "Expert-Team", "Hard", num_games=num_games, verbose=False)
    print(f"Results: Expert-Team={results1['Expert-Team']}, Hard={results1['Hard']}")

    # Test 2: Expert(ffa) vs Hard - 2v2 format
    print("\n" + "-" * 50)
    print("Test 2: Expert(ffa) vs Hard [2v2 format]")
    print("Expert(ffa) at positions 0,2 - Hard at positions 1,3")
    print("-" * 50)
    results2 = play_match(expert_ffa, hard, "Expert-FFA", "Hard", num_games=num_games, verbose=False)
    print(f"Results: Expert-FFA={results2['Expert-FFA']}, Hard={results2['Hard']}")

    # Test 3: Expert(team) vs Expert(ffa) - 2v2 format
    print("\n" + "-" * 50)
    print("Test 3: Expert(team) vs Expert(ffa) [2v2 format]")
    print("Expert(team) at positions 0,2 - Expert(ffa) at positions 1,3")
    print("-" * 50)
    results3 = play_match(expert_team, expert_ffa, "Expert-Team", "Expert-FFA", num_games=num_games, verbose=False)
    print(f"Results: Expert-Team={results3['Expert-Team']}, Expert-FFA={results3['Expert-FFA']}")

    # Test 4: 4-way - Expert(team) vs Expert(ffa) vs Hard vs Easy
    print("\n" + "-" * 50)
    print("Test 4: 4-way free-for-all")
    print("P0=Expert-Team, P1=Expert-FFA, P2=Hard, P3=Easy")
    print("-" * 50)
    results4 = play_4way_match(
        [expert_team, expert_ffa, hard, easy],
        ["Expert-Team", "Expert-FFA", "Hard", "Easy"],
        num_games=num_games,
    )
    print(f"\nWins: {results4['wins']}")
    print(f"Avg scores: {results4['avg_scores']}")

    # Summary
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    print(f"Test 1 - Expert(team) vs Hard [2v2]:     Expert-Team={results1['Expert-Team']}, Hard={results1['Hard']}")
    print(f"Test 2 - Expert(ffa) vs Hard [2v2]:      Expert-FFA={results2['Expert-FFA']}, Hard={results2['Hard']}")
    print(f"Test 3 - Expert(team) vs Expert(ffa):    Expert-Team={results3['Expert-Team']}, Expert-FFA={results3['Expert-FFA']}")
    print(f"Test 4 - 4-way: {results4['wins']}")
