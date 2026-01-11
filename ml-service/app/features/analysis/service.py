"""
Analysis service using MCTS for move recommendations.
"""

import time

from app.lib import FeatureMetrics, get_logger
from app.schemas import AnalysisStats, AnalyzeBoardResponse, BoardCell, PieceShape, PlayerColor

from .game_state import GameState
from .mcts import MCTS

metrics = FeatureMetrics()


class AnalysisService:
    """Move analysis using MCTS."""

    def __init__(self):
        # MCTS with reasonable defaults for responsive gameplay
        self.mcts = MCTS(iterations=1000, time_limit_ms=3000, exploration_constant=1.414)

    def analyze(self, board: list[list[BoardCell]], color: PlayerColor, cid: str) -> AnalyzeBoardResponse:
        log = get_logger(cid)
        start = time.perf_counter()

        # Convert board cells to game state
        game_state = GameState.from_board_cells(board, color)

        # Run MCTS to find best move
        best_move = self.mcts.search(game_state, color)

        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record(latency_ms)

        if best_move is None:
            log.info("analysis_complete", latency_ms=round(latency_ms, 2), result="no_valid_moves")
            # No valid moves - return a placeholder indicating player should pass
            return AnalyzeBoardResponse(
                recommendedPiece=None,
                placement=None,
                stats=self._compute_stats(game_state, color),
            )

        # Convert move to API response format
        recommended_piece = PieceShape(
            name=best_move.piece_name,
            cells=list(best_move.cells),
            rotation=best_move.rotation,
        )

        placement = {
            "row": best_move.row,
            "col": best_move.col,
        }

        stats = self._compute_stats(game_state, color)

        log.info(
            "analysis_complete",
            latency_ms=round(latency_ms, 2),
            piece=best_move.piece_name,
            placement=f"({best_move.row}, {best_move.col})",
        )

        return AnalyzeBoardResponse(
            recommendedPiece=recommended_piece,
            placement=placement,
            stats=stats,
        )

    def _compute_stats(self, state: GameState, color: PlayerColor) -> AnalysisStats:
        """Compute game statistics for the player."""
        remaining_pieces = state.pieces_remaining[color]
        pieces_remaining = len(remaining_pieces)
        pieces_used = 21 - pieces_remaining

        # Calculate total cells placed (score)
        score = state.get_score(color)

        # Rough win probability based on score differential and remaining pieces
        opponent_scores = [
            state.get_score(c) for c in PlayerColor if c != color
        ]
        avg_opponent_score = sum(opponent_scores) / len(opponent_scores) if opponent_scores else 0

        score_diff = score - avg_opponent_score
        # Normalize to a probability (very rough estimate)
        base_probability = 25.0  # 25% base for 4-player game
        win_probability = base_probability + (score_diff * 2.0)
        win_probability = max(5.0, min(95.0, win_probability))  # Clamp to 5-95%

        return AnalysisStats(
            piecesUsed=pieces_used,
            piecesRemaining=pieces_remaining,
            winProbability=round(win_probability, 1),
        )
