"""
Heuristic evaluation functions for Blokus game states.
"""


from app.schemas import BOARD_SIZE, PlayerColor

from .game_state import GameState, Move
from .pieces import PIECE_SIZES

# Heuristic weights (tunable)
WEIGHTS = {
    "corner_availability": 3.0,
    "piece_size_bonus": 2.0,
    "territorial_spread": 2.5,
    "opponent_blocking": 2.0,
    "center_control": 1.5,
    "pieces_remaining_penalty": -0.5,
}


def count_valid_corners(state: GameState, color: PlayerColor) -> int:
    """Count available corner positions for placing pieces."""
    return len(state.get_valid_corners(color))


def calculate_territorial_spread(state: GameState, color: PlayerColor) -> float:
    """
    Calculate how spread out the player's pieces are.
    Higher spread = more board control.
    """
    positions = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if state.board[row, col] == color:
                positions.append((row, col))

    if len(positions) < 2:
        return 0.0

    # Calculate bounding box area
    rows = [p[0] for p in positions]
    cols = [p[1] for p in positions]
    width = max(cols) - min(cols) + 1
    height = max(rows) - min(rows) + 1

    # Normalize by board size
    return (width * height) / (BOARD_SIZE * BOARD_SIZE) * 100


def calculate_center_control(state: GameState, color: PlayerColor) -> float:
    """
    Calculate control of center region (higher value = more central pieces).
    """
    center_start = BOARD_SIZE // 4
    center_end = BOARD_SIZE * 3 // 4
    center_cells = 0
    total_cells = 0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if state.board[row, col] == color:
                total_cells += 1
                if center_start <= row < center_end and center_start <= col < center_end:
                    center_cells += 1

    if total_cells == 0:
        return 0.0

    return (center_cells / total_cells) * 100


def count_blocked_opponent_corners(state: GameState, color: PlayerColor) -> int:
    """Count how many opponent corner positions we're blocking."""
    blocked = 0

    for opponent in PlayerColor:
        if opponent == color:
            continue

        opponent_corners = state.get_valid_corners(opponent)
        for corner_row, corner_col in opponent_corners:
            # Check if any of our pieces are adjacent (blocking expansion)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = corner_row + dr, corner_col + dc
                if state.is_valid_position(nr, nc) and state.board[nr, nc] == color:
                    blocked += 1
                    break

    return blocked


def evaluate_state(state: GameState, color: PlayerColor) -> float:
    """
    Evaluate game state for a player. Higher = better.
    """
    score = 0.0

    # Corner availability (more options = better)
    corners = count_valid_corners(state, color)
    score += corners * WEIGHTS["corner_availability"]

    # Piece size bonus (reward placing large pieces early)
    remaining = state.pieces_remaining[color]
    large_pieces = sum(1 for p in remaining if PIECE_SIZES[p] >= 4)
    small_pieces = sum(1 for p in remaining if PIECE_SIZES[p] <= 2)
    # Early game: penalize having large pieces remaining
    # Late game: reward having small pieces for flexibility
    total_remaining = len(remaining)
    if total_remaining > 15:  # Early game
        score -= large_pieces * WEIGHTS["piece_size_bonus"]
    else:  # Late game
        score += small_pieces * WEIGHTS["piece_size_bonus"]

    # Territorial spread
    spread = calculate_territorial_spread(state, color)
    score += spread * WEIGHTS["territorial_spread"] / 100

    # Center control
    center = calculate_center_control(state, color)
    score += center * WEIGHTS["center_control"] / 100

    # Opponent blocking
    blocked = count_blocked_opponent_corners(state, color)
    score += blocked * WEIGHTS["opponent_blocking"]

    # Pieces remaining penalty
    score += len(remaining) * WEIGHTS["pieces_remaining_penalty"]

    # Current score (cells on board)
    score += state.get_score(color) * 0.5

    return score


def evaluate_move(state: GameState, move: Move, color: PlayerColor) -> float:
    """
    Quick evaluation of a move without full state simulation.
    Used for move ordering in MCTS.
    """
    score = 0.0

    # Piece size (prefer larger pieces early)
    piece_size = len(move.cells)
    remaining = len(state.pieces_remaining[color])

    if remaining > 15:  # Early game
        score += piece_size * 2.0
    else:  # Late game - save large pieces is bad
        score += piece_size * 0.5

    # Position bonus (prefer moves toward center)
    board_cells = move.get_board_cells()
    center = BOARD_SIZE // 2
    avg_dist = sum(
        abs(r - center) + abs(c - center)
        for r, c in board_cells
    ) / len(board_cells)
    score += (BOARD_SIZE - avg_dist) * 0.3

    # Expansion bonus (prefer moves that create new corners)
    new_corners = 0
    for r, c in board_cells:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (state.is_valid_position(nr, nc) and
                state.is_cell_empty(nr, nc) and
                not state.has_edge_touch(nr, nc, color)):
                new_corners += 1
    score += new_corners * 1.5

    return score
