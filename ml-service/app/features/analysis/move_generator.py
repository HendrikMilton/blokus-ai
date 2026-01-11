"""
Generate all valid moves for a player.
"""

from app.schemas import PlayerColor

from .game_state import GameState, Move
from .pieces import PIECE_ORIENTATIONS, PIECES, get_rotation_degrees


def generate_moves(state: GameState, color: PlayerColor) -> list[Move]:
    """Generate all valid moves for a player."""
    moves = []
    remaining_pieces = state.pieces_remaining[color]
    valid_corners = state.get_valid_corners(color)

    if not valid_corners:
        return moves

    for piece_name in remaining_pieces:
        original_cells = PIECES[piece_name]
        orientations = PIECE_ORIENTATIONS[piece_name]

        for oriented_cells in orientations:
            rotation = get_rotation_degrees(original_cells, oriented_cells)

            # Try to place piece at each valid corner
            for corner_row, corner_col in valid_corners:
                # Try each cell of the piece at the corner
                for cell_idx, (cell_r, cell_c) in enumerate(oriented_cells):
                    # Position piece so this cell lands on the corner
                    place_row = corner_row - cell_r
                    place_col = corner_col - cell_c

                    move = Move(
                        piece_name=piece_name,
                        cells=oriented_cells,
                        row=place_row,
                        col=place_col,
                        rotation=rotation,
                    )

                    if state.is_valid_move(move, color):
                        moves.append(move)

    # Remove duplicates (same piece placement can be generated multiple ways)
    seen = set()
    unique_moves = []
    for move in moves:
        key = (move.piece_name, tuple(sorted(move.get_board_cells())))
        if key not in seen:
            seen.add(key)
            unique_moves.append(move)

    return unique_moves


def generate_moves_fast(state: GameState, color: PlayerColor, max_moves: int = 50) -> list[Move]:
    """
    Generate moves with early termination for performance.
    Prioritizes larger pieces and limits total moves.
    """
    moves = []
    remaining_pieces = state.pieces_remaining[color]
    valid_corners = state.get_valid_corners(color)

    if not valid_corners:
        return moves

    # Sort pieces by size (larger first) for better moves
    sorted_pieces = sorted(
        remaining_pieces,
        key=lambda p: -len(PIECES[p])
    )

    seen = set()

    for piece_name in sorted_pieces:
        if len(moves) >= max_moves:
            break

        original_cells = PIECES[piece_name]
        orientations = PIECE_ORIENTATIONS[piece_name]

        for oriented_cells in orientations:
            if len(moves) >= max_moves:
                break

            rotation = get_rotation_degrees(original_cells, oriented_cells)

            for corner_row, corner_col in valid_corners:
                if len(moves) >= max_moves:
                    break

                # Try first cell at corner (simplified)
                cell_r, cell_c = oriented_cells[0]
                place_row = corner_row - cell_r
                place_col = corner_col - cell_c

                move = Move(
                    piece_name=piece_name,
                    cells=oriented_cells,
                    row=place_row,
                    col=place_col,
                    rotation=rotation,
                )

                key = (move.piece_name, tuple(sorted(move.get_board_cells())))
                if key not in seen and state.is_valid_move(move, color):
                    seen.add(key)
                    moves.append(move)

    return moves
