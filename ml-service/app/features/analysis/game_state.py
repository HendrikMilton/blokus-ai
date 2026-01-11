"""
Blokus game state representation.
"""

from dataclasses import dataclass, field

import numpy as np

from app.schemas import BOARD_SIZE, BoardCell, PlayerColor

from .pieces import PIECE_SIZES, PIECES

# Starting corners for each player
STARTING_CORNERS = {
    PlayerColor.BLUE: (0, 0),
    PlayerColor.YELLOW: (0, BOARD_SIZE - 1),
    PlayerColor.RED: (BOARD_SIZE - 1, BOARD_SIZE - 1),
    PlayerColor.GREEN: (BOARD_SIZE - 1, 0),
}

# Turn order
TURN_ORDER = [PlayerColor.BLUE, PlayerColor.YELLOW, PlayerColor.RED, PlayerColor.GREEN]


@dataclass
class Move:
    """Represents a move: placing a piece at a position."""
    piece_name: str
    cells: tuple[tuple[int, int], ...]  # Oriented piece cells
    row: int  # Position on board
    col: int
    rotation: int = 0  # Degrees (0, 90, 180, 270)

    def get_board_cells(self) -> list[tuple[int, int]]:
        """Get actual board positions for this move."""
        return [(self.row + r, self.col + c) for r, c in self.cells]


@dataclass
class GameState:
    """Represents the full game state."""
    board: np.ndarray  # BOARD_SIZE x BOARD_SIZE, values are PlayerColor or None
    pieces_remaining: dict[PlayerColor, set[str]]
    current_player: PlayerColor
    passed_players: set[PlayerColor] = field(default_factory=set)

    @classmethod
    def new_game(cls) -> "GameState":
        """Create a new game state."""
        board = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=object)
        board.fill(None)

        pieces_remaining = {
            color: set(PIECES.keys())
            for color in PlayerColor
        }

        return cls(
            board=board,
            pieces_remaining=pieces_remaining,
            current_player=PlayerColor.BLUE,
        )

    @classmethod
    def from_board_cells(cls, board_cells: list[list[BoardCell]], player: PlayerColor) -> "GameState":
        """Create game state from detected board and specified player."""
        board = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=object)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = board_cells[row][col]
                board[row, col] = cell.color

        # Infer pieces remaining (approximate - we can't know exactly which pieces were used)
        # For now, count cells per color and estimate pieces used
        pieces_remaining = {}
        for color in PlayerColor:
            color_cells = np.sum(board == color)
            # Rough estimate: remove largest pieces first based on cell count
            remaining = set(PIECES.keys())
            cells_to_remove = color_cells

            for piece_name in sorted(PIECES.keys(), key=lambda p: -PIECE_SIZES[p]):
                if cells_to_remove >= PIECE_SIZES[piece_name]:
                    remaining.discard(piece_name)
                    cells_to_remove -= PIECE_SIZES[piece_name]
                if cells_to_remove <= 0:
                    break

            pieces_remaining[color] = remaining

        return cls(
            board=board,
            pieces_remaining=pieces_remaining,
            current_player=player,
        )

    def copy(self) -> "GameState":
        """Create a deep copy of the game state."""
        return GameState(
            board=self.board.copy(),
            pieces_remaining={c: s.copy() for c, s in self.pieces_remaining.items()},
            current_player=self.current_player,
            passed_players=self.passed_players.copy(),
        )

    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def is_cell_empty(self, row: int, col: int) -> bool:
        """Check if a cell is empty."""
        return self.board[row, col] is None

    def has_corner_touch(self, row: int, col: int, color: PlayerColor) -> bool:
        """Check if cell has diagonal touch with same color."""
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_valid_position(nr, nc) and self.board[nr, nc] == color:
                return True
        return False

    def has_edge_touch(self, row: int, col: int, color: PlayerColor) -> bool:
        """Check if cell has edge touch with same color (not allowed)."""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_valid_position(nr, nc) and self.board[nr, nc] == color:
                return True
        return False

    def is_starting_corner(self, row: int, col: int, color: PlayerColor) -> bool:
        """Check if position is the starting corner for this color."""
        return (row, col) == STARTING_CORNERS[color]

    def has_played_first_piece(self, color: PlayerColor) -> bool:
        """Check if player has placed their first piece."""
        return len(self.pieces_remaining[color]) < 21

    def is_valid_move(self, move: Move, color: PlayerColor) -> bool:
        """Check if a move is valid for the given color."""
        board_cells = move.get_board_cells()

        # Check all cells are in bounds and empty
        for row, col in board_cells:
            if not self.is_valid_position(row, col):
                return False
            if not self.is_cell_empty(row, col):
                return False

        # Check no edge touch with same color
        for row, col in board_cells:
            if self.has_edge_touch(row, col, color):
                return False

        # First piece must touch starting corner
        if not self.has_played_first_piece(color):
            start_row, start_col = STARTING_CORNERS[color]
            if not any(r == start_row and c == start_col for r, c in board_cells):
                return False
        else:
            # Subsequent pieces must have corner touch
            has_corner = any(
                self.has_corner_touch(row, col, color)
                for row, col in board_cells
            )
            if not has_corner:
                return False

        return True

    def apply_move(self, move: Move, color: PlayerColor) -> "GameState":
        """Apply a move and return new state."""
        new_state = self.copy()

        # Place piece on board
        for row, col in move.get_board_cells():
            new_state.board[row, col] = color

        # Remove piece from remaining
        new_state.pieces_remaining[color].discard(move.piece_name)

        # Advance to next player
        new_state.current_player = self._next_player(color)

        return new_state

    def _next_player(self, current: PlayerColor) -> PlayerColor:
        """Get next player in turn order."""
        idx = TURN_ORDER.index(current)
        return TURN_ORDER[(idx + 1) % 4]

    def pass_turn(self) -> "GameState":
        """Current player passes (no valid moves)."""
        new_state = self.copy()
        new_state.passed_players.add(self.current_player)
        new_state.current_player = self._next_player(self.current_player)
        return new_state

    def is_terminal(self) -> bool:
        """Check if game is over (all players passed)."""
        return len(self.passed_players) >= 4

    def get_score(self, color: PlayerColor) -> int:
        """Get score for a player (cells placed on board)."""
        return int(np.sum(self.board == color))

    def get_valid_corners(self, color: PlayerColor) -> list[tuple[int, int]]:
        """Get all valid corner positions for placing new pieces."""
        corners = set()

        if not self.has_played_first_piece(color):
            # Only starting corner is valid
            return [STARTING_CORNERS[color]]

        # Find all diagonal neighbors of existing pieces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row, col] == color:
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = row + dr, col + dc
                        if (self.is_valid_position(nr, nc) and
                            self.is_cell_empty(nr, nc) and
                            not self.has_edge_touch(nr, nc, color)):
                            corners.add((nr, nc))

        return list(corners)
