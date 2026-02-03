"""
Native Python Blokus game implementation optimized for AlphaZero self-play.
Uses NumPy for fast board operations.
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np

from .pieces import (
    PIECE_CATALOG,
    PIECE_BY_INDEX,
    PIECE_INDEX_BY_ID,
    NUM_PIECES,
    PieceDefinition,
)

# Board constants
BOARD_SIZE = 20
NUM_PLAYERS = 4

# Player colors (indices)
BLUE = 0
YELLOW = 1
RED = 2
GREEN = 3

PLAYER_NAMES = ['blue', 'yellow', 'red', 'green']

# Starting corners for each player (row, col)
STARTING_CORNERS = {
    BLUE: (0, 0),         # Top-left
    YELLOW: (0, 19),      # Top-right
    RED: (19, 19),        # Bottom-right
    GREEN: (19, 0),       # Bottom-left
}

# Direction offsets for orthogonal neighbors (edge adjacency)
EDGE_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Direction offsets for diagonal neighbors (corner adjacency)
DIAGONAL_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class Move:
    """A move in Blokus."""
    player: int            # Player index (0-3)
    piece_idx: int         # Piece index (0-20)
    orientation_idx: int   # Orientation index
    row: int               # Position row
    col: int               # Position col

    def __hash__(self):
        return hash((self.player, self.piece_idx, self.orientation_idx, self.row, self.col))

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.player == other.player and
                self.piece_idx == other.piece_idx and
                self.orientation_idx == other.orientation_idx and
                self.row == other.row and
                self.col == other.col)


@dataclass
class PlayerState:
    """State for a single player."""
    remaining_pieces: Set[int] = field(default_factory=lambda: set(range(NUM_PIECES)))
    has_played: bool = False
    is_eliminated: bool = False
    score: int = 0

    def clone(self) -> 'PlayerState':
        return PlayerState(
            remaining_pieces=self.remaining_pieces.copy(),
            has_played=self.has_played,
            is_eliminated=self.is_eliminated,
            score=self.score,
        )


class BlokusGame:
    """
    Native Python Blokus game optimized for self-play.

    Board representation:
    - board: numpy array of shape (20, 20), dtype int8
    - Values: 0 = empty, 1-4 = player color

    For fast legal move generation, we track:
    - valid_corners: set of (row, col) positions that are valid attachment points for each player
    """

    def __init__(self):
        # Board: 0 = empty, 1 = blue, 2 = yellow, 3 = red, 4 = green
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        # Player states
        self.players = [PlayerState() for _ in range(NUM_PLAYERS)]

        # Current player (0-3)
        self.current_player = BLUE

        # Turn tracking
        self.turn_number = 1
        self.consecutive_passes = 0
        self.is_game_over = False

        # For each player, track valid corner attachment points
        # Initially, each player can only attach at their starting corner
        self._valid_corners: List[Set[Tuple[int, int]]] = [
            {STARTING_CORNERS[i]} for i in range(NUM_PLAYERS)
        ]

        # Move history for potential replay
        self.move_history: List[Move] = []

    def clone(self) -> 'BlokusGame':
        """Create a deep copy of the game state."""
        game = BlokusGame.__new__(BlokusGame)
        game.board = self.board.copy()
        game.players = [p.clone() for p in self.players]
        game.current_player = self.current_player
        game.turn_number = self.turn_number
        game.consecutive_passes = self.consecutive_passes
        game.is_game_over = self.is_game_over
        game._valid_corners = [s.copy() for s in self._valid_corners]
        game.move_history = self.move_history.copy()
        return game

    def _is_within_bounds(self, row: int, col: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def _get_piece_cells(self, piece: PieceDefinition, orientation_idx: int,
                         row: int, col: int) -> np.ndarray:
        """Get absolute board coordinates for piece at position."""
        return piece.get_cells(orientation_idx, row, col)

    def _cells_within_bounds(self, cells: np.ndarray) -> bool:
        """Check if all cells are within board bounds."""
        return (cells[:, 0] >= 0).all() and (cells[:, 0] < BOARD_SIZE).all() and \
               (cells[:, 1] >= 0).all() and (cells[:, 1] < BOARD_SIZE).all()

    def _cells_overlap(self, cells: np.ndarray) -> bool:
        """Check if any cell overlaps with existing pieces."""
        for r, c in cells:
            if self.board[r, c] != 0:
                return True
        return False

    def _touches_starting_corner(self, cells: np.ndarray, player: int) -> bool:
        """Check if any cell touches the player's starting corner."""
        corner = STARTING_CORNERS[player]
        for r, c in cells:
            if r == corner[0] and c == corner[1]:
                return True
        return False

    def _touches_own_edge(self, cells: np.ndarray, player: int) -> bool:
        """Check if any cell is edge-adjacent to own color (not allowed)."""
        player_color = player + 1  # Board uses 1-4 for colors
        cell_set = set(map(tuple, cells))

        for r, c in cells:
            for dr, dc in EDGE_OFFSETS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cell_set:
                    continue  # Part of the piece being placed
                if self._is_within_bounds(nr, nc) and self.board[nr, nc] == player_color:
                    return True
        return False

    def _touches_own_diagonal(self, cells: np.ndarray, player: int) -> bool:
        """Check if any cell is diagonally adjacent to own color (required after first move)."""
        player_color = player + 1
        cell_set = set(map(tuple, cells))

        for r, c in cells:
            for dr, dc in DIAGONAL_OFFSETS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cell_set:
                    continue
                if self._is_within_bounds(nr, nc) and self.board[nr, nc] == player_color:
                    return True
        return False

    def is_valid_move(self, move: Move) -> bool:
        """Check if a move is valid."""
        player = move.player
        piece_idx = move.piece_idx
        player_state = self.players[player]

        # Check if piece is available
        if piece_idx not in player_state.remaining_pieces:
            return False

        piece = PIECE_BY_INDEX[piece_idx]

        # Check orientation is valid
        if move.orientation_idx < 0 or move.orientation_idx >= piece.num_orientations:
            return False

        # Get absolute cells
        cells = self._get_piece_cells(piece, move.orientation_idx, move.row, move.col)

        # Check bounds
        if not self._cells_within_bounds(cells):
            return False

        # Check overlap
        if self._cells_overlap(cells):
            return False

        # Check edge adjacency to own color (not allowed)
        if self._touches_own_edge(cells, player):
            return False

        # First move must cover starting corner
        if not player_state.has_played:
            if not self._touches_starting_corner(cells, player):
                return False
        else:
            # Subsequent moves must touch own color diagonally
            if not self._touches_own_diagonal(cells, player):
                return False

        return True

    def get_legal_moves(self, player: Optional[int] = None) -> List[Move]:
        """
        Get all legal moves for the specified player (or current player).

        Optimized by checking only positions near valid corners.
        """
        if player is None:
            player = self.current_player

        if self.is_game_over:
            return []

        player_state = self.players[player]
        if player_state.is_eliminated:
            return []

        moves = []

        # First move: must cover starting corner
        if not player_state.has_played:
            corner = STARTING_CORNERS[player]
            for piece_idx in player_state.remaining_pieces:
                piece = PIECE_BY_INDEX[piece_idx]
                for orient_idx in range(piece.num_orientations):
                    # Try positions where the piece covers the corner
                    shape = piece.orientations[orient_idx]
                    for cell_r, cell_c in shape:
                        # Position such that this cell lands on corner
                        pos_r = corner[0] - cell_r
                        pos_c = corner[1] - cell_c
                        move = Move(player, piece_idx, orient_idx, pos_r, pos_c)
                        if self.is_valid_move(move):
                            moves.append(move)
        else:
            # Subsequent moves: must touch own color diagonally
            # Use valid corners for faster search
            valid_corners = self._valid_corners[player]

            for piece_idx in player_state.remaining_pieces:
                piece = PIECE_BY_INDEX[piece_idx]
                for orient_idx in range(piece.num_orientations):
                    shape = piece.orientations[orient_idx]
                    # For each valid corner, try positions where a piece cell lands on it
                    for corner_r, corner_c in valid_corners:
                        for cell_r, cell_c in shape:
                            pos_r = corner_r - cell_r
                            pos_c = corner_c - cell_c
                            move = Move(player, piece_idx, orient_idx, pos_r, pos_c)
                            if self.is_valid_move(move):
                                moves.append(move)

        return moves

    def _update_valid_corners(self, cells: np.ndarray, player: int):
        """Update valid corner attachment points after placing a piece."""
        player_color = player + 1

        # Remove corners that are now occupied
        for p in range(NUM_PLAYERS):
            self._valid_corners[p] -= set(map(tuple, cells))

        # Also remove corners that are now edge-adjacent to the placed piece
        for r, c in cells:
            for dr, dc in EDGE_OFFSETS:
                nr, nc = r + dr, c + dc
                if self._is_within_bounds(nr, nc):
                    self._valid_corners[player].discard((nr, nc))

        # Add new diagonal corners for the placing player
        for r, c in cells:
            for dr, dc in DIAGONAL_OFFSETS:
                nr, nc = r + dr, c + dc
                if not self._is_within_bounds(nr, nc):
                    continue
                if self.board[nr, nc] != 0:
                    continue  # Already occupied

                # Check it's not edge-adjacent to own color
                is_edge_adjacent = False
                for edr, edc in EDGE_OFFSETS:
                    er, ec = nr + edr, nc + edc
                    if self._is_within_bounds(er, ec) and self.board[er, ec] == player_color:
                        is_edge_adjacent = True
                        break

                if not is_edge_adjacent:
                    self._valid_corners[player].add((nr, nc))

    def make_move(self, move: Move) -> bool:
        """
        Apply a move to the game.
        Returns True if successful.
        """
        if self.is_game_over:
            return False

        if move.player != self.current_player:
            return False

        if not self.is_valid_move(move):
            return False

        player = move.player
        player_state = self.players[player]
        piece = PIECE_BY_INDEX[move.piece_idx]

        # Get cells and place on board
        cells = self._get_piece_cells(piece, move.orientation_idx, move.row, move.col)
        player_color = player + 1

        for r, c in cells:
            self.board[r, c] = player_color

        # Update player state
        player_state.remaining_pieces.remove(move.piece_idx)
        player_state.has_played = True
        player_state.score += piece.size

        # Update valid corners
        self._update_valid_corners(cells, player)

        # Record move
        self.move_history.append(move)

        # Reset consecutive passes
        self.consecutive_passes = 0

        # Advance turn
        self._advance_turn()

        return True

    def pass_turn(self) -> bool:
        """
        Pass the current player's turn (only allowed if no legal moves).
        Returns True if successful.
        """
        if self.is_game_over:
            return False

        # Check if player has any legal moves
        if self.get_legal_moves():
            return False  # Cannot pass if moves are available

        player_state = self.players[self.current_player]
        player_state.is_eliminated = True
        self.consecutive_passes += 1

        self._advance_turn()
        return True

    def _advance_turn(self):
        """Advance to the next player."""
        next_player = (self.current_player + 1) % NUM_PLAYERS
        checked = 0

        while checked < NUM_PLAYERS:
            if not self.players[next_player].is_eliminated:
                break
            next_player = (next_player + 1) % NUM_PLAYERS
            checked += 1

        # Check if game is over
        if checked >= NUM_PLAYERS or self.consecutive_passes >= NUM_PLAYERS:
            self.is_game_over = True
            return

        self.current_player = next_player
        self.turn_number += 1

    def auto_pass_if_needed(self) -> bool:
        """Auto-pass for players with no moves. Returns True if passed."""
        if self.is_game_over:
            return False

        if not self.get_legal_moves():
            return self.pass_turn()
        return False

    def get_scores(self) -> List[int]:
        """
        Get final scores for all players.
        Score = cells placed + bonuses - remaining piece cells
        """
        scores = []
        for i, player_state in enumerate(self.players):
            score = player_state.score

            # Check if all pieces used
            if len(player_state.remaining_pieces) == 0:
                score += 15  # Bonus for using all pieces

                # Additional bonus if last piece was monomino
                if self.move_history:
                    last_move = None
                    for m in reversed(self.move_history):
                        if m.player == i:
                            last_move = m
                            break
                    if last_move and last_move.piece_idx == 0:  # Monomino
                        score += 5
            else:
                # Subtract remaining piece sizes
                for piece_idx in player_state.remaining_pieces:
                    score -= PIECE_BY_INDEX[piece_idx].size

            scores.append(score)

        return scores

    def get_winner(self) -> Optional[int]:
        """Get the winning player index, or None if tie."""
        scores = self.get_scores()
        max_score = max(scores)
        winners = [i for i, s in enumerate(scores) if s == max_score]

        if len(winners) == 1:
            return winners[0]
        return None  # Tie

    def get_state_for_neural_net(self, perspective_player: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state encoding for neural network input.

        Returns:
        - board_tensor: shape (8, 20, 20)
            Channel 0: Current player's pieces
            Channel 1-3: Other players' pieces (in turn order from current)
            Channel 4: Valid corner attachment points for current player
            Channel 5: Empty cells
            Channel 6: Turn number (normalized, broadcast)
            Channel 7: Current player indicator (one-hot style broadcast)

        - pieces_vector: shape (84,) - remaining pieces for all 4 players
            21 pieces * 4 players = 84 binary values
        """
        if perspective_player is None:
            perspective_player = self.current_player

        board_tensor = np.zeros((8, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        # Player order from perspective of current player
        player_order = [(perspective_player + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]

        # Channel 0: Current player's pieces
        board_tensor[0] = (self.board == player_order[0] + 1).astype(np.float32)

        # Channels 1-3: Other players' pieces
        for i in range(1, NUM_PLAYERS):
            board_tensor[i] = (self.board == player_order[i] + 1).astype(np.float32)

        # Channel 4: Valid corners for current player
        for r, c in self._valid_corners[perspective_player]:
            board_tensor[4, r, c] = 1.0

        # Channel 5: Empty cells
        board_tensor[5] = (self.board == 0).astype(np.float32)

        # Channel 6: Turn number (normalized to 0-1, max ~80 turns)
        board_tensor[6] = min(self.turn_number / 80.0, 1.0)

        # Channel 7: Current player indicator (0.25 * (player + 1))
        board_tensor[7] = (perspective_player + 1) / 4.0

        # Remaining pieces vector
        pieces_vector = np.zeros(NUM_PIECES * NUM_PLAYERS, dtype=np.float32)
        for i, player_idx in enumerate(player_order):
            for piece_idx in self.players[player_idx].remaining_pieces:
                pieces_vector[i * NUM_PIECES + piece_idx] = 1.0

        return board_tensor, pieces_vector

    def __repr__(self):
        return f"BlokusGame(turn={self.turn_number}, player={PLAYER_NAMES[self.current_player]}, over={self.is_game_over})"


def play_random_game() -> BlokusGame:
    """Play a random game and return the final state."""
    import random

    game = BlokusGame()

    while not game.is_game_over:
        moves = game.get_legal_moves()
        if moves:
            move = random.choice(moves)
            game.make_move(move)
        else:
            game.pass_turn()

    return game
