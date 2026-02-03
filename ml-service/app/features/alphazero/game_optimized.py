"""
Optimized Blokus game implementation with fast legal move generation.

Key optimizations:
1. Precomputed move tables: For each (piece, orientation), precompute which cell
   offsets can land on a corner to create valid diagonal adjacency
2. Vectorized validation: Check multiple moves at once using numpy operations
3. Move deduplication: Use set to avoid checking same move twice
4. Early exit validation: Order checks from cheapest to most expensive
5. Cached piece data: All piece cells stored as numpy arrays
6. Numba JIT compilation for hot validation loop
"""

from typing import List, Tuple, Optional, Set, Dict, FrozenSet
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from numba import njit, types
from numba.typed import List as NumbaList

from .pieces import (
    PIECE_CATALOG,
    PIECE_BY_INDEX,
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
    BLUE: (0, 0),
    YELLOW: (0, 19),
    RED: (19, 19),
    GREEN: (19, 0),
}

# Direction offsets
EDGE_OFFSETS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
DIAGONAL_OFFSETS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.int8)

# Precomputed center distance heatmap for neural network input
# Cells closer to center (9.5, 9.5) have higher values (0-1 normalized)
def _compute_center_distance_heatmap() -> np.ndarray:
    """Compute center distance heatmap once at module load."""
    heatmap = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    center = (BOARD_SIZE - 1) / 2.0  # 9.5 for 20x20 board
    max_dist = np.sqrt(2) * center  # ~13.44
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            dist = np.sqrt((r - center) ** 2 + (c - center) ** 2)
            heatmap[r, c] = 1.0 - (dist / max_dist)  # Closer to center = higher value
    return heatmap

_CENTER_DISTANCE_HEATMAP = _compute_center_distance_heatmap()


@njit(cache=True)
def _validate_move_numba(
    board: np.ndarray,
    cells: np.ndarray,
    row: int,
    col: int,
    player_color: int,
    has_played: bool,
    corner_r: int,
    corner_c: int,
) -> bool:
    """
    Numba-compiled move validation.
    Returns True if move is valid.
    """
    num_cells = cells.shape[0]

    # Check bounds and overlap, collect absolute positions
    abs_rows = np.empty(num_cells, dtype=np.int32)
    abs_cols = np.empty(num_cells, dtype=np.int32)

    for k in range(num_cells):
        r = cells[k, 0] + row
        c = cells[k, 1] + col

        if r < 0 or r >= 20 or c < 0 or c >= 20:
            return False

        if board[r, c] != 0:
            return False

        abs_rows[k] = r
        abs_cols[k] = c

    # Build set of relative cells for checking
    # (Using simple array search since numba doesn't support sets well)

    # Edge adjacency check
    edge_offsets = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)

    for k in range(num_cells):
        r, c = abs_rows[k], abs_cols[k]

        for e in range(4):
            nr = r + edge_offsets[e, 0]
            nc = c + edge_offsets[e, 1]

            # Check if neighbor is part of piece
            is_piece = False
            for m in range(num_cells):
                if abs_rows[m] == nr and abs_cols[m] == nc:
                    is_piece = True
                    break

            if is_piece:
                continue

            if 0 <= nr < 20 and 0 <= nc < 20:
                if board[nr, nc] == player_color:
                    return False

    # Connection check
    if not has_played:
        # Must cover starting corner
        covers = False
        for k in range(num_cells):
            if abs_rows[k] == corner_r and abs_cols[k] == corner_c:
                covers = True
                break
        if not covers:
            return False
    else:
        # Must have diagonal adjacency
        diag_offsets = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.int32)
        has_diag = False

        for k in range(num_cells):
            if has_diag:
                break
            r, c = abs_rows[k], abs_cols[k]

            for d in range(4):
                nr = r + diag_offsets[d, 0]
                nc = c + diag_offsets[d, 1]

                # Check if neighbor is part of piece
                is_piece = False
                for m in range(num_cells):
                    if abs_rows[m] == nr and abs_cols[m] == nc:
                        is_piece = True
                        break

                if is_piece:
                    continue

                if 0 <= nr < 20 and 0 <= nc < 20:
                    if board[nr, nc] == player_color:
                        has_diag = True
                        break

        if not has_diag:
            return False

    return True


@njit(cache=True)
def _batch_validate_numba(
    board: np.ndarray,
    candidates: np.ndarray,  # (N, 4) - piece_idx, orient_idx, row, col
    remaining_mask: np.ndarray,  # (21,) bool - which pieces available
    num_orients: np.ndarray,  # (21,) - num orientations per piece
    all_cells: np.ndarray,  # (21, 8, 5, 2) - cell offsets
    cell_counts: np.ndarray,  # (21,) - num cells per piece
    player_color: int,
    has_played: bool,
    corner_r: int,
    corner_c: int,
) -> np.ndarray:
    """
    Batch validate moves with numba.
    """
    n = candidates.shape[0]
    results = np.ones(n, dtype=np.bool_)

    for i in range(n):
        piece_idx = candidates[i, 0]
        orient_idx = candidates[i, 1]
        row = candidates[i, 2]
        col = candidates[i, 3]

        # Check piece available
        if not remaining_mask[piece_idx]:
            results[i] = False
            continue

        # Check valid orientation
        if orient_idx >= num_orients[piece_idx]:
            results[i] = False
            continue

        # Get cells for this piece/orientation
        num_cells = cell_counts[piece_idx]
        cells = all_cells[piece_idx, orient_idx, :num_cells, :]

        # Validate the move
        if not _validate_move_numba(
            board, cells, row, col, player_color, has_played, corner_r, corner_c
        ):
            results[i] = False

    return results


@dataclass
class Move:
    """A move in Blokus."""
    player: int
    piece_idx: int
    orientation_idx: int
    row: int
    col: int

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


class PrecomputedMoveData:
    """
    Precomputed data for fast move generation.

    For each (piece, orientation), stores:
    - cells: numpy array of cell offsets
    - cell_set: frozenset for fast membership testing
    - For each cell index, the position offset to place piece such that
      this cell lands on a given corner
    """

    _instance = None

    @classmethod
    def get(cls) -> 'PrecomputedMoveData':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.piece_cells: List[List[np.ndarray]] = []  # [piece_idx][orient_idx] -> (N, 2)
        self.piece_cell_sets: List[List[FrozenSet[Tuple[int, int]]]] = []
        self.piece_sizes: List[int] = []
        self.piece_num_orients: List[int] = []

        # For starting corner moves: (piece_idx, orient_idx, pos_row, pos_col) tuples
        # that place a piece covering each starting corner
        self.starting_corner_moves: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}

        # Numba-compatible arrays
        self.all_cells: np.ndarray = None  # (21, 8, 5, 2)
        self.cell_counts: np.ndarray = None  # (21,)
        self.num_orients_arr: np.ndarray = None  # (21,)

        self._build()

    def _build(self):
        """Build all precomputed data."""
        # Initialize numba arrays
        self.all_cells = np.zeros((NUM_PIECES, 8, 5, 2), dtype=np.int32)
        self.cell_counts = np.zeros(NUM_PIECES, dtype=np.int32)
        self.num_orients_arr = np.zeros(NUM_PIECES, dtype=np.int32)

        for piece_idx, piece in enumerate(PIECE_CATALOG):
            self.piece_sizes.append(piece.size)
            self.piece_num_orients.append(piece.num_orientations)
            self.cell_counts[piece_idx] = piece.size
            self.num_orients_arr[piece_idx] = piece.num_orientations

            piece_cells = []
            piece_cell_sets = []

            for orient_idx in range(piece.num_orientations):
                cells = np.array(piece.orientations[orient_idx], dtype=np.int8)
                piece_cells.append(cells)
                piece_cell_sets.append(frozenset(map(tuple, cells)))

                # Store in numba array
                for k, (r, c) in enumerate(piece.orientations[orient_idx]):
                    self.all_cells[piece_idx, orient_idx, k, 0] = r
                    self.all_cells[piece_idx, orient_idx, k, 1] = c

            # Pad with empty arrays for unused orientations
            while len(piece_cells) < 8:
                piece_cells.append(np.zeros((0, 2), dtype=np.int8))
                piece_cell_sets.append(frozenset())

            self.piece_cells.append(piece_cells)
            self.piece_cell_sets.append(piece_cell_sets)

        # Precompute starting corner moves
        for player in range(NUM_PLAYERS):
            corner = STARTING_CORNERS[player]
            moves = []

            for piece_idx in range(NUM_PIECES):
                for orient_idx in range(self.piece_num_orients[piece_idx]):
                    cells = self.piece_cells[piece_idx][orient_idx]
                    for cell in cells:
                        # Position such that this cell lands on corner
                        pos_r = corner[0] - cell[0]
                        pos_c = corner[1] - cell[1]
                        moves.append((piece_idx, orient_idx, pos_r, pos_c))

            self.starting_corner_moves[corner] = moves


class BlokusGameOptimized:
    """
    Optimized Blokus game with fast legal move generation.
    """

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.players = [PlayerState() for _ in range(NUM_PLAYERS)]
        self.current_player = BLUE
        self.turn_number = 1
        self.consecutive_passes = 0
        self.is_game_over = False

        self._valid_corners: List[Set[Tuple[int, int]]] = [
            {STARTING_CORNERS[i]} for i in range(NUM_PLAYERS)
        ]

        self.move_history: List[Move] = []
        self._precomputed = PrecomputedMoveData.get()

        # Cache for legal moves (invalidated on state change)
        self._legal_moves_cache: Optional[List[Move]] = None
        self._legal_moves_cache_player: Optional[int] = None

    def clone(self) -> 'BlokusGameOptimized':
        """Create a deep copy of the game state."""
        game = BlokusGameOptimized.__new__(BlokusGameOptimized)
        game.board = self.board.copy()
        game.players = [p.clone() for p in self.players]
        game.current_player = self.current_player
        game.turn_number = self.turn_number
        game.consecutive_passes = self.consecutive_passes
        game.is_game_over = self.is_game_over
        game._valid_corners = [s.copy() for s in self._valid_corners]
        game.move_history = []  # Don't copy history - not needed for MCTS
        game._precomputed = self._precomputed
        game._legal_moves_cache = None
        game._legal_moves_cache_player = None
        return game

    def _invalidate_cache(self):
        """Invalidate the legal moves cache."""
        self._legal_moves_cache = None
        self._legal_moves_cache_player = None

    def is_valid_move_fast(self, piece_idx: int, orient_idx: int, row: int, col: int, player: int) -> bool:
        """
        Fast move validation with early exits.
        Checks are ordered from cheapest to most expensive.
        """
        player_state = self.players[player]

        # Check 1: Piece available (O(1) set lookup)
        if piece_idx not in player_state.remaining_pieces:
            return False

        # Check 2: Valid orientation
        if orient_idx >= self._precomputed.piece_num_orients[piece_idx]:
            return False

        # Get cells
        cells = self._precomputed.piece_cells[piece_idx][orient_idx]
        num_cells = len(cells)

        # Check 3: Bounds check (vectorized)
        abs_rows = cells[:, 0] + row
        abs_cols = cells[:, 1] + col

        if (abs_rows < 0).any() or (abs_rows >= BOARD_SIZE).any():
            return False
        if (abs_cols < 0).any() or (abs_cols >= BOARD_SIZE).any():
            return False

        # Check 4: Overlap check (vectorized board lookup)
        board_values = self.board[abs_rows, abs_cols]
        if (board_values != 0).any():
            return False

        player_color = player + 1

        # Check 5: Edge adjacency to own color (not allowed)
        # Build set of piece cells for fast lookup
        cell_set = self._precomputed.piece_cell_sets[piece_idx][orient_idx]

        for i in range(num_cells):
            r, c = abs_rows[i], abs_cols[i]
            for dr, dc in EDGE_OFFSETS:
                nr, nc = r + dr, c + dc
                # Skip if neighbor is part of the piece
                rel_nr, rel_nc = nr - row, nc - col
                if (rel_nr, rel_nc) in cell_set:
                    continue
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if self.board[nr, nc] == player_color:
                        return False

        # Check 6: Connection rule
        if not player_state.has_played:
            # First move: must cover starting corner
            corner = STARTING_CORNERS[player]
            covers_corner = False
            for i in range(num_cells):
                if abs_rows[i] == corner[0] and abs_cols[i] == corner[1]:
                    covers_corner = True
                    break
            if not covers_corner:
                return False
        else:
            # Subsequent moves: must have diagonal adjacency to own color
            has_diagonal = False
            for i in range(num_cells):
                r, c = abs_rows[i], abs_cols[i]
                for dr, dc in DIAGONAL_OFFSETS:
                    nr, nc = r + dr, c + dc
                    rel_nr, rel_nc = nr - row, nc - col
                    if (rel_nr, rel_nc) in cell_set:
                        continue
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if self.board[nr, nc] == player_color:
                            has_diagonal = True
                            break
                if has_diagonal:
                    break
            if not has_diagonal:
                return False

        return True

    def get_legal_moves(self, player: Optional[int] = None) -> List[Move]:
        """
        Get all legal moves with caching and batch validation.
        """
        if player is None:
            player = self.current_player

        # Check cache
        if self._legal_moves_cache is not None and self._legal_moves_cache_player == player:
            return self._legal_moves_cache

        if self.is_game_over:
            return []

        player_state = self.players[player]
        if player_state.is_eliminated:
            return []

        remaining = player_state.remaining_pieces

        # Collect all candidate moves first, then batch validate
        candidates: List[Tuple[int, int, int, int]] = []
        seen: Set[Tuple[int, int, int, int]] = set()

        if not player_state.has_played:
            corner = STARTING_CORNERS[player]
            for piece_idx, orient_idx, pos_r, pos_c in self._precomputed.starting_corner_moves[corner]:
                if piece_idx not in remaining:
                    continue
                key = (piece_idx, orient_idx, pos_r, pos_c)
                if key not in seen:
                    seen.add(key)
                    candidates.append(key)
        else:
            valid_corners = list(self._valid_corners[player])

            for piece_idx in remaining:
                num_orients = self._precomputed.piece_num_orients[piece_idx]

                for orient_idx in range(num_orients):
                    cells = self._precomputed.piece_cells[piece_idx][orient_idx]

                    for corner_r, corner_c in valid_corners:
                        for cell in cells:
                            pos_r = corner_r - cell[0]
                            pos_c = corner_c - cell[1]

                            key = (piece_idx, orient_idx, pos_r, pos_c)
                            if key not in seen:
                                seen.add(key)
                                candidates.append(key)

        # Batch validate all candidates
        if not candidates:
            self._legal_moves_cache = []
            self._legal_moves_cache_player = player
            return []

        valid_mask = self._batch_validate_moves(candidates, player)
        moves = [
            Move(player, c[0], c[1], c[2], c[3])
            for c, valid in zip(candidates, valid_mask)
            if valid
        ]

        self._legal_moves_cache = moves
        self._legal_moves_cache_player = player
        return moves

    def _batch_validate_moves(
        self,
        candidates: List[Tuple[int, int, int, int]],
        player: int
    ) -> List[bool]:
        """
        Validate multiple moves using numba-compiled function.
        """
        n = len(candidates)
        if n == 0:
            return []

        player_state = self.players[player]
        player_color = player + 1
        has_played = player_state.has_played
        remaining = player_state.remaining_pieces
        corner = STARTING_CORNERS[player]
        precomputed = self._precomputed

        # Build remaining mask
        remaining_mask = np.zeros(NUM_PIECES, dtype=np.bool_)
        for p in remaining:
            remaining_mask[p] = True

        # Convert candidates to numpy array
        candidates_arr = np.array(candidates, dtype=np.int32)

        # Call numba function
        results = _batch_validate_numba(
            self.board.astype(np.int32),
            candidates_arr,
            remaining_mask,
            precomputed.num_orients_arr,
            precomputed.all_cells,
            precomputed.cell_counts,
            player_color,
            has_played,
            corner[0],
            corner[1],
        )

        return results.tolist()

    def is_valid_move(self, move: Move) -> bool:
        """Check if a move is valid."""
        return self.is_valid_move_fast(
            move.piece_idx, move.orientation_idx, move.row, move.col, move.player
        )

    def _update_valid_corners(self, cells: np.ndarray, player: int):
        """Update valid corner attachment points after placing a piece."""
        player_color = player + 1
        cell_tuples = set(map(tuple, cells))

        # Remove occupied cells from all players' corners
        for p in range(NUM_PLAYERS):
            self._valid_corners[p] -= cell_tuples

        # Remove edge-adjacent cells from placing player's corners
        for r, c in cells:
            for dr, dc in EDGE_OFFSETS:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    self._valid_corners[player].discard((nr, nc))

        # Add diagonal cells as new valid corners
        for r, c in cells:
            for dr, dc in DIAGONAL_OFFSETS:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if self.board[nr, nc] != 0:
                    continue

                # Check not edge-adjacent to own color
                is_edge_adjacent = False
                for edr, edc in EDGE_OFFSETS:
                    er, ec = nr + edr, nc + edc
                    if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
                        if self.board[er, ec] == player_color:
                            is_edge_adjacent = True
                            break

                if not is_edge_adjacent:
                    self._valid_corners[player].add((nr, nc))

    def make_move(self, move: Move) -> bool:
        """Apply a move to the game."""
        if self.is_game_over:
            return False

        if move.player != self.current_player:
            return False

        if not self.is_valid_move(move):
            return False

        player = move.player
        player_state = self.players[player]

        # Get cells and place
        cells = self._precomputed.piece_cells[move.piece_idx][move.orientation_idx]
        abs_cells = cells + np.array([[move.row, move.col]], dtype=np.int8)

        player_color = player + 1
        for r, c in abs_cells:
            self.board[r, c] = player_color

        # Update player state
        player_state.remaining_pieces.remove(move.piece_idx)
        player_state.has_played = True
        player_state.score += len(cells)

        # Update valid corners
        self._update_valid_corners(abs_cells, player)

        # Record and reset
        self.move_history.append(move)
        self.consecutive_passes = 0
        self._invalidate_cache()

        # Advance turn
        self._advance_turn()

        return True

    def pass_turn(self) -> bool:
        """Pass the current player's turn."""
        if self.is_game_over:
            return False

        if self.get_legal_moves():
            return False

        self.players[self.current_player].is_eliminated = True
        self.consecutive_passes += 1
        self._invalidate_cache()

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

        if checked >= NUM_PLAYERS or self.consecutive_passes >= NUM_PLAYERS:
            self.is_game_over = True
            return

        self.current_player = next_player
        self.turn_number += 1
        self._invalidate_cache()

    def get_scores(self) -> List[int]:
        """Get final scores for all players."""
        scores = []
        for i, player_state in enumerate(self.players):
            score = player_state.score

            if len(player_state.remaining_pieces) == 0:
                score += 15
                if self.move_history:
                    for m in reversed(self.move_history):
                        if m.player == i:
                            if m.piece_idx == 0:  # Monomino
                                score += 5
                            break
            else:
                for piece_idx in player_state.remaining_pieces:
                    score -= self._precomputed.piece_sizes[piece_idx]

            scores.append(score)

        return scores

    def get_winner(self) -> Optional[int]:
        """Get the winning player index, or None if tie."""
        scores = self.get_scores()
        max_score = max(scores)
        winners = [i for i, s in enumerate(scores) if s == max_score]

        if len(winners) == 1:
            return winners[0]
        return None

    def get_state_for_neural_net(self, perspective_player: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state encoding for neural network input.

        Returns:
        - board_tensor: shape (10, 20, 20)
            Channel 0: Current player's pieces
            Channel 1-3: Other players' pieces (in turn order from current)
            Channel 4: Valid corner attachment points for current player
            Channel 5: Empty cells
            Channel 6: Turn number (normalized, broadcast)
            Channel 7: Current player indicator (one-hot style broadcast)
            Channel 8: Center distance (cells closer to center have higher values)
            Channel 9: Game phase (0.0=early, 0.5=mid, 1.0=late)

        - pieces_vector: shape (84,) - remaining pieces for all 4 players
        """
        if perspective_player is None:
            perspective_player = self.current_player

        board_tensor = np.zeros((10, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        player_order = [(perspective_player + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]

        # Channels 0-3: Player pieces
        for i in range(NUM_PLAYERS):
            board_tensor[i] = (self.board == player_order[i] + 1).astype(np.float32)

        # Channel 4: Valid corners
        for r, c in self._valid_corners[perspective_player]:
            board_tensor[4, r, c] = 1.0

        # Channel 5: Empty cells
        board_tensor[5] = (self.board == 0).astype(np.float32)

        # Channel 6: Turn number
        board_tensor[6] = min(self.turn_number / 80.0, 1.0)

        # Channel 7: Current player
        board_tensor[7] = (perspective_player + 1) / 4.0

        # Channel 8: Center distance heatmap
        board_tensor[8] = _CENTER_DISTANCE_HEATMAP

        # Channel 9: Game phase (based on total pieces remaining)
        # 84 total pieces at start (21 per player * 4 players)
        total_remaining = sum(len(p.remaining_pieces) for p in self.players)
        pieces_played = 84 - total_remaining
        if pieces_played <= 28:
            game_phase = 0.0  # Early game
        elif pieces_played <= 56:
            game_phase = 0.5  # Mid game
        else:
            game_phase = 1.0  # Late game
        board_tensor[9] = game_phase

        # Pieces vector
        pieces_vector = np.zeros(NUM_PIECES * NUM_PLAYERS, dtype=np.float32)
        for i, player_idx in enumerate(player_order):
            for piece_idx in self.players[player_idx].remaining_pieces:
                pieces_vector[i * NUM_PIECES + piece_idx] = 1.0

        return board_tensor, pieces_vector


def benchmark_optimized():
    """Benchmark optimized vs original implementation."""
    import time
    from .game import BlokusGame
    import random

    print("Benchmarking legal move generation...")
    print()

    # Original
    print("Original BlokusGame:")
    game_orig = BlokusGame()

    start = time.time()
    for _ in range(100):
        game_orig.get_legal_moves()
    orig_time = (time.time() - start) * 10  # ms per call
    print(f"  First move: {orig_time:.2f}ms per call")

    # Optimized
    print("\nOptimized BlokusGameOptimized:")
    game_opt = BlokusGameOptimized()

    # Warm up precomputation
    _ = game_opt.get_legal_moves()

    start = time.time()
    for _ in range(100):
        game_opt._invalidate_cache()  # Force recomputation
        game_opt.get_legal_moves()
    opt_time = (time.time() - start) * 10
    print(f"  First move: {opt_time:.2f}ms per call")
    print(f"  Speedup: {orig_time/opt_time:.1f}x")

    # With caching
    start = time.time()
    for _ in range(100):
        game_opt.get_legal_moves()  # Cached
    cached_time = (time.time() - start) * 10
    print(f"  Cached: {cached_time:.4f}ms per call")

    # Verify correctness
    print("\nVerifying correctness...")
    for trial in range(10):
        game_orig = BlokusGame()
        game_opt = BlokusGameOptimized()

        for move_num in range(30):
            moves_orig = game_orig.get_legal_moves()
            moves_opt = game_opt.get_legal_moves()

            # Convert to comparable format
            set_orig = {(m.piece_idx, m.orientation_idx, m.row, m.col) for m in moves_orig}
            set_opt = {(m.piece_idx, m.orientation_idx, m.row, m.col) for m in moves_opt}

            if set_orig != set_opt:
                print(f"  MISMATCH at trial {trial}, move {move_num}!")
                print(f"    Original: {len(set_orig)} moves")
                print(f"    Optimized: {len(set_opt)} moves")
                print(f"    Missing: {set_orig - set_opt}")
                print(f"    Extra: {set_opt - set_orig}")
                return False

            if not moves_orig:
                game_orig.pass_turn()
                game_opt.pass_turn()
                continue

            # Make same random move
            move = random.choice(moves_orig)
            game_orig.make_move(move)

            opt_move = Move(move.player, move.piece_idx, move.orientation_idx, move.row, move.col)
            game_opt.make_move(opt_move)

    print("  All tests passed!")

    # Benchmark mid-game
    print("\nMid-game benchmark (after 10 moves):")
    game_orig = BlokusGame()
    game_opt = BlokusGameOptimized()

    for _ in range(10):
        moves = game_orig.get_legal_moves()
        if moves:
            m = random.choice(moves)
            game_orig.make_move(m)
            game_opt.make_move(Move(m.player, m.piece_idx, m.orientation_idx, m.row, m.col))

    start = time.time()
    for _ in range(100):
        game_orig.get_legal_moves()
    orig_mid = (time.time() - start) * 10

    start = time.time()
    for _ in range(100):
        game_opt._invalidate_cache()
        game_opt.get_legal_moves()
    opt_mid = (time.time() - start) * 10

    print(f"  Original: {orig_mid:.2f}ms")
    print(f"  Optimized: {opt_mid:.2f}ms")
    print(f"  Speedup: {orig_mid/opt_mid:.1f}x")


if __name__ == "__main__":
    benchmark_optimized()
