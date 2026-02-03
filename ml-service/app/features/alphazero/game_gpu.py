"""
GPU-accelerated Blokus game engine for AlphaZero MCTS.

Represents batched game states as GPU tensors for parallel operations:
- Clone N game states: O(1) tensor indexing
- Apply N moves: batched scatter operations
- Validate N moves: batched convolution/masking

This eliminates the CPU bottleneck in MCTS tree traversal where game.clone()
and game.make_move() dominate runtime.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

from .game import BlokusGame, Move, PlayerState, BOARD_SIZE, NUM_PLAYERS, STARTING_CORNERS
from .pieces import PIECE_CATALOG, PIECE_BY_INDEX, NUM_PIECES


# Constants
BLUE, YELLOW, RED, GREEN = 0, 1, 2, 3

# Edge and diagonal offsets for neighbor checking
EDGE_OFFSETS = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.int64)
DIAGONAL_OFFSETS = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.int64)


def _build_piece_tensors(device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build pre-computed piece data as GPU tensors.

    Returns:
        piece_cells: (21, 8, 5, 2) - cell offsets for each (piece, orientation)
                     Padded with -1 for unused cells/orientations
        piece_num_cells: (21,) - number of cells per piece
        piece_num_orients: (21,) - number of valid orientations per piece
    """
    max_cells = 5  # Pentominoes have 5 cells
    max_orients = 8  # Maximum orientations

    # Initialize with -1 (invalid marker)
    piece_cells = torch.full((NUM_PIECES, max_orients, max_cells, 2), -1,
                             dtype=torch.int64, device=device)
    piece_num_cells = torch.zeros(NUM_PIECES, dtype=torch.int64, device=device)
    piece_num_orients = torch.zeros(NUM_PIECES, dtype=torch.int64, device=device)

    for piece_idx, piece in enumerate(PIECE_CATALOG):
        piece_num_cells[piece_idx] = piece.size
        piece_num_orients[piece_idx] = piece.num_orientations

        for orient_idx, orient in enumerate(piece.orientations):
            for cell_idx, (row, col) in enumerate(orient):
                piece_cells[piece_idx, orient_idx, cell_idx, 0] = row
                piece_cells[piece_idx, orient_idx, cell_idx, 1] = col

    return piece_cells, piece_num_cells, piece_num_orients


class PieceDataGPU:
    """Pre-computed piece data on GPU for fast lookup."""

    _instances: dict = {}

    @classmethod
    def get(cls, device: str = 'cuda') -> 'PieceDataGPU':
        """Get or create piece data for device (singleton per device)."""
        if device not in cls._instances:
            cls._instances[device] = cls(device)
        return cls._instances[device]

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.piece_cells, self.piece_num_cells, self.piece_num_orients = \
            _build_piece_tensors(device)

        # Also store starting corners as tensor
        self.starting_corners = torch.tensor([
            STARTING_CORNERS[BLUE],
            STARTING_CORNERS[YELLOW],
            STARTING_CORNERS[RED],
            STARTING_CORNERS[GREEN],
        ], dtype=torch.int64, device=device)

        # Edge and diagonal offsets
        self.edge_offsets = EDGE_OFFSETS.to(device)
        self.diagonal_offsets = DIAGONAL_OFFSETS.to(device)


class BlokusGameBatchGPU:
    """
    Batched Blokus game states on GPU.

    All operations are vectorized across the batch dimension for maximum
    GPU utilization.
    """

    def __init__(self, batch_size: int, device: str = 'cuda'):
        """
        Initialize batch of game states.

        Args:
            batch_size: Number of parallel game states
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.batch_size = batch_size
        self.device = device
        self.piece_data = PieceDataGPU.get(device)

        # Board states: (B, 20, 20) - 0=empty, 1-4=player color
        self.boards = torch.zeros(batch_size, BOARD_SIZE, BOARD_SIZE,
                                  dtype=torch.int8, device=device)

        # Remaining pieces: (B, 4, 21) - bool mask per player
        self.remaining_pieces = torch.ones(batch_size, NUM_PLAYERS, NUM_PIECES,
                                           dtype=torch.bool, device=device)

        # Current player: (B,) - 0-3
        self.current_player = torch.zeros(batch_size, dtype=torch.int8, device=device)

        # Player has played first move: (B, 4)
        self.has_played = torch.zeros(batch_size, NUM_PLAYERS,
                                      dtype=torch.bool, device=device)

        # Player is eliminated: (B, 4)
        self.is_eliminated = torch.zeros(batch_size, NUM_PLAYERS,
                                         dtype=torch.bool, device=device)

        # Valid corners for each player: (B, 4, 20, 20) - bool mask
        self.valid_corners = torch.zeros(batch_size, NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE,
                                         dtype=torch.bool, device=device)

        # Set starting corners
        self.valid_corners[:, BLUE, 0, 0] = True      # Blue: top-left
        self.valid_corners[:, YELLOW, 0, 19] = True   # Yellow: top-right
        self.valid_corners[:, RED, 19, 19] = True     # Red: bottom-right
        self.valid_corners[:, GREEN, 19, 0] = True    # Green: bottom-left

        # Turn tracking: (B,)
        self.turn_number = torch.ones(batch_size, dtype=torch.int32, device=device)
        self.consecutive_passes = torch.zeros(batch_size, dtype=torch.int8, device=device)
        self.is_game_over = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Player scores: (B, 4)
        self.scores = torch.zeros(batch_size, NUM_PLAYERS, dtype=torch.int32, device=device)

    def clone_states(self, indices: torch.Tensor) -> 'BlokusGameBatchGPU':
        """
        Clone specific game states into a new batch.

        Args:
            indices: (N,) tensor of indices to clone

        Returns:
            New BlokusGameBatchGPU of size N with cloned states
        """
        n = indices.shape[0]
        new_batch = BlokusGameBatchGPU.__new__(BlokusGameBatchGPU)
        new_batch.batch_size = n
        new_batch.device = self.device
        new_batch.piece_data = self.piece_data

        # Clone all tensors via indexing (efficient GPU operation)
        new_batch.boards = self.boards[indices].clone()
        new_batch.remaining_pieces = self.remaining_pieces[indices].clone()
        new_batch.current_player = self.current_player[indices].clone()
        new_batch.has_played = self.has_played[indices].clone()
        new_batch.is_eliminated = self.is_eliminated[indices].clone()
        new_batch.valid_corners = self.valid_corners[indices].clone()
        new_batch.turn_number = self.turn_number[indices].clone()
        new_batch.consecutive_passes = self.consecutive_passes[indices].clone()
        new_batch.is_game_over = self.is_game_over[indices].clone()
        new_batch.scores = self.scores[indices].clone()

        return new_batch

    def _get_piece_cells_batched(
        self,
        piece_indices: torch.Tensor,
        orient_indices: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get absolute board coordinates for batched pieces.

        Args:
            piece_indices: (N,) piece indices 0-20
            orient_indices: (N,) orientation indices 0-7
            rows: (N,) position rows
            cols: (N,) position cols

        Returns:
            cells: (N, 5, 2) absolute coordinates, -1 for unused cells
            num_cells: (N,) number of valid cells per piece
        """
        n = piece_indices.shape[0]

        # Get relative cells from piece data: (N, 5, 2)
        relative_cells = self.piece_data.piece_cells[piece_indices, orient_indices]

        # Create offset tensor: (N, 1, 2)
        offsets = torch.stack([rows, cols], dim=1).unsqueeze(1)

        # Add offsets to get absolute positions
        # Only add where cells are valid (>= 0)
        cells = relative_cells.clone()
        valid_mask = (relative_cells >= 0).all(dim=2, keepdim=True)  # (N, 5, 1)
        cells = torch.where(valid_mask, relative_cells + offsets, relative_cells)

        num_cells = self.piece_data.piece_num_cells[piece_indices]

        return cells, num_cells

    def validate_moves(
        self,
        batch_indices: torch.Tensor,
        piece_indices: torch.Tensor,
        orient_indices: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if N moves are valid in parallel.

        Args:
            batch_indices: (N,) which game states
            piece_indices: (N,) piece 0-20
            orient_indices: (N,) orientation 0-7
            rows: (N,) position row
            cols: (N,) position col

        Returns:
            (N,) bool tensor - True if move is valid
        """
        n = batch_indices.shape[0]
        device = self.device

        # Get current players for these games
        players = self.current_player[batch_indices].long()  # (N,)

        # Check 1: Piece is available
        piece_available = self.remaining_pieces[batch_indices, players, piece_indices]  # (N,)

        # Check 2: Valid orientation
        valid_orient = orient_indices < self.piece_data.piece_num_orients[piece_indices]

        # Get cell coordinates: (N, 5, 2)
        cells, num_cells = self._get_piece_cells_batched(
            piece_indices, orient_indices, rows, cols
        )

        # Check 3: All cells within bounds
        # Only check valid cells (num_cells vary per piece)
        cell_mask = torch.arange(5, device=device).unsqueeze(0) < num_cells.unsqueeze(1)  # (N, 5)
        rows_valid = (cells[:, :, 0] >= 0) & (cells[:, :, 0] < BOARD_SIZE)
        cols_valid = (cells[:, :, 1] >= 0) & (cells[:, :, 1] < BOARD_SIZE)
        within_bounds = ((rows_valid & cols_valid) | ~cell_mask).all(dim=1)  # (N,)

        # For overlap and adjacency checks, clamp coordinates to valid range
        clamped_rows = cells[:, :, 0].clamp(0, BOARD_SIZE - 1)
        clamped_cols = cells[:, :, 1].clamp(0, BOARD_SIZE - 1)

        # Check 4: No overlap with existing pieces
        # boards[batch_indices] gives (N, 20, 20)
        boards_selected = self.boards[batch_indices]  # (N, 20, 20)

        # Gather board values at cell positions
        # Using advanced indexing: for each move, check all 5 cells
        no_overlap = torch.ones(n, dtype=torch.bool, device=device)
        for cell_idx in range(5):
            is_valid_cell = cell_idx < num_cells  # (N,)
            r = clamped_rows[:, cell_idx]  # (N,)
            c = clamped_cols[:, cell_idx]  # (N,)
            cell_value = boards_selected[torch.arange(n, device=device), r, c]  # (N,)
            cell_occupied = cell_value != 0
            # Only invalid if this is a valid cell AND it's occupied
            no_overlap = no_overlap & (~(is_valid_cell & cell_occupied))

        # Check 5: First move covers starting corner OR diagonal adjacency
        has_played_mask = self.has_played[batch_indices, players]  # (N,)

        # First move check: piece must cover starting corner
        starting_corners = self.piece_data.starting_corners[players]  # (N, 2)
        covers_start = torch.zeros(n, dtype=torch.bool, device=device)
        for cell_idx in range(5):
            is_valid_cell = cell_idx < num_cells
            r = clamped_rows[:, cell_idx]
            c = clamped_cols[:, cell_idx]
            matches_corner = (r == starting_corners[:, 0]) & (c == starting_corners[:, 1])
            covers_start = covers_start | (is_valid_cell & matches_corner)

        # Subsequent move check: diagonal adjacency to own color
        player_colors = (players + 1).to(torch.int8)  # Board uses 1-4

        # Check diagonal neighbors for each cell
        has_diagonal = torch.zeros(n, dtype=torch.bool, device=device)
        for cell_idx in range(5):
            is_valid_cell = cell_idx < num_cells
            r = clamped_rows[:, cell_idx]
            c = clamped_cols[:, cell_idx]

            for diag_idx in range(4):
                dr = self.piece_data.diagonal_offsets[diag_idx, 0]
                dc = self.piece_data.diagonal_offsets[diag_idx, 1]
                nr = (r + dr).clamp(0, BOARD_SIZE - 1)
                nc = (c + dc).clamp(0, BOARD_SIZE - 1)

                # Check if within bounds
                in_bounds = ((r + dr) >= 0) & ((r + dr) < BOARD_SIZE) & \
                           ((c + dc) >= 0) & ((c + dc) < BOARD_SIZE)

                neighbor_value = boards_selected[torch.arange(n, device=device), nr, nc]
                is_own_color = neighbor_value == player_colors

                has_diagonal = has_diagonal | (is_valid_cell & in_bounds & is_own_color)

        # Connection rule: first move needs starting corner, others need diagonal
        connection_valid = torch.where(has_played_mask, has_diagonal, covers_start)

        # Check 6: No edge adjacency to own color
        no_edge = torch.ones(n, dtype=torch.bool, device=device)

        # Build set of cells being placed (for each move)
        # We need to exclude cells that are part of the piece itself
        for cell_idx in range(5):
            is_valid_cell = cell_idx < num_cells
            r = clamped_rows[:, cell_idx]
            c = clamped_cols[:, cell_idx]

            for edge_idx in range(4):
                dr = self.piece_data.edge_offsets[edge_idx, 0]
                dc = self.piece_data.edge_offsets[edge_idx, 1]
                nr = r + dr
                nc = c + dc

                # Check if within bounds
                in_bounds = (nr >= 0) & (nr < BOARD_SIZE) & (nc >= 0) & (nc < BOARD_SIZE)
                nr_clamped = nr.clamp(0, BOARD_SIZE - 1)
                nc_clamped = nc.clamp(0, BOARD_SIZE - 1)

                # Check if neighbor is part of the piece (skip if so)
                is_part_of_piece = torch.zeros(n, dtype=torch.bool, device=device)
                for other_cell_idx in range(5):
                    other_valid = other_cell_idx < num_cells
                    other_r = clamped_rows[:, other_cell_idx]
                    other_c = clamped_cols[:, other_cell_idx]
                    is_part_of_piece = is_part_of_piece | (
                        other_valid & (nr == other_r) & (nc == other_c)
                    )

                neighbor_value = boards_selected[torch.arange(n, device=device), nr_clamped, nc_clamped]
                is_edge_adj = neighbor_value == player_colors

                # Invalid if: this cell is valid, neighbor in bounds, not part of piece, and is own color
                violation = is_valid_cell & in_bounds & ~is_part_of_piece & is_edge_adj
                no_edge = no_edge & ~violation

        # Combine all checks
        valid = (piece_available & valid_orient & within_bounds &
                 no_overlap & connection_valid & no_edge)

        return valid

    def apply_moves(
        self,
        batch_indices: torch.Tensor,
        piece_indices: torch.Tensor,
        orient_indices: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        skip_validation: bool = False,
    ):
        """
        Apply N moves in parallel.

        Args:
            batch_indices: (N,) which games to modify
            piece_indices: (N,) piece 0-20
            orient_indices: (N,) orientation 0-7
            rows: (N,) position row
            cols: (N,) position col
            skip_validation: If True, skip validity check (for speed when pre-validated)
        """
        n = batch_indices.shape[0]
        if n == 0:
            return

        device = self.device

        # Get current players
        players = self.current_player[batch_indices].long()  # (N,)
        player_colors = (players + 1).to(torch.int8)  # (N,)

        # Get cell coordinates: (N, 5, 2)
        cells, num_cells = self._get_piece_cells_batched(
            piece_indices, orient_indices, rows, cols
        )

        # 1. Place pieces on board using vectorized scatter
        # Create flat indices for scatter: board[batch_idx, row, col]
        # Flatten cells for efficient processing
        cell_mask = torch.arange(5, device=device).unsqueeze(0) < num_cells.unsqueeze(1)  # (N, 5)
        valid_cells = cell_mask & (cells[:, :, 0] >= 0) & (cells[:, :, 0] < BOARD_SIZE) & \
                     (cells[:, :, 1] >= 0) & (cells[:, :, 1] < BOARD_SIZE)

        # Get valid (move_idx, cell_idx) pairs
        valid_move_cell = torch.where(valid_cells)  # (valid_count,), (valid_count,)
        move_idxs = valid_move_cell[0]  # Which move
        cell_idxs = valid_move_cell[1]  # Which cell within move

        if move_idxs.numel() > 0:
            # Get batch indices, rows, cols for valid cells
            bi = batch_indices[move_idxs]
            ri = cells[move_idxs, cell_idxs, 0]
            ci = cells[move_idxs, cell_idxs, 1]
            colors = player_colors[move_idxs]

            # Place pieces using index_put_ (batched scatter)
            self.boards.index_put_((bi, ri, ci), colors)

        # 2. Update remaining pieces
        self.remaining_pieces[batch_indices, players, piece_indices] = False

        # 3. Update has_played
        self.has_played[batch_indices, players] = True

        # 4. Update scores
        self.scores[batch_indices, players] += num_cells.to(torch.int32)

        # 5. Reset consecutive passes
        self.consecutive_passes[batch_indices] = 0

        # 6. Update valid corners - optimized version
        self._update_valid_corners_fast(batch_indices, cells, num_cells, players, valid_cells)

        # 7. Advance turn
        self._advance_turn(batch_indices)

    def _update_valid_corners_fast(
        self,
        batch_indices: torch.Tensor,
        cells: torch.Tensor,
        num_cells: torch.Tensor,
        players: torch.Tensor,
        valid_cells: torch.Tensor,
    ):
        """
        Update valid corner attachment points after placing pieces.
        Optimized vectorized version.
        """
        device = self.device

        # Get valid (move_idx, cell_idx) pairs
        valid_move_cell = torch.where(valid_cells)
        move_idxs = valid_move_cell[0]
        cell_idxs = valid_move_cell[1]

        if move_idxs.numel() == 0:
            return

        # Get batch, row, col, player for each valid cell
        bi = batch_indices[move_idxs]  # (K,)
        ri = cells[move_idxs, cell_idxs, 0]  # (K,)
        ci = cells[move_idxs, cell_idxs, 1]  # (K,)
        pi = players[move_idxs]  # (K,)

        # 1. Remove placed cells from ALL players' valid corners
        # This is vectorized - set all 4 players' corners at these positions to False
        for p in range(NUM_PLAYERS):
            self.valid_corners.index_put_(
                (bi, torch.full_like(bi, p), ri, ci),
                torch.zeros(bi.shape[0], dtype=torch.bool, device=device)
            )

        # 2. Remove edge-adjacent cells from placing player's corners
        for edge_idx in range(4):
            dr = self.piece_data.edge_offsets[edge_idx, 0]
            dc = self.piece_data.edge_offsets[edge_idx, 1]
            nr = ri + dr
            nc = ci + dc

            # Filter to in-bounds neighbors
            in_bounds = (nr >= 0) & (nr < BOARD_SIZE) & (nc >= 0) & (nc < BOARD_SIZE)
            if in_bounds.any():
                valid_bi = bi[in_bounds]
                valid_pi = pi[in_bounds]
                valid_nr = nr[in_bounds]
                valid_nc = nc[in_bounds]

                self.valid_corners.index_put_(
                    (valid_bi, valid_pi, valid_nr, valid_nc),
                    torch.zeros(valid_bi.shape[0], dtype=torch.bool, device=device)
                )

        # 3. Add diagonal cells as new valid corners for placing player
        # This needs more careful checking (not occupied, not edge-adjacent to own color)
        player_colors = (pi + 1).to(torch.int8)

        for diag_idx in range(4):
            dr = self.piece_data.diagonal_offsets[diag_idx, 0]
            dc = self.piece_data.diagonal_offsets[diag_idx, 1]
            nr = ri + dr
            nc = ci + dc

            # Filter to in-bounds
            in_bounds = (nr >= 0) & (nr < BOARD_SIZE) & (nc >= 0) & (nc < BOARD_SIZE)
            if not in_bounds.any():
                continue

            # Get candidates
            cand_bi = bi[in_bounds]
            cand_pi = pi[in_bounds]
            cand_nr = nr[in_bounds]
            cand_nc = nc[in_bounds]
            cand_colors = player_colors[in_bounds]

            # Check not occupied
            cell_values = self.boards[cand_bi, cand_nr, cand_nc]
            not_occupied = cell_values == 0

            if not not_occupied.any():
                continue

            # Filter to unoccupied candidates
            cand_bi = cand_bi[not_occupied]
            cand_pi = cand_pi[not_occupied]
            cand_nr = cand_nr[not_occupied]
            cand_nc = cand_nc[not_occupied]
            cand_colors = cand_colors[not_occupied]

            # Check not edge-adjacent to own color
            not_edge_adjacent = torch.ones(cand_bi.shape[0], dtype=torch.bool, device=device)

            for edge_idx in range(4):
                edr = self.piece_data.edge_offsets[edge_idx, 0]
                edc = self.piece_data.edge_offsets[edge_idx, 1]
                er = cand_nr + edr
                ec = cand_nc + edc

                # For out-of-bounds neighbors, they don't cause edge adjacency
                edge_in_bounds = (er >= 0) & (er < BOARD_SIZE) & (ec >= 0) & (ec < BOARD_SIZE)

                # Get neighbor values (clamp for safety, will be masked anyway)
                er_safe = er.clamp(0, BOARD_SIZE - 1)
                ec_safe = ec.clamp(0, BOARD_SIZE - 1)
                neighbor_vals = self.boards[cand_bi, er_safe, ec_safe]

                # Edge adjacent if in bounds AND neighbor is own color
                is_own_edge = edge_in_bounds & (neighbor_vals == cand_colors)
                not_edge_adjacent = not_edge_adjacent & ~is_own_edge

            if not_edge_adjacent.any():
                final_bi = cand_bi[not_edge_adjacent]
                final_pi = cand_pi[not_edge_adjacent]
                final_nr = cand_nr[not_edge_adjacent]
                final_nc = cand_nc[not_edge_adjacent]

                self.valid_corners.index_put_(
                    (final_bi, final_pi, final_nr, final_nc),
                    torch.ones(final_bi.shape[0], dtype=torch.bool, device=device)
                )

    def _update_valid_corners(
        self,
        batch_indices: torch.Tensor,
        cells: torch.Tensor,
        num_cells: torch.Tensor,
        players: torch.Tensor,
    ):
        """Update valid corner attachment points after placing pieces (legacy non-vectorized version)."""
        n = batch_indices.shape[0]
        device = self.device

        for cell_idx in range(5):
            is_valid_cell = cell_idx < num_cells
            r = cells[:, cell_idx, 0]
            c = cells[:, cell_idx, 1]

            # Skip invalid cells
            valid_mask = is_valid_cell & (r >= 0) & (r < BOARD_SIZE) & (c >= 0) & (c < BOARD_SIZE)
            if not valid_mask.any():
                continue

            valid_indices = torch.where(valid_mask)[0]

            for idx in valid_indices:
                bi = batch_indices[idx]
                ri, ci = r[idx].item(), c[idx].item()
                player = players[idx].item()

                # Remove this cell from all players' valid corners
                self.valid_corners[bi, :, ri, ci] = False

                # Remove edge-adjacent cells from placing player's corners
                for edge_idx in range(4):
                    dr = self.piece_data.edge_offsets[edge_idx, 0].item()
                    dc = self.piece_data.edge_offsets[edge_idx, 1].item()
                    nr, nc = ri + dr, ci + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        self.valid_corners[bi, player, nr, nc] = False

                # Add diagonal cells as new valid corners for placing player
                player_color = player + 1
                for diag_idx in range(4):
                    dr = self.piece_data.diagonal_offsets[diag_idx, 0].item()
                    dc = self.piece_data.diagonal_offsets[diag_idx, 1].item()
                    nr, nc = ri + dr, ci + dc

                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                        continue
                    if self.boards[bi, nr, nc] != 0:
                        continue  # Already occupied

                    # Check not edge-adjacent to own color
                    is_edge_adjacent = False
                    for edge_idx in range(4):
                        edr = self.piece_data.edge_offsets[edge_idx, 0].item()
                        edc = self.piece_data.edge_offsets[edge_idx, 1].item()
                        er, ec = nr + edr, nc + edc
                        if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
                            if self.boards[bi, er, ec] == player_color:
                                is_edge_adjacent = True
                                break

                    if not is_edge_adjacent:
                        self.valid_corners[bi, player, nr, nc] = True

    def _advance_turn(self, batch_indices: torch.Tensor):
        """Advance to next player for specified games."""
        for bi in batch_indices:
            bi = bi.item()

            next_player = (self.current_player[bi].item() + 1) % NUM_PLAYERS
            checked = 0

            while checked < NUM_PLAYERS:
                if not self.is_eliminated[bi, next_player]:
                    break
                next_player = (next_player + 1) % NUM_PLAYERS
                checked += 1

            if checked >= NUM_PLAYERS or self.consecutive_passes[bi] >= NUM_PLAYERS:
                self.is_game_over[bi] = True
                return

            self.current_player[bi] = next_player
            self.turn_number[bi] += 1

    def pass_turn(self, batch_indices: torch.Tensor):
        """Pass turn for specified games."""
        for bi in batch_indices:
            bi = bi.item()
            if self.is_game_over[bi]:
                continue

            player = self.current_player[bi].item()
            self.is_eliminated[bi, player] = True
            self.consecutive_passes[bi] += 1

        self._advance_turn(batch_indices)

    @classmethod
    def from_cpu_game(cls, game: BlokusGame, device: str = 'cuda') -> 'BlokusGameBatchGPU':
        """Convert a single CPU game to GPU batch of size 1."""
        batch = cls(batch_size=1, device=device)

        # Copy board
        batch.boards[0] = torch.from_numpy(game.board.astype(np.int8)).to(device)

        # Copy player states
        for player in range(NUM_PLAYERS):
            ps = game.players[player]
            for piece_idx in range(NUM_PIECES):
                batch.remaining_pieces[0, player, piece_idx] = piece_idx in ps.remaining_pieces
            batch.has_played[0, player] = ps.has_played
            batch.is_eliminated[0, player] = ps.is_eliminated
            batch.scores[0, player] = ps.score

            # Copy valid corners
            batch.valid_corners[0, player] = False
            for r, c in game._valid_corners[player]:
                batch.valid_corners[0, player, r, c] = True

        batch.current_player[0] = game.current_player
        batch.turn_number[0] = game.turn_number
        batch.consecutive_passes[0] = game.consecutive_passes
        batch.is_game_over[0] = game.is_game_over

        return batch

    @classmethod
    def from_cpu_games(cls, games: List[BlokusGame], device: str = 'cuda') -> 'BlokusGameBatchGPU':
        """Convert multiple CPU games to a GPU batch."""
        n = len(games)
        batch = cls(batch_size=n, device=device)

        for i, game in enumerate(games):
            batch.boards[i] = torch.from_numpy(game.board.astype(np.int8)).to(device)

            for player in range(NUM_PLAYERS):
                ps = game.players[player]
                for piece_idx in range(NUM_PIECES):
                    batch.remaining_pieces[i, player, piece_idx] = piece_idx in ps.remaining_pieces
                batch.has_played[i, player] = ps.has_played
                batch.is_eliminated[i, player] = ps.is_eliminated
                batch.scores[i, player] = ps.score

                batch.valid_corners[i, player] = False
                for r, c in game._valid_corners[player]:
                    batch.valid_corners[i, player, r, c] = True

            batch.current_player[i] = game.current_player
            batch.turn_number[i] = game.turn_number
            batch.consecutive_passes[i] = game.consecutive_passes
            batch.is_game_over[i] = game.is_game_over

        return batch

    def to_cpu_game(self, idx: int = 0) -> BlokusGame:
        """Convert a single game state back to CPU BlokusGame."""
        game = BlokusGame.__new__(BlokusGame)

        # Copy board
        game.board = self.boards[idx].cpu().numpy()

        # Create player states
        game.players = []
        for player in range(NUM_PLAYERS):
            remaining = set()
            for piece_idx in range(NUM_PIECES):
                if self.remaining_pieces[idx, player, piece_idx]:
                    remaining.add(piece_idx)

            ps = PlayerState(
                remaining_pieces=remaining,
                has_played=self.has_played[idx, player].item(),
                is_eliminated=self.is_eliminated[idx, player].item(),
                score=self.scores[idx, player].item(),
            )
            game.players.append(ps)

        # Copy valid corners
        game._valid_corners = []
        for player in range(NUM_PLAYERS):
            corners = set()
            corner_mask = self.valid_corners[idx, player].cpu().numpy()
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if corner_mask[r, c]:
                        corners.add((r, c))
            game._valid_corners.append(corners)

        game.current_player = self.current_player[idx].item()
        game.turn_number = self.turn_number[idx].item()
        game.consecutive_passes = self.consecutive_passes[idx].item()
        game.is_game_over = self.is_game_over[idx].item()
        game.move_history = []  # Cannot recover move history

        return game

    def get_state_for_neural_net(
        self,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get state encoding for neural network input.

        Args:
            batch_indices: Optional (N,) tensor of which states to encode.
                          If None, encode all states.

        Returns:
            board_tensor: (N, 8, 20, 20) float tensor
            pieces_vector: (N, 84) float tensor
        """
        if batch_indices is None:
            batch_indices = torch.arange(self.batch_size, device=self.device)

        n = batch_indices.shape[0]

        board_tensor = torch.zeros(n, 8, BOARD_SIZE, BOARD_SIZE,
                                   dtype=torch.float32, device=self.device)
        pieces_vector = torch.zeros(n, NUM_PIECES * NUM_PLAYERS,
                                    dtype=torch.float32, device=self.device)

        boards_selected = self.boards[batch_indices]  # (N, 20, 20)
        players = self.current_player[batch_indices].long()  # (N,)

        for i in range(n):
            player = players[i].item()
            board = boards_selected[i]

            # Player order from perspective of current player
            player_order = [(player + j) % NUM_PLAYERS for j in range(NUM_PLAYERS)]

            # Channel 0: Current player's pieces
            board_tensor[i, 0] = (board == player_order[0] + 1).float()

            # Channels 1-3: Other players' pieces
            for j in range(1, NUM_PLAYERS):
                board_tensor[i, j] = (board == player_order[j] + 1).float()

            # Channel 4: Valid corners for current player
            board_tensor[i, 4] = self.valid_corners[batch_indices[i], player].float()

            # Channel 5: Empty cells
            board_tensor[i, 5] = (board == 0).float()

            # Channel 6: Turn number (normalized)
            board_tensor[i, 6] = min(self.turn_number[batch_indices[i]].item() / 80.0, 1.0)

            # Channel 7: Current player indicator
            board_tensor[i, 7] = (player + 1) / 4.0

            # Remaining pieces vector
            for j, p_idx in enumerate(player_order):
                for piece_idx in range(NUM_PIECES):
                    if self.remaining_pieces[batch_indices[i], p_idx, piece_idx]:
                        pieces_vector[i, j * NUM_PIECES + piece_idx] = 1.0

        return board_tensor, pieces_vector


def test_gpu_vs_cpu():
    """Test that GPU implementation matches CPU implementation."""
    import random

    # Test on CPU if CUDA not available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    # Create games
    cpu_game = BlokusGame()
    gpu_batch = BlokusGameBatchGPU.from_cpu_game(cpu_game, device)

    # Verify initial state matches
    gpu_game = gpu_batch.to_cpu_game(0)
    assert np.array_equal(cpu_game.board, gpu_game.board), "Initial boards don't match"
    print("Initial state matches")

    # Play some random moves
    for move_num in range(20):
        legal_moves = cpu_game.get_legal_moves()
        if not legal_moves:
            cpu_game.pass_turn()
            gpu_batch.pass_turn(torch.tensor([0], device=device))
            continue

        move = random.choice(legal_moves)

        # Apply to CPU
        cpu_game.make_move(move)

        # Apply to GPU
        gpu_batch.apply_moves(
            batch_indices=torch.tensor([0], device=device),
            piece_indices=torch.tensor([move.piece_idx], device=device),
            orient_indices=torch.tensor([move.orientation_idx], device=device),
            rows=torch.tensor([move.row], device=device),
            cols=torch.tensor([move.col], device=device),
        )

        # Verify states match
        gpu_game = gpu_batch.to_cpu_game(0)

        if not np.array_equal(cpu_game.board, gpu_game.board):
            print(f"Move {move_num}: Board mismatch!")
            print(f"CPU board:\n{cpu_game.board}")
            print(f"GPU board:\n{gpu_game.board}")
            return False

        if cpu_game.current_player != gpu_game.current_player:
            print(f"Move {move_num}: Current player mismatch: CPU={cpu_game.current_player}, GPU={gpu_game.current_player}")
            return False

        print(f"Move {move_num}: OK (player {move.player}, piece {move.piece_idx})")

    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    test_gpu_vs_cpu()
