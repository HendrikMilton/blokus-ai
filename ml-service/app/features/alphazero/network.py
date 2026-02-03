"""
Neural network for AlphaZero-style Blokus AI.

Architecture:
- Residual tower for board feature extraction
- Policy head: scores legal moves via attention mechanism
- Value head: predicts game outcome [-1, 1]

Optimized for RTX 5090 (32GB VRAM) with large batch sizes.
Uses fully vectorized move encoding for maximum GPU utilization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np

from .pieces import NUM_PIECES, PIECE_CATALOG


class ResBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class MoveEncoderVectorized(nn.Module):
    """
    Fully vectorized move encoder for GPU-efficient batched inference.

    Pre-computes piece/orientation encodings at init time, then uses
    tensor indexing for O(1) encoding lookup instead of Python loops.

    Move encoding (41-dim):
    - Piece ID: 21-dim one-hot
    - Orientation: 8-dim one-hot
    - Position: 2-dim normalized (row/20, col/20)
    - Piece shape: 10-dim (5 cells × 2 coords, normalized)
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.input_dim = 41
        self.embed_dim = embed_dim

        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Pre-compute static encodings for piece_id and orientation one-hots
        # and shape encodings for all (piece, orientation) combinations
        self._precompute_static_encodings()

    def _precompute_static_encodings(self):
        """Pre-compute all static parts of move encodings."""
        # Piece one-hot lookup: (21, 21)
        self.register_buffer('piece_onehot', torch.eye(NUM_PIECES, dtype=torch.float32))

        # Orientation one-hot lookup: (8, 8)
        self.register_buffer('orient_onehot', torch.eye(8, dtype=torch.float32))

        # Shape encodings for all (piece_idx, orient_idx) combinations: (21, 8, 10)
        shape_encodings = torch.zeros(NUM_PIECES, 8, 10, dtype=torch.float32)
        for piece_idx, piece in enumerate(PIECE_CATALOG):
            for orient_idx, shape in enumerate(piece.orientations):
                for i, (r, c) in enumerate(shape[:5]):
                    shape_encodings[piece_idx, orient_idx, i * 2] = r / 5.0
                    shape_encodings[piece_idx, orient_idx, i * 2 + 1] = c / 5.0
        self.register_buffer('shape_encodings', shape_encodings)

    def encode_moves_vectorized(
        self,
        piece_indices: torch.Tensor,    # (N,) int64
        orient_indices: torch.Tensor,   # (N,) int64
        rows: torch.Tensor,             # (N,) float32
        cols: torch.Tensor,             # (N,) float32
    ) -> torch.Tensor:
        """
        Encode moves using pure tensor operations - no Python loops.

        Args:
            piece_indices: Piece IDs for each move
            orient_indices: Orientation IDs for each move
            rows: Row positions (0-19) for each move
            cols: Column positions (0-19) for each move

        Returns:
            Tensor of shape (N, 41) - encoded moves
        """
        N = piece_indices.shape[0]
        if N == 0:
            return torch.zeros((0, self.input_dim), dtype=torch.float32, device=piece_indices.device)

        # Gather one-hot encodings using advanced indexing
        piece_enc = self.piece_onehot[piece_indices]  # (N, 21)
        orient_enc = self.orient_onehot[orient_indices]  # (N, 8)

        # Normalized positions
        pos_enc = torch.stack([rows / 20.0, cols / 20.0], dim=1)  # (N, 2)

        # Gather shape encodings
        shape_enc = self.shape_encodings[piece_indices, orient_indices]  # (N, 10)

        # Concatenate all parts
        return torch.cat([piece_enc, orient_enc, pos_enc, shape_enc], dim=1)  # (N, 41)

    def forward(self, move_encodings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            move_encodings: Tensor of shape (N, 41)

        Returns:
            Tensor of shape (N, embed_dim)
        """
        return self.embed(move_encodings)


class MoveEncoder(nn.Module):
    """
    Legacy move encoder (for backward compatibility).
    Use MoveEncoderVectorized for batched inference.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.input_dim = 41
        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode_moves(self, moves: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """Encode moves (legacy Python loop version)."""
        if not moves:
            return torch.zeros((0, self.input_dim), dtype=torch.float32)

        encodings = []
        for piece_idx, orient_idx, row, col in moves:
            piece = PIECE_CATALOG[piece_idx]

            piece_onehot = torch.zeros(NUM_PIECES, dtype=torch.float32)
            piece_onehot[piece_idx] = 1.0

            orient_onehot = torch.zeros(8, dtype=torch.float32)
            orient_onehot[orient_idx] = 1.0

            pos = torch.tensor([row / 20.0, col / 20.0], dtype=torch.float32)

            shape = piece.orientations[orient_idx]
            shape_encoding = torch.zeros(10, dtype=torch.float32)
            for i, (r, c) in enumerate(shape[:5]):
                shape_encoding[i * 2] = r / 5.0
                shape_encoding[i * 2 + 1] = c / 5.0

            encoding = torch.cat([piece_onehot, orient_onehot, pos, shape_encoding])
            encodings.append(encoding)

        return torch.stack(encodings)

    def forward(self, move_encodings: torch.Tensor) -> torch.Tensor:
        return self.embed(move_encodings)


class BlokusNet(nn.Module):
    """
    AlphaZero-style neural network for Blokus.

    Input:
    - board_tensor: (batch, 10, 20, 20) - board state encoding
        Channel 0: Current player's pieces
        Channel 1-3: Other players' pieces (in turn order)
        Channel 4: Valid corner attachment points
        Channel 5: Empty cells
        Channel 6: Turn number (normalized)
        Channel 7: Current player indicator
        Channel 8: Center distance heatmap (closer to center = higher value)
        Channel 9: Game phase (0=early, 0.5=mid, 1=late)
    - pieces_vector: (batch, 84) - remaining pieces for all players

    Output:
    - policy_logits: (batch, num_legal_moves) - scores for each legal move
    - value: (batch, 1) - predicted game outcome [-1, 1]

    Uses fully vectorized move encoding and batched attention for
    maximum GPU utilization during batched inference.
    """

    # Number of input channels for the board tensor
    INPUT_CHANNELS = 10

    def __init__(
        self,
        num_res_blocks: int = 15,
        channels: int = 256,
        policy_embed_dim: int = 128,
        value_hidden: int = 256,
    ):
        super().__init__()

        self.channels = channels
        self.policy_embed_dim = policy_embed_dim

        # Stem: initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(self.INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Residual tower
        self.res_tower = nn.Sequential(*[
            ResBlock(channels) for _ in range(num_res_blocks)
        ])

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Combine board features with pieces vector
        self.fc_combine = nn.Sequential(
            nn.Linear(channels + 84, value_hidden * 2),
            nn.ReLU(),
        )

        # Policy head - vectorized encoder for GPU efficiency
        self.move_encoder = MoveEncoderVectorized(embed_dim=policy_embed_dim)

        # Legacy encoder for backward compatibility
        self.move_encoder_legacy = MoveEncoder(embed_dim=policy_embed_dim)

        # Project combined features to policy embedding space for attention
        self.policy_query = nn.Linear(value_hidden * 2, policy_embed_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(value_hidden * 2, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def _extract_move_tensors(
        self,
        legal_moves_list: List[List],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Extract move components into flat tensors for vectorized encoding.

        Returns:
            piece_indices: (total_moves,) int64
            orient_indices: (total_moves,) int64
            rows: (total_moves,) float32
            cols: (total_moves,) float32
            move_counts: List of move counts per batch item
        """
        move_counts = [len(moves) for moves in legal_moves_list]
        total_moves = sum(move_counts)

        if total_moves == 0:
            return (
                torch.tensor([], dtype=torch.int64, device=device),
                torch.tensor([], dtype=torch.int64, device=device),
                torch.tensor([], dtype=torch.float32, device=device),
                torch.tensor([], dtype=torch.float32, device=device),
                move_counts,
            )

        # Pre-allocate numpy arrays
        all_pieces = np.empty(total_moves, dtype=np.int64)
        all_orients = np.empty(total_moves, dtype=np.int64)
        all_rows = np.empty(total_moves, dtype=np.float32)
        all_cols = np.empty(total_moves, dtype=np.float32)

        idx = 0
        for moves in legal_moves_list:
            for m in moves:
                if hasattr(m, 'piece_idx'):
                    all_pieces[idx] = m.piece_idx
                    all_orients[idx] = m.orientation_idx
                    all_rows[idx] = m.row
                    all_cols[idx] = m.col
                else:
                    # Tuple format (piece_idx, orient_idx, row, col)
                    all_pieces[idx] = m[0]
                    all_orients[idx] = m[1]
                    all_rows[idx] = m[2]
                    all_cols[idx] = m[3]
                idx += 1

        return (
            torch.from_numpy(all_pieces).to(device),
            torch.from_numpy(all_orients).to(device),
            torch.from_numpy(all_rows).to(device),
            torch.from_numpy(all_cols).to(device),
            move_counts,
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        pieces_vector: torch.Tensor,
        legal_moves_list: Optional[List[List[Tuple[int, int, int, int]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with fully vectorized policy computation.

        Args:
            board_tensor: (batch, 8, 20, 20)
            pieces_vector: (batch, 84)
            legal_moves_list: List of legal moves per batch item

        Returns:
            policy_logits: (batch, max_num_moves) - padded with -inf
            value: (batch, 1)
        """
        batch_size = board_tensor.size(0)
        device = board_tensor.device

        # Board feature extraction
        x = self.stem(board_tensor)
        x = self.res_tower(x)
        x = self.global_pool(x).view(batch_size, -1)

        # Combine with pieces vector
        combined = torch.cat([x, pieces_vector], dim=1)
        features = self.fc_combine(combined)

        # Value prediction
        value = self.value_head(features)

        # Policy prediction
        if legal_moves_list is None:
            policy_logits = torch.zeros(batch_size, 1, device=device)
        else:
            policy_logits = self._compute_policy_vectorized(features, legal_moves_list, device)

        return policy_logits, value

    def _compute_policy_vectorized(
        self,
        features: torch.Tensor,
        legal_moves_list: List[List],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute policy logits using fully vectorized operations.

        Instead of looping over batch items, this:
        1. Flattens all moves into single tensors
        2. Encodes all moves in one GPU operation
        3. Embeds all moves in one GPU operation
        4. Computes attention scores using batched operations
        """
        batch_size = features.size(0)

        # Get query vectors for all positions
        query = self.policy_query(features)  # (batch, embed_dim)

        # Extract all moves into flat tensors
        piece_idx, orient_idx, rows, cols, move_counts = self._extract_move_tensors(
            legal_moves_list, device
        )

        total_moves = piece_idx.shape[0]
        max_moves = max(move_counts) if move_counts else 1
        max_moves = max(max_moves, 1)

        if total_moves == 0:
            # Initialize output with -inf (illegal moves)
            return torch.full((batch_size, max_moves), float('-inf'), device=device)

        # Encode ALL moves in one vectorized operation (no Python loops!)
        move_encodings = self.move_encoder.encode_moves_vectorized(
            piece_idx, orient_idx, rows, cols
        )  # (total_moves, 41)

        # Embed ALL moves in one GPU operation
        move_embeddings = self.move_encoder(move_encodings)  # (total_moves, embed_dim)

        # Initialize output with -inf (illegal moves) - use same dtype as embeddings for mixed precision
        policy_logits = torch.full(
            (batch_size, max_moves), float('-inf'),
            device=device, dtype=move_embeddings.dtype
        )

        # Compute attention scores using batched matrix multiplication
        # Pad move embeddings into (batch_size, max_moves, embed_dim) tensor
        # Use same dtype as move_embeddings to handle mixed precision (float16/float32)
        embed_dim = move_embeddings.size(1)
        padded_embeddings = torch.zeros(
            batch_size, max_moves, embed_dim,
            device=device, dtype=move_embeddings.dtype
        )

        # Create indices for scatter - this avoids the Python loop
        # Build batch indices and position indices for all moves
        batch_indices = []
        pos_indices = []
        offset = 0
        for i, count in enumerate(move_counts):
            if count > 0:
                batch_indices.extend([i] * count)
                pos_indices.extend(range(count))
            offset += count

        if batch_indices:
            batch_idx = torch.tensor(batch_indices, dtype=torch.long, device=device)
            pos_idx = torch.tensor(pos_indices, dtype=torch.long, device=device)
            padded_embeddings[batch_idx, pos_idx] = move_embeddings

        # Batched attention: (batch, 1, embed) @ (batch, embed, max_moves) -> (batch, 1, max_moves)
        scores = torch.bmm(query.unsqueeze(1), padded_embeddings.transpose(1, 2)).squeeze(1)

        # Create validity mask and apply scores
        move_counts_t = torch.tensor(move_counts, device=device)
        position_indices = torch.arange(max_moves, device=device).unsqueeze(0)  # (1, max_moves)
        valid_mask = position_indices < move_counts_t.unsqueeze(1)  # (batch, max_moves)

        # Copy scores where valid, keep -inf elsewhere
        policy_logits[valid_mask] = scores[valid_mask]

        return policy_logits

    def forward_fully_batched(
        self,
        board_tensor: torch.Tensor,
        pieces_vector: torch.Tensor,
        legal_moves_list: List[List],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fully batched forward pass using padded move tensors and bmm.

        Uses numpy for fast move tensor creation, then GPU for all neural ops.
        """
        batch_size = board_tensor.size(0)
        device = board_tensor.device

        # Board feature extraction (GPU)
        x = self.stem(board_tensor)
        x = self.res_tower(x)
        x = self.global_pool(x).view(batch_size, -1)

        combined = torch.cat([x, pieces_vector], dim=1)
        features = self.fc_combine(combined)
        value = self.value_head(features)

        # Query vectors
        query = self.policy_query(features)  # (batch, embed_dim)

        # Fast move tensor creation using numpy
        move_counts = [len(moves) for moves in legal_moves_list]
        max_moves = max(move_counts) if move_counts else 1
        max_moves = max(max_moves, 1)

        # Pre-allocate numpy arrays (much faster than torch tensor item assignment)
        all_piece_idx = np.zeros((batch_size, max_moves), dtype=np.int64)
        all_orient_idx = np.zeros((batch_size, max_moves), dtype=np.int64)
        all_rows = np.zeros((batch_size, max_moves), dtype=np.float32)
        all_cols = np.zeros((batch_size, max_moves), dtype=np.float32)
        mask = np.zeros((batch_size, max_moves), dtype=bool)

        # Fill using numpy (faster than torch tensor assignment)
        for i, moves in enumerate(legal_moves_list):
            n = len(moves)
            if n == 0:
                continue
            mask[i, :n] = True

            # Check if moves are objects or tuples
            if moves and hasattr(moves[0], 'piece_idx'):
                # Object format - extract attributes
                for j, m in enumerate(moves):
                    all_piece_idx[i, j] = m.piece_idx
                    all_orient_idx[i, j] = m.orientation_idx
                    all_rows[i, j] = m.row
                    all_cols[i, j] = m.col
            else:
                # Tuple format - use numpy array conversion for speed
                move_arr = np.array(moves, dtype=np.float32)
                all_piece_idx[i, :n] = move_arr[:, 0].astype(np.int64)
                all_orient_idx[i, :n] = move_arr[:, 1].astype(np.int64)
                all_rows[i, :n] = move_arr[:, 2]
                all_cols[i, :n] = move_arr[:, 3]

        # Convert to GPU tensors in one shot
        all_piece_idx = torch.from_numpy(all_piece_idx).to(device)
        all_orient_idx = torch.from_numpy(all_orient_idx).to(device)
        all_rows = torch.from_numpy(all_rows).to(device)
        all_cols = torch.from_numpy(all_cols).to(device)
        mask = torch.from_numpy(mask).to(device)

        # Flatten for vectorized encoding
        flat_piece = all_piece_idx.view(-1)
        flat_orient = all_orient_idx.view(-1)
        flat_rows = all_rows.view(-1)
        flat_cols = all_cols.view(-1)

        # Encode all moves at once (GPU)
        move_encodings = self.move_encoder.encode_moves_vectorized(
            flat_piece, flat_orient, flat_rows, flat_cols
        )

        # Embed all moves at once (GPU)
        move_embeddings = self.move_encoder(move_encodings)

        # Reshape for batched attention
        move_embeddings = move_embeddings.view(batch_size, max_moves, self.policy_embed_dim)

        # Batched attention using bmm (GPU)
        query_expanded = query.unsqueeze(1)
        keys_transposed = move_embeddings.transpose(1, 2)
        policy_logits = torch.bmm(query_expanded, keys_transposed).squeeze(1)

        # Mask out invalid moves
        policy_logits = policy_logits.masked_fill(~mask, float('-inf'))

        return policy_logits, value

    def predict(
        self,
        board_tensor: np.ndarray,
        pieces_vector: np.ndarray,
        legal_moves: List,
        device: str = 'cuda',
    ) -> Tuple[np.ndarray, float]:
        """
        Get policy and value predictions for a single position.

        Args:
            board_tensor: (8, 20, 20) numpy array
            pieces_vector: (84,) numpy array
            legal_moves: List of Move objects or tuples
            device: torch device

        Returns:
            policy: (num_moves,) numpy array of probabilities
            value: float in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            bt = torch.from_numpy(board_tensor).float().unsqueeze(0).to(device)
            pv = torch.from_numpy(pieces_vector).float().unsqueeze(0).to(device)

            policy_logits, value = self.forward(bt, pv, [legal_moves])

            # Convert logits to probabilities
            policy = F.softmax(policy_logits[0, :len(legal_moves)], dim=0)

            return policy.cpu().numpy(), value.item()

    def predict_batch(
        self,
        board_tensors: List[np.ndarray],
        pieces_vectors: List[np.ndarray],
        legal_moves_list: List[List],
        device: str = 'cuda',
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Evaluate multiple positions in one GPU batch for efficient inference.

        Uses fully vectorized move encoding and batched attention (torch.bmm)
        to maximize GPU utilization. This is the key method for batched MCTS.

        Args:
            board_tensors: List of (8, 20, 20) numpy arrays
            pieces_vectors: List of (84,) numpy arrays
            legal_moves_list: List of legal moves for each position
            device: torch device

        Returns:
            policies: List of probability arrays, one per position
            values: List of floats in [-1, 1]
        """
        if not board_tensors:
            return [], []

        batch_size = len(board_tensors)
        self.eval()

        with torch.no_grad():
            # Stack into batch tensors on GPU
            bt = torch.stack([
                torch.from_numpy(b).float() for b in board_tensors
            ]).to(device)
            pv = torch.stack([
                torch.from_numpy(p).float() for p in pieces_vectors
            ]).to(device)

            # Use fully batched forward pass with bmm
            policy_logits, values = self.forward_fully_batched(bt, pv, legal_moves_list)

            # Extract individual policies
            policies = []
            for i in range(batch_size):
                num_moves = len(legal_moves_list[i])
                if num_moves > 0:
                    policy = F.softmax(policy_logits[i, :num_moves], dim=0)
                    policies.append(policy.cpu().numpy())
                else:
                    policies.append(np.array([], dtype=np.float32))

            # Extract values as list
            values_list = values.squeeze(-1).cpu().tolist()

        return policies, values_list

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = 'cuda'):
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location=device))


def create_network(
    num_res_blocks: int = 15,
    channels: int = 256,
    device: str = 'cuda',
) -> BlokusNet:
    """Create a new network with specified architecture."""
    net = BlokusNet(
        num_res_blocks=num_res_blocks,
        channels=channels,
    )
    return net.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
