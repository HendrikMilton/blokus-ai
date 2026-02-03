"""
Blokus piece definitions and orientation generation.
All 21 standard Blokus pieces with pre-computed orientations.
"""

from typing import List, Tuple, Set
import numpy as np

# Coordinate type: (row, col) relative to piece origin
Coord = Tuple[int, int]
PieceShape = List[Coord]


# Base shapes for all 21 Blokus pieces
# Coordinates are [row, col] relative to top-left of bounding box
BASE_PIECES = [
    # 1 cell (monomino)
    ('1', [(0, 0)]),

    # 2 cells (domino)
    ('2', [(0, 0), (0, 1)]),

    # 3 cells (triominoes)
    ('I3', [(0, 0), (0, 1), (0, 2)]),
    ('V3', [(0, 0), (0, 1), (1, 0)]),

    # 4 cells (tetrominoes)
    ('I4', [(0, 0), (0, 1), (0, 2), (0, 3)]),
    ('O4', [(0, 0), (0, 1), (1, 0), (1, 1)]),
    ('T4', [(0, 0), (0, 1), (0, 2), (1, 1)]),
    ('L4', [(0, 0), (1, 0), (2, 0), (2, 1)]),
    ('S4', [(0, 1), (0, 2), (1, 0), (1, 1)]),

    # 5 cells (pentominoes)
    ('F', [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]),
    ('I5', [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]),
    ('L5', [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)]),
    ('N', [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)]),
    ('P', [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]),
    ('T5', [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)]),
    ('U', [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]),
    ('V', [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]),
    ('W', [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]),
    ('X', [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]),
    ('Y', [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3)]),
    ('Z', [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]),
]


def normalize_shape(shape: PieceShape) -> PieceShape:
    """
    Normalize a shape by translating it so the minimum row and col are 0,
    then sorting coordinates for consistent comparison.
    """
    if not shape:
        return []

    min_row = min(r for r, c in shape)
    min_col = min(c for r, c in shape)

    normalized = [(r - min_row, c - min_col) for r, c in shape]
    normalized.sort()
    return normalized


def rotate_shape(shape: PieceShape) -> PieceShape:
    """
    Rotate shape 90 degrees clockwise.
    [row, col] -> [col, -row] (then normalize)
    """
    rotated = [(c, -r) for r, c in shape]
    return normalize_shape(rotated)


def flip_shape(shape: PieceShape) -> PieceShape:
    """
    Flip shape horizontally (mirror).
    [row, col] -> [row, -col] (then normalize)
    """
    flipped = [(r, -c) for r, c in shape]
    return normalize_shape(flipped)


def shape_to_tuple(shape: PieceShape) -> Tuple[Coord, ...]:
    """Convert shape to tuple for hashing."""
    return tuple(shape)


def generate_orientations(base_shape: PieceShape) -> List[PieceShape]:
    """
    Generate all unique orientations (up to 8) for a shape.
    """
    seen: Set[Tuple[Coord, ...]] = set()
    orientations: List[PieceShape] = []

    current = normalize_shape(base_shape)

    # Try all 4 rotations
    for _ in range(4):
        key = shape_to_tuple(current)
        if key not in seen:
            seen.add(key)
            orientations.append(current)
        current = rotate_shape(current)

    # Flip and try all 4 rotations again
    current = flip_shape(normalize_shape(base_shape))
    for _ in range(4):
        key = shape_to_tuple(current)
        if key not in seen:
            seen.add(key)
            orientations.append(current)
        current = rotate_shape(current)

    return orientations


class PieceDefinition:
    """A Blokus piece with all orientations pre-computed."""

    def __init__(self, piece_id: str, base_shape: PieceShape):
        self.id = piece_id
        self.size = len(base_shape)
        self.orientations = generate_orientations(base_shape)
        self.num_orientations = len(self.orientations)

        # Pre-compute numpy arrays for each orientation for fast operations
        self._numpy_orientations = [
            np.array(orient, dtype=np.int8) for orient in self.orientations
        ]

    def get_cells(self, orientation_idx: int, row: int, col: int) -> np.ndarray:
        """
        Get absolute board coordinates for piece at given position.
        Returns array of shape (size, 2) with [row, col] coordinates.
        """
        base = self._numpy_orientations[orientation_idx]
        offset = np.array([[row, col]], dtype=np.int8)
        return base + offset

    def __repr__(self):
        return f"PieceDefinition({self.id}, size={self.size}, orientations={self.num_orientations})"


# Build the piece catalog
PIECE_CATALOG: List[PieceDefinition] = [
    PieceDefinition(piece_id, shape) for piece_id, shape in BASE_PIECES
]

# Maps for quick lookup
PIECE_BY_ID = {p.id: p for p in PIECE_CATALOG}
PIECE_BY_INDEX = {i: p for i, p in enumerate(PIECE_CATALOG)}
PIECE_INDEX_BY_ID = {p.id: i for i, p in enumerate(PIECE_CATALOG)}

# All piece IDs in order
ALL_PIECE_IDS = [p.id for p in PIECE_CATALOG]
NUM_PIECES = len(PIECE_CATALOG)

# Total unique orientations across all pieces
TOTAL_ORIENTATIONS = sum(p.num_orientations for p in PIECE_CATALOG)


def get_piece(piece_id: str) -> PieceDefinition:
    """Get piece by ID."""
    return PIECE_BY_ID[piece_id]


def get_piece_by_index(idx: int) -> PieceDefinition:
    """Get piece by index (0-20)."""
    return PIECE_BY_INDEX[idx]
