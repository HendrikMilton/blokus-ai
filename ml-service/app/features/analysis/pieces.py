"""
All 21 Blokus pieces with their cell coordinates.

Each piece is defined as a list of (row, col) offsets from the origin (0, 0).
Pieces are normalized so the first cell is always at (0, 0).
"""


# All 21 Blokus pieces
# Format: name -> list of (row, col) cells
PIECES = {
    # 1-square (monomino)
    "I1": [(0, 0)],

    # 2-square (domino)
    "I2": [(0, 0), (0, 1)],

    # 3-square (trominoes)
    "I3": [(0, 0), (0, 1), (0, 2)],
    "V3": [(0, 0), (0, 1), (1, 0)],

    # 4-square (tetrominoes)
    "I4": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "L4": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "T4": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "O4": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "Z4": [(0, 0), (0, 1), (1, 1), (1, 2)],

    # 5-square (pentominoes) - 12 pieces
    "I5": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    "L5": [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    "Y": [(0, 0), (1, 0), (1, 1), (2, 0), (3, 0)],
    "N": [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)],
    "P": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    "U": [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
    "V5": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    "W": [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    "Z5": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    "T5": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    "F": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "X": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
}

# Piece sizes for heuristics
PIECE_SIZES = {name: len(cells) for name, cells in PIECES.items()}


def normalize_piece(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Normalize piece so minimum row and col are 0, return as sorted tuple."""
    if not cells:
        return ()
    min_row = min(c[0] for c in cells)
    min_col = min(c[1] for c in cells)
    normalized = [(c[0] - min_row, c[1] - min_col) for c in cells]
    return tuple(sorted(normalized))


def rotate_90(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Rotate piece 90 degrees clockwise."""
    # (row, col) -> (col, -row)
    return [(c[1], -c[0]) for c in cells]


def flip_horizontal(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Flip piece horizontally."""
    return [(c[0], -c[1]) for c in cells]


def get_all_orientations(cells: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    """Get all unique orientations (rotations + flips) of a piece."""
    orientations = set()
    current = list(cells)

    # 4 rotations
    for _ in range(4):
        orientations.add(normalize_piece(current))
        current = rotate_90(current)

    # Flip and 4 more rotations
    current = flip_horizontal(cells)
    for _ in range(4):
        orientations.add(normalize_piece(current))
        current = rotate_90(current)

    return list(orientations)


# Precompute all orientations for each piece
PIECE_ORIENTATIONS: dict[str, list[tuple[tuple[int, int], ...]]] = {
    name: get_all_orientations(list(cells))
    for name, cells in PIECES.items()
}


def get_rotation_degrees(original: list[tuple[int, int]], oriented: tuple[tuple[int, int], ...]) -> int:
    """Determine rotation in degrees (0, 90, 180, 270) for an orientation."""
    # This is approximate - we just check which rotation matches
    current = list(original)
    for degrees in [0, 90, 180, 270]:
        if normalize_piece(current) == oriented:
            return degrees
        current = rotate_90(current)

    # Check flipped versions
    current = flip_horizontal(original)
    for degrees in [0, 90, 180, 270]:
        if normalize_piece(current) == oriented:
            return degrees
        current = rotate_90(current)

    return 0  # Default
