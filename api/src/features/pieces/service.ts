import type { GridLabels, PlayerColor } from '@blokus-ai/shared';

// Coordinate in a piece (relative to origin)
export type Coord = [number, number];

// A piece shape as a set of relative coordinates, normalized to origin
export type PieceShape = Coord[];

// Piece definition with all orientations pre-computed
export interface PieceDefinition {
    id: string;
    name: string;
    size: number;
    orientations: PieceShape[]; // All unique rotations/flips
}

// Result of analyzing a grid
export interface PieceAnalysis {
    onBoard: string[];   // Piece IDs found on board
    offBoard: string[];  // Piece IDs not found
}

export interface GridAnalysisResult {
    blue: PieceAnalysis;
    yellow: PieceAnalysis;
    red: PieceAnalysis;
    green: PieceAnalysis;
}

// ============================================================================
// Piece Catalog - All 21 Blokus pieces
// ============================================================================

// Base shapes (one canonical orientation each)
// Coordinates are [row, col] relative to top-left of bounding box
const BASE_PIECES: { id: string; name: string; shape: PieceShape }[] = [
    // 1 cell (monomino)
    { id: '1', name: 'O1', shape: [[0, 0]] },

    // 2 cells (domino)
    { id: '2', name: 'I2', shape: [[0, 0], [0, 1]] },

    // 3 cells (triominoes)
    { id: 'I3', name: 'I3', shape: [[0, 0], [0, 1], [0, 2]] },
    { id: 'V3', name: 'V3', shape: [[0, 0], [0, 1], [1, 0]] },

    // 4 cells (tetrominoes)
    { id: 'I4', name: 'I4', shape: [[0, 0], [0, 1], [0, 2], [0, 3]] },
    { id: 'O4', name: 'O4', shape: [[0, 0], [0, 1], [1, 0], [1, 1]] },
    { id: 'T4', name: 'T4', shape: [[0, 0], [0, 1], [0, 2], [1, 1]] },
    { id: 'L4', name: 'L4', shape: [[0, 0], [1, 0], [2, 0], [2, 1]] },
    { id: 'S4', name: 'S4', shape: [[0, 1], [0, 2], [1, 0], [1, 1]] },

    // 5 cells (pentominoes)
    { id: 'F', name: 'F', shape: [[0, 1], [0, 2], [1, 0], [1, 1], [2, 1]] },
    { id: 'I5', name: 'I5', shape: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]] },
    { id: 'L5', name: 'L5', shape: [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]] },
    { id: 'N', name: 'N', shape: [[0, 0], [0, 1], [1, 1], [1, 2], [1, 3]] },
    { id: 'P', name: 'P', shape: [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]] },
    { id: 'T5', name: 'T5', shape: [[0, 0], [0, 1], [0, 2], [1, 1], [2, 1]] },
    { id: 'U', name: 'U', shape: [[0, 0], [0, 2], [1, 0], [1, 1], [1, 2]] },
    { id: 'V', name: 'V', shape: [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]] },
    { id: 'W', name: 'W', shape: [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]] },
    { id: 'X', name: 'X', shape: [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]] },
    { id: 'Y', name: 'Y', shape: [[0, 1], [1, 0], [1, 1], [1, 2], [1, 3]] },
    { id: 'Z', name: 'Z', shape: [[0, 0], [0, 1], [1, 1], [2, 1], [2, 2]] },
];

// ============================================================================
// Shape Transformation Utilities
// ============================================================================

/**
 * Normalize a shape by translating it so the minimum row and col are 0,
 * then sorting coordinates for consistent comparison.
 */
export const normalizeShape = (shape: PieceShape): PieceShape => {
    if (shape.length === 0) {
        return [];
    }

    const minRow = Math.min(...shape.map(([r]) => r));
    const minCol = Math.min(...shape.map(([, c]) => c));

    const normalized = shape.map(([r, c]): Coord => [r - minRow, c - minCol]);

    // Sort by row, then by col for consistent ordering
    normalized.sort((a, b) => a[0] - b[0] || a[1] - b[1]);

    return normalized;
};

/**
 * Rotate shape 90 degrees clockwise.
 * [row, col] -> [col, -row] (then normalize)
 */
export const rotateShape = (shape: PieceShape): PieceShape => {
    const rotated = shape.map(([r, c]): Coord => [c, -r]);
    return normalizeShape(rotated);
};

/**
 * Flip shape horizontally (mirror).
 * [row, col] -> [row, -col] (then normalize)
 */
export const flipShape = (shape: PieceShape): PieceShape => {
    const flipped = shape.map(([r, c]): Coord => [r, -c]);
    return normalizeShape(flipped);
};

/**
 * Convert shape to string for comparison/hashing.
 */
export const shapeToString = (shape: PieceShape): string => shape.map(([r, c]) => `${r},${c}`).join(';');

/**
 * Generate all unique orientations (up to 8) for a shape.
 */
export const generateOrientations = (baseShape: PieceShape): PieceShape[] => {
    const seen = new Set<string>();
    const orientations: PieceShape[] = [];

    let current = normalizeShape(baseShape);

    // Try all 4 rotations
    for (let i = 0; i < 4; i++) {
        const key = shapeToString(current);
        if (!seen.has(key)) {
            seen.add(key);
            orientations.push(current);
        }
        current = rotateShape(current);
    }

    // Flip and try all 4 rotations again
    current = flipShape(normalizeShape(baseShape));
    for (let i = 0; i < 4; i++) {
        const key = shapeToString(current);
        if (!seen.has(key)) {
            seen.add(key);
            orientations.push(current);
        }
        current = rotateShape(current);
    }

    return orientations;
};

// ============================================================================
// Piece Catalog (with all orientations pre-computed)
// ============================================================================

export const PIECE_CATALOG: PieceDefinition[] = BASE_PIECES.map(({ id, name, shape }) => ({
    id,
    name,
    size: shape.length,
    orientations: generateOrientations(shape),
}));

// Map from piece ID to definition for quick lookup
export const PIECE_BY_ID = new Map(PIECE_CATALOG.map(p => [p.id, p]));

// All piece IDs
export const ALL_PIECE_IDS = PIECE_CATALOG.map(p => p.id);

// ============================================================================
// Connected Component Detection (Flood Fill)
// ============================================================================

/**
 * Find all connected components of a given color using edge-adjacency.
 * Returns array of piece shapes (each is array of [row, col] coordinates).
 */
export const findConnectedComponents = (grid: GridLabels, color: PlayerColor): PieceShape[] => {
    const visited = new Set<string>();
    const components: PieceShape[] = [];

    const key = (r: number, c: number) => `${r},${c}`;

    const floodFill = (startRow: number, startCol: number): PieceShape => {
        const component: PieceShape = [];
        const stack: Coord[] = [[startRow, startCol]];

        while (stack.length > 0) {
            const [r, c] = stack.pop()!;
            const k = key(r, c);

            if (visited.has(k)) {
                continue;
            }
            if (r < 0 || r >= 20 || c < 0 || c >= 20) {
                continue;
            }
            if (grid[r][c] !== color) {
                continue;
            }

            visited.add(k);
            component.push([r, c]);

            // Edge-adjacent neighbors (not diagonal)
            stack.push([r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]);
        }

        return component;
    };

    for (let r = 0; r < 20; r++) {
        for (let c = 0; c < 20; c++) {
            if (grid[r][c] === color && !visited.has(key(r, c))) {
                const component = floodFill(r, c);
                if (component.length > 0) {
                    components.push(component);
                }
            }
        }
    }

    return components;
};

// ============================================================================
// Piece Matching
// ============================================================================

/**
 * Match a detected shape against the piece catalog.
 * Returns the piece ID if found, null otherwise.
 */
export const matchPiece = (detectedShape: PieceShape): string | null => {
    const normalized = normalizeShape(detectedShape);
    const normalizedKey = shapeToString(normalized);

    for (const piece of PIECE_CATALOG) {
        if (piece.size !== detectedShape.length) {
            continue;
        }

        for (const orientation of piece.orientations) {
            if (shapeToString(orientation) === normalizedKey) {
                return piece.id;
            }
        }
    }

    return null;
};

// ============================================================================
// Main Analysis Function
// ============================================================================

/**
 * Analyze a grid and return which pieces are on/off board for each color.
 */
export const analyzeGrid = (grid: GridLabels): GridAnalysisResult => {
    const colors: PlayerColor[] = ['blue', 'yellow', 'red', 'green'];
    const result: GridAnalysisResult = {
        blue: { onBoard: [], offBoard: [] },
        yellow: { onBoard: [], offBoard: [] },
        red: { onBoard: [], offBoard: [] },
        green: { onBoard: [], offBoard: [] },
    };

    for (const color of colors) {
        const components = findConnectedComponents(grid, color);
        const foundPieces = new Set<string>();

        for (const component of components) {
            const pieceId = matchPiece(component);
            if (pieceId) {
                foundPieces.add(pieceId);
            }
            // If no match, it's either an invalid shape or detection error
            // We silently ignore for now
        }

        result[color].onBoard = Array.from(foundPieces).sort();
        result[color].offBoard = ALL_PIECE_IDS.filter(id => !foundPieces.has(id)).sort();
    }

    return result;
};
