import type { Board, PlayerColor, Move, PlayerState } from './types';
import { BOARD_SIZE, STARTING_CORNERS } from './types';
import { PIECE_BY_ID } from '../pieces/service';

// Get the absolute cells a piece would occupy given a move
export const getPieceCells = (move: Move): { row: number; col: number }[] | null => {
    const piece = PIECE_BY_ID.get(move.pieceId);
    if (!piece) {
        return null;
    }

    const orientation = piece.orientations[move.orientationIndex];
    if (!orientation) {
        return null;
    }

    return orientation.map(([r, c]) => ({
        row: move.position.row + r,
        col: move.position.col + c,
    }));
};

// Check if all cells are within board bounds
export const isWithinBounds = (cells: { row: number; col: number }[]): boolean => {
    return cells.every(
        ({ row, col }) => row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE
    );
};

// Check if any cell overlaps with existing pieces on the board
export const hasOverlap = (board: Board, cells: { row: number; col: number }[]): boolean => {
    return cells.some(({ row, col }) => board[row][col] !== null);
};

// Check if piece touches the starting corner (for first move)
export const touchesStartingCorner = (
    cells: { row: number; col: number }[],
    player: PlayerColor
): boolean => {
    const corner = STARTING_CORNERS[player];
    return cells.some(({ row, col }) => row === corner.row && col === corner.col);
};

// Get all edge-adjacent cells (orthogonal neighbors)
const getEdgeNeighbors = (row: number, col: number): { row: number; col: number }[] => {
    return [
        { row: row - 1, col },
        { row: row + 1, col },
        { row, col: col - 1 },
        { row, col: col + 1 },
    ].filter(({ row: r, col: c }) => r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE);
};

// Get all diagonal-adjacent cells (corner neighbors)
const getDiagonalNeighbors = (row: number, col: number): { row: number; col: number }[] => {
    return [
        { row: row - 1, col: col - 1 },
        { row: row - 1, col: col + 1 },
        { row: row + 1, col: col - 1 },
        { row: row + 1, col: col + 1 },
    ].filter(({ row: r, col: c }) => r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE);
};

// Check if piece touches own color edge-to-edge (NOT allowed)
export const touchesOwnEdge = (
    board: Board,
    cells: { row: number; col: number }[],
    player: PlayerColor
): boolean => {
    const cellSet = new Set(cells.map(({ row, col }) => `${row},${col}`));

    for (const { row, col } of cells) {
        for (const neighbor of getEdgeNeighbors(row, col)) {
            // Skip if neighbor is part of the piece being placed
            if (cellSet.has(`${neighbor.row},${neighbor.col}`)) {
                continue;
            }
            if (board[neighbor.row][neighbor.col] === player) {
                return true;
            }
        }
    }
    return false;
};

// Check if piece touches own color diagonally (REQUIRED after first move)
export const touchesOwnDiagonal = (
    board: Board,
    cells: { row: number; col: number }[],
    player: PlayerColor
): boolean => {
    const cellSet = new Set(cells.map(({ row, col }) => `${row},${col}`));

    for (const { row, col } of cells) {
        for (const neighbor of getDiagonalNeighbors(row, col)) {
            // Skip if neighbor is part of the piece being placed
            if (cellSet.has(`${neighbor.row},${neighbor.col}`)) {
                continue;
            }
            if (board[neighbor.row][neighbor.col] === player) {
                return true;
            }
        }
    }
    return false;
};

// Validation result
export interface ValidationResult {
    valid: boolean;
    error?: string;
}

// Validate a move according to all Blokus rules
export const validateMove = (
    board: Board,
    playerState: PlayerState,
    move: Move
): ValidationResult => {
    // Check player matches
    if (move.player !== playerState.color) {
        return { valid: false, error: 'Move player does not match current player' };
    }

    // Check piece exists
    const piece = PIECE_BY_ID.get(move.pieceId);
    if (!piece) {
        return { valid: false, error: `Unknown piece: ${move.pieceId}` };
    }

    // Check player still has this piece
    if (!playerState.remainingPieces.has(move.pieceId)) {
        return { valid: false, error: `Piece ${move.pieceId} already played` };
    }

    // Check orientation is valid
    if (move.orientationIndex < 0 || move.orientationIndex >= piece.orientations.length) {
        return { valid: false, error: `Invalid orientation index: ${move.orientationIndex}` };
    }

    // Get absolute cell positions
    const cells = getPieceCells(move);
    if (!cells) {
        return { valid: false, error: 'Failed to compute piece cells' };
    }

    // Check bounds
    if (!isWithinBounds(cells)) {
        return { valid: false, error: 'Piece extends outside board' };
    }

    // Check overlap
    if (hasOverlap(board, cells)) {
        return { valid: false, error: 'Piece overlaps existing pieces' };
    }

    // Check edge-to-edge with own color (not allowed)
    if (touchesOwnEdge(board, cells, move.player)) {
        return { valid: false, error: 'Piece touches own color edge-to-edge' };
    }

    // First move rules
    if (!playerState.hasPlayed) {
        if (!touchesStartingCorner(cells, move.player)) {
            return { valid: false, error: 'First piece must cover starting corner' };
        }
    } else {
        // Subsequent moves must touch own color diagonally
        if (!touchesOwnDiagonal(board, cells, move.player)) {
            return { valid: false, error: 'Piece must touch own color diagonally' };
        }
    }

    return { valid: true };
};

// Find all valid positions for a specific piece orientation
export const findValidPositions = (
    board: Board,
    playerState: PlayerState,
    pieceId: string,
    orientationIndex: number
): { row: number; col: number }[] => {
    const piece = PIECE_BY_ID.get(pieceId);
    if (!piece || !playerState.remainingPieces.has(pieceId)) {
        return [];
    }

    const orientation = piece.orientations[orientationIndex];
    if (!orientation) {
        return [];
    }

    const validPositions: { row: number; col: number }[] = [];

    // Try every position on the board
    for (let row = -4; row < BOARD_SIZE; row++) {
        for (let col = -4; col < BOARD_SIZE; col++) {
            const move: Move = {
                player: playerState.color,
                pieceId,
                position: { row, col },
                orientationIndex,
            };

            const result = validateMove(board, playerState, move);
            if (result.valid) {
                validPositions.push({ row, col });
            }
        }
    }

    return validPositions;
};

// Generate all legal moves for a player
export const generateLegalMoves = (
    board: Board,
    playerState: PlayerState
): Move[] => {
    const moves: Move[] = [];

    for (const pieceId of playerState.remainingPieces) {
        const piece = PIECE_BY_ID.get(pieceId);
        if (!piece) {
            continue;
        }

        for (let orientationIndex = 0; orientationIndex < piece.orientations.length; orientationIndex++) {
            const positions = findValidPositions(board, playerState, pieceId, orientationIndex);
            for (const position of positions) {
                moves.push({
                    player: playerState.color,
                    pieceId,
                    position,
                    orientationIndex,
                });
            }
        }
    }

    return moves;
};
