import type { PlayerColor } from '@blokus-ai/shared';

// Re-export for convenience
export type { PlayerColor };

// Board dimensions
export const BOARD_SIZE = 20;

// Starting corners for each player (where first piece must touch)
export const STARTING_CORNERS: Record<PlayerColor, { row: number; col: number }> = {
    blue: { row: 0, col: 0 },           // Top-left
    yellow: { row: 0, col: 19 },        // Top-right
    red: { row: 19, col: 19 },          // Bottom-right
    green: { row: 19, col: 0 },         // Bottom-left
};

// Turn order
export const PLAYER_ORDER: PlayerColor[] = ['blue', 'yellow', 'red', 'green'];

// Cell on the board
export type CellState = PlayerColor | null;

// 20x20 board
export type Board = CellState[][];

// A move in the game
export interface Move {
    player: PlayerColor;
    pieceId: string;
    // Position of the piece's origin (top-left of bounding box) on the board
    position: { row: number; col: number };
    // Which orientation to use (index into piece's orientations array)
    orientationIndex: number;
}

// Player state
export interface PlayerState {
    color: PlayerColor;
    remainingPieces: Set<string>;  // Piece IDs not yet played
    hasPlayed: boolean;            // Has played at least one piece
    isEliminated: boolean;         // Cannot make any more moves
    score: number;                 // Total cells placed
}

// Full game state
export interface GameState {
    board: Board;
    players: Record<PlayerColor, PlayerState>;
    currentPlayer: PlayerColor;
    turnNumber: number;
    isGameOver: boolean;
    consecutivePasses: number;     // Track when all players pass
}

// Result of attempting a move
export interface MoveResult {
    success: boolean;
    error?: string;
    newState?: GameState;
}

// A placed piece on the board (for move history)
export interface PlacedPiece {
    player: PlayerColor;
    pieceId: string;
    cells: { row: number; col: number }[];  // Absolute board positions
}

// Game summary at end
export interface GameSummary {
    winner: PlayerColor | null;    // null if tie
    scores: Record<PlayerColor, number>;
    totalMoves: number;
    finalBoard: Board;
}

// Create an empty board
export const createEmptyBoard = (): Board =>
    Array.from({ length: BOARD_SIZE }, () =>
        Array.from({ length: BOARD_SIZE }, () => null)
    );

// Create initial player state
export const createPlayerState = (color: PlayerColor, allPieceIds: string[]): PlayerState => ({
    color,
    remainingPieces: new Set(allPieceIds),
    hasPlayed: false,
    isEliminated: false,
    score: 0,
});

// Get next player in turn order
export const getNextPlayer = (current: PlayerColor): PlayerColor => {
    const idx = PLAYER_ORDER.indexOf(current);
    return PLAYER_ORDER[(idx + 1) % 4];
};
