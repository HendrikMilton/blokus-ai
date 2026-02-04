import { BOARD_SIZE } from './ml';
import type { PlayerColor } from './ml';

// Re-export for convenience
export type { PlayerColor };
export { BOARD_SIZE };

// Starting corners for each player
export const STARTING_CORNERS: Record<PlayerColor, { row: number; col: number }> = {
    blue: { row: 0, col: 0 },           // Top-left
    yellow: { row: 0, col: 19 },        // Top-right
    red: { row: 19, col: 19 },          // Bottom-right
    green: { row: 19, col: 0 },         // Bottom-left
};

// Turn order (clockwise from top-left)
export const PLAYER_ORDER: PlayerColor[] = ['blue', 'yellow', 'red', 'green'];

// Cell on the board
export type CellState = PlayerColor | null;

// 20x20 board
export type Board = CellState[][];

// A move in the game
export interface Move {
    player: PlayerColor;
    pieceId: string;
    position: { row: number; col: number };
    orientationIndex: number;
}

// Player state (serializable version)
export interface PlayerState {
    color: PlayerColor;
    remainingPieceIds: string[];  // Array instead of Set for JSON
    hasPlayed: boolean;
    isEliminated: boolean;
    score: number;
}

// Full game state (serializable)
export interface GameState {
    board: Board;
    players: Record<PlayerColor, PlayerState>;
    currentPlayer: PlayerColor;
    turnNumber: number;
    isGameOver: boolean;
    consecutivePasses: number;
}

// Game configuration
export type GameMode = '1v1' | '4-player' | 'vs-ai';

export interface GameConfig {
    mode: GameMode;
    humanColors: PlayerColor[];  // Colors controlled by human(s)
    aiColors: PlayerColor[];     // Colors controlled by AI
    aiModelId?: string;          // Checkpoint ID for AlphaZero (undefined = random AI) - legacy, use aiModels
    aiModels?: Partial<Record<PlayerColor, string>>;  // Per-player AI model checkpoint IDs
}

// Result of attempting a move
export interface MoveResult {
    success: boolean;
    error?: string;
    newState?: GameState;
    aiMoves?: Move[];  // AI moves made after human move
}

// Game summary at end
export interface GameSummary {
    winner: PlayerColor | null;
    scores: Record<PlayerColor, number>;
    totalMoves: number;
}

// Piece definition for display (same as in ml.ts but re-exported for game context)
export interface GamePieceDefinition {
    id: string;
    name: string;
    size: number;
    orientations: [number, number][][];  // All rotations/flips
}

// Create an empty board
export const createEmptyBoard = (): Board =>
    Array.from({ length: BOARD_SIZE }, () =>
        Array.from({ length: BOARD_SIZE }, () => null)
    );

// Get next player in turn order
export const getNextPlayer = (current: PlayerColor): PlayerColor => {
    const idx = PLAYER_ORDER.indexOf(current);
    return PLAYER_ORDER[(idx + 1) % 4];
};

// For 1v1: which colors each player controls
export const TWO_PLAYER_TEAMS: Record<'player1' | 'player2', PlayerColor[]> = {
    player1: ['blue', 'red'],     // Opposite corners
    player2: ['yellow', 'green'], // Opposite corners
};
