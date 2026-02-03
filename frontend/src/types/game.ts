// Re-export from shared
export type { PlayerColor, BoardCell } from '@blokus-ai/shared';
export { PLAYER_COLORS, BOARD_SIZE } from '@blokus-ai/shared';

import type { PlayerColor, BoardCell } from '@blokus-ai/shared';

// Local types for game analysis (not in shared)
export interface PieceShape {
    name: string;
    cells: [number, number][];
    rotation: 0 | 90 | 180 | 270;
}

export interface AnalysisStats {
    piecesUsed: number;
    piecesRemaining: number;
    winProbability: number;
}

export interface AnalysisResult {
    recommendedPiece: PieceShape | null;
    placement: { row: number; col: number } | null;
    stats: AnalysisStats;
}

export interface GameState {
    selectedColor: PlayerColor | null;
    capturedImage: string | null;
    boardState: BoardCell[][] | null;
    analysis: AnalysisResult | null;
    isDetecting: boolean;
    isAnalyzing: boolean;
}
