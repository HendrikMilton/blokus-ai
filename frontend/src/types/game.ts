// Re-export from shared
export type { PlayerColor, BoardCell, PieceShape, AnalysisStats } from '@blokus-ai/shared';
export { PLAYER_COLORS, BOARD_SIZE } from '@blokus-ai/shared';

import type { PlayerColor, BoardCell, PieceShape, AnalysisStats } from '@blokus-ai/shared';

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
