// Board and game types
export type PlayerColor = 'blue' | 'yellow' | 'red' | 'green';

export interface BoardCell {
    color: PlayerColor | null;
}

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

// API Request/Response types
export interface DetectBoardRequest {
    image: string; // base64 encoded image
}

export interface DetectBoardResponse {
    boardState: BoardCell[][];
}

export interface AnalyzeBoardRequest {
    boardState: BoardCell[][];
    selectedColor: PlayerColor;
}

export interface AnalyzeBoardResponse {
    recommendedPiece: PieceShape | null;  // null if no valid moves
    placement: { row: number; col: number } | null;  // null if no valid moves
    stats: AnalysisStats;
}

// API paths
export const mlDetectBoardPath = '/api/ml/detect-board';
export const mlAnalyzeBoardPath = '/api/ml/analyze';

// Constants
export const BOARD_SIZE = 20;

export const PLAYER_COLORS: Record<PlayerColor, string> = {
    blue: '#0066CC',
    yellow: '#FFD700',
    red: '#DC143C',
    green: '#228B22',
};
