export type PlayerColor = 'blue' | 'yellow' | 'red' | 'green';

export interface BoardCell {
    color: PlayerColor | null;
}

export interface PieceShape {
    name: string;
    cells: [number, number][];
    rotation: 0 | 90 | 180 | 270;
}

export interface AnalysisResult {
    recommendedPiece: PieceShape;
    placement: { row: number; col: number };
    stats: {
        piecesUsed: number;
        piecesRemaining: number;
        winProbability: number;
    };
}

export interface GameState {
    selectedColor: PlayerColor | null;
    capturedImage: string | null;
    boardState: BoardCell[][] | null;
    analysis: AnalysisResult | null;
}

export const PLAYER_COLORS: Record<PlayerColor, string> = {
    blue: '#0066CC',
    yellow: '#FFD700',
    red: '#DC143C',
    green: '#228B22',
};

export const BOARD_SIZE = 21;
