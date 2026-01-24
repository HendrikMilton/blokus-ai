import { AnalysisResult, BoardCell, PlayerColor, BOARD_SIZE, PieceShape } from '../types/game';

export const BLOKUS_PIECES: PieceShape[] = [
    { name: 'I5', cells: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]], rotation: 0 },
    { name: 'L5', cells: [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]], rotation: 0 },
    { name: 'T5', cells: [[0, 0], [0, 1], [0, 2], [1, 1], [2, 1]], rotation: 0 },
    { name: 'Y', cells: [[0, 0], [1, 0], [1, 1], [2, 0], [3, 0]], rotation: 0 },
    { name: 'Z5', cells: [[0, 0], [0, 1], [1, 1], [2, 1], [2, 2]], rotation: 0 },
    { name: 'F', cells: [[0, 1], [0, 2], [1, 0], [1, 1], [2, 1]], rotation: 0 },
    { name: 'P', cells: [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]], rotation: 0 },
    { name: 'W', cells: [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]], rotation: 0 },
    { name: 'U', cells: [[0, 0], [0, 2], [1, 0], [1, 1], [1, 2]], rotation: 0 },
    { name: 'V5', cells: [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], rotation: 0 },
    { name: 'N', cells: [[0, 0], [0, 1], [1, 1], [1, 2], [1, 3]], rotation: 0 },
    { name: 'X', cells: [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]], rotation: 0 },
];

export const generateMockBoardState = (): BoardCell[][] => {
    const board: BoardCell[][] = Array(BOARD_SIZE).fill(null).map(() =>
        Array(BOARD_SIZE).fill(null).map(() => ({ color: null }))
    );

    const colors: PlayerColor[] = ['blue', 'yellow', 'red', 'green'];
    const cornerOffsets: [number, number][] = [[0, 0], [0, BOARD_SIZE - 5], [BOARD_SIZE - 5, 0], [BOARD_SIZE - 5, BOARD_SIZE - 5]];

    colors.forEach((color, index) => {
        const [rowOffset, colOffset] = cornerOffsets[index];
        const piece = BLOKUS_PIECES[index % BLOKUS_PIECES.length];

        piece.cells.forEach(([r, c]) => {
            const row = rowOffset + r;
            const col = colOffset + c;
            if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
                board[row][col] = { color };
            }
        });
    });

    return board;
};

export const generateMockAnalysis = (_playerColor: PlayerColor): AnalysisResult => {
    const randomPiece = BLOKUS_PIECES[Math.floor(Math.random() * BLOKUS_PIECES.length)];
    const rotations: (0 | 90 | 180 | 270)[] = [0, 90, 180, 270];

    return {
        recommendedPiece: {
            ...randomPiece,
            rotation: rotations[Math.floor(Math.random() * rotations.length)],
        },
        placement: {
            row: 5 + Math.floor(Math.random() * 10),
            col: 5 + Math.floor(Math.random() * 10),
        },
        stats: {
            piecesUsed: Math.floor(Math.random() * 10) + 1,
            piecesRemaining: 21 - (Math.floor(Math.random() * 10) + 1),
            winProbability: Math.floor(Math.random() * 40) + 30,
        },
    };
};
