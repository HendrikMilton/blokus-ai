import { BoardCell, BOARD_SIZE, PLAYER_COLORS, PlayerColor, AnalysisResult } from '../types/game';

interface BoardOverlayProps {
    boardState: BoardCell[][];
    analysis?: AnalysisResult | null;
    playerColor?: PlayerColor | null;
    showRecommendation?: boolean;
}

export const BoardOverlay = ({ boardState, analysis, playerColor, showRecommendation = false }: BoardOverlayProps) => {
    const recommendationColor = playerColor ? PLAYER_COLORS[playerColor] : '#3b82f6';
    const cellSize = 100 / BOARD_SIZE;

    return (
        <div className="absolute inset-0 pointer-events-none">
            <svg viewBox="0 0 100 100" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
                {/* Grid lines */}
                {Array.from({ length: BOARD_SIZE + 1 }).map((_, i) => (
                    <g key={i}>
                        <line
                            x1={i * cellSize}
                            y1={0}
                            x2={i * cellSize}
                            y2={100}
                            stroke="rgba(255,255,255,0.2)"
                            strokeWidth={0.1}
                        />
                        <line
                            x1={0}
                            y1={i * cellSize}
                            x2={100}
                            y2={i * cellSize}
                            stroke="rgba(255,255,255,0.2)"
                            strokeWidth={0.1}
                        />
                    </g>
                ))}

                {/* Placed pieces */}
                {boardState.map((row, rowIndex) =>
                    row.map((cell, colIndex) => {
                        if (!cell.color) {
                            return null;
                        }
                        return (
                            <rect
                                key={`${rowIndex}-${colIndex}`}
                                x={colIndex * cellSize + 0.1}
                                y={rowIndex * cellSize + 0.1}
                                width={cellSize - 0.2}
                                height={cellSize - 0.2}
                                fill={PLAYER_COLORS[cell.color]}
                                opacity={0.7}
                                rx={0.2}
                            />
                        );
                    })
                )}

                {/* Recommended move with pulsing glow effect */}
                {showRecommendation && analysis?.recommendedPiece && analysis?.placement && (
                    <g className="animate-pulse">
                        {analysis.recommendedPiece.cells.map(([r, c], i) => {
                            const row = analysis.placement!.row + r;
                            const col = analysis.placement!.col + c;
                            if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) {
                                return null;
                            }
                            return (
                                <rect
                                    key={i}
                                    x={col * cellSize + 0.1}
                                    y={row * cellSize + 0.1}
                                    width={cellSize - 0.2}
                                    height={cellSize - 0.2}
                                    fill={recommendationColor}
                                    opacity={0.9}
                                    rx={0.2}
                                    style={{
                                        filter: `drop-shadow(0 0 3px ${recommendationColor})`,
                                    }}
                                />
                            );
                        })}
                    </g>
                )}
            </svg>
        </div>
    );
};
