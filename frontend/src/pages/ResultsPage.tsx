import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGame } from '../contexts/GameContext';
import { BoardOverlay } from '../components/BoardOverlay';
import { StatsCard } from '../components/StatsCard';
import { Button } from '../components/ui';
import { PLAYER_COLORS, PieceShape } from '../types/game';

function PiecePreview({ piece, color }: { piece: PieceShape; color: string }) {
    const minRow = Math.min(...piece.cells.map(([r]) => r));
    const maxRow = Math.max(...piece.cells.map(([r]) => r));
    const minCol = Math.min(...piece.cells.map(([, c]) => c));
    const maxCol = Math.max(...piece.cells.map(([, c]) => c));

    const height = maxRow - minRow + 1;
    const width = maxCol - minCol + 1;
    const cellSize = 20;

    return (
        <div className="bg-surface rounded-xl p-4 border border-border">
            <h3 className="text-sm font-medium text-muted-foreground mb-3">Recommended Piece</h3>
            <div className="flex items-center justify-center">
                <svg
                    width={width * cellSize + 4}
                    height={height * cellSize + 4}
                    className="drop-shadow-lg"
                >
                    {piece.cells.map(([r, c], i) => (
                        <rect
                            key={i}
                            x={(c - minCol) * cellSize + 2}
                            y={(r - minRow) * cellSize + 2}
                            width={cellSize - 2}
                            height={cellSize - 2}
                            fill={color}
                            rx={2}
                            className="animate-pulse"
                            style={{ filter: `drop-shadow(0 0 4px ${color})` }}
                        />
                    ))}
                </svg>
            </div>
            <p className="text-center mt-3 text-foreground font-medium">{piece.name}</p>
            <p className="text-center text-sm text-muted-foreground">Rotation: {piece.rotation}°</p>
        </div>
    );
}

export function ResultsPage() {
    const navigate = useNavigate();
    const { state, reset } = useGame();

    useEffect(() => {
        if (!state.analysis || !state.selectedColor) {
            navigate('/');
        }
    }, [state.analysis, state.selectedColor, navigate]);

    const handleNewScan = () => {
        reset();
        navigate('/');
    };

    if (!state.analysis || !state.selectedColor || !state.boardState || !state.capturedImage) {
        return null;
    }

    const playerColor = PLAYER_COLORS[state.selectedColor];

    return (
        <div className="fixed inset-0 bg-background flex flex-col overflow-auto">
            {/* Board with recommendation overlay */}
            <div className="relative aspect-square max-h-[50vh]">
                <img
                    src={state.capturedImage}
                    alt="Board"
                    className="absolute inset-0 w-full h-full object-cover"
                />
                <BoardOverlay
                    boardState={state.boardState}
                    analysis={state.analysis}
                    showRecommendation
                />
            </div>

            {/* Results content */}
            <div className="flex-1 p-4 space-y-4">
                <PiecePreview
                    piece={state.analysis.recommendedPiece}
                    color={playerColor}
                />

                <StatsCard
                    stats={state.analysis.stats}
                    color={state.selectedColor}
                />

                <Button
                    variant="primary"
                    onClick={handleNewScan}
                    fullWidth
                    className="mt-4"
                >
                    New Scan
                </Button>
            </div>
        </div>
    );
}
