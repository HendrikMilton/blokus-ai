import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGame } from '../contexts/GameContext';
import { BoardOverlay } from '../components/BoardOverlay';
import { StatsCard } from '../components/StatsCard';
import { Button, Loading } from '../components/ui';
import { PLAYER_COLORS, PieceShape } from '../types/game';

function PiecePreview({ piece, color }: { piece: PieceShape; color: string }) {
    const minRow = Math.min(...piece.cells.map(([r]) => r));
    const maxRow = Math.max(...piece.cells.map(([r]) => r));
    const minCol = Math.min(...piece.cells.map(([, c]) => c));
    const maxCol = Math.max(...piece.cells.map(([, c]) => c));

    const height = maxRow - minRow + 1;
    const width = maxCol - minCol + 1;
    const cellSize = 24;

    return (
        <div
            className="bg-surface rounded-2xl p-5 border-2 transition-all"
            style={{ borderColor: `${color}40` }}
        >
            <div className="flex items-center gap-2 mb-4">
                <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: color }}
                />
                <h3 className="text-sm font-semibold text-foreground">Recommended Move</h3>
            </div>
            <div className="flex items-center justify-center py-4">
                <svg
                    width={width * cellSize + 8}
                    height={height * cellSize + 8}
                    style={{ filter: `drop-shadow(0 0 12px ${color}60)` }}
                >
                    {piece.cells.map(([r, c], i) => (
                        <rect
                            key={i}
                            x={(c - minCol) * cellSize + 4}
                            y={(r - minRow) * cellSize + 4}
                            width={cellSize - 3}
                            height={cellSize - 3}
                            fill={color}
                            rx={3}
                            className="animate-pulse"
                        />
                    ))}
                </svg>
            </div>
            <div className="text-center space-y-1">
                <p className="text-foreground font-bold text-lg">{piece.name}</p>
                <p className="text-sm text-muted-foreground flex items-center justify-center gap-1">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    {piece.rotation}° rotation
                </p>
            </div>
        </div>
    );
}

export function ResultsPage() {
    const navigate = useNavigate();
    const { state, reset } = useGame();

    useEffect(() => {
        if (!state.selectedColor) {
            navigate('/');
        }
    }, [state.selectedColor, navigate]);

    const handleNewScan = () => {
        reset();
        navigate('/');
    };

    if (!state.selectedColor) {
        return null;
    }

    const playerColor = PLAYER_COLORS[state.selectedColor];

    // Loading state
    if (state.isAnalyzing || !state.analysis) {
        return (
            <div className="fixed inset-0 bg-background flex flex-col items-center justify-center">
                <Loading />
                <p className="mt-4 text-muted-foreground">Analyzing best move...</p>
                <p className="mt-2 text-sm text-muted-foreground/60">This may take a moment</p>
            </div>
        );
    }

    if (!state.boardState || !state.capturedImage) {
        return null;
    }

    return (
        <div className="fixed inset-0 bg-background flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-4 bg-background/80 backdrop-blur-sm border-b border-border">
                <h1 className="text-lg font-semibold text-foreground">Analysis Results</h1>
                <div
                    className="px-3 py-1 rounded-full text-sm font-medium text-white"
                    style={{ backgroundColor: playerColor }}
                >
                    {state.selectedColor}
                </div>
            </div>

            {/* Scrollable content */}
            <div className="flex-1 overflow-auto">
                {/* Board with recommendation overlay */}
                <div className="relative aspect-square">
                    <img
                        src={state.capturedImage}
                        alt="Board"
                        className="absolute inset-0 w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-black/20" />
                    <BoardOverlay
                        boardState={state.boardState}
                        analysis={state.analysis}
                        playerColor={state.selectedColor}
                        showRecommendation
                    />
                </div>

                {/* Results content */}
                <div className="p-4 space-y-4">
                    {state.analysis.recommendedPiece ? (
                        <PiecePreview
                            piece={state.analysis.recommendedPiece}
                            color={playerColor}
                        />
                    ) : (
                        <div className="bg-surface rounded-2xl p-5 border-2 border-yellow-500/40">
                            <p className="text-foreground font-medium text-center">No valid moves available</p>
                            <p className="text-muted-foreground text-sm text-center mt-1">You should pass this turn</p>
                        </div>
                    )}

                    <StatsCard
                        stats={state.analysis.stats}
                        color={state.selectedColor}
                    />

                    <Button
                        variant="primary"
                        onClick={handleNewScan}
                        fullWidth
                        className="mt-2"
                    >
                        New Scan
                    </Button>
                </div>
            </div>
        </div>
    );
}
