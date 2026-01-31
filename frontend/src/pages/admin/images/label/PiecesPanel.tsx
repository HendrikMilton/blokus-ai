import { useState, useEffect, useMemo } from 'react';
import { Loader2, AlertCircle } from 'lucide-react';
import { PLAYER_COLORS } from '@blokus-ai/shared';
import type { GridLabels, GridAnalysisResult, PieceDefinition, PlayerColor } from '@blokus-ai/shared';
import { PIECES_API } from '../../../../api/pieces';

interface PiecesPanelProps {
    gridLabels: GridLabels | null;
    aiPrediction: GridLabels | null;
}

// Render a single piece shape as a mini grid
const PieceShape = ({
    piece,
    isOnBoard,
    color,
}: {
    piece: PieceDefinition;
    isOnBoard: boolean;
    color: string;
}) => {
    // Calculate bounding box of the shape
    const { minRow, maxRow, minCol, maxCol } = useMemo(() => {
        let minR = Infinity, maxR = -Infinity, minC = Infinity, maxC = -Infinity;
        for (const [r, c] of piece.shape) {
            minR = Math.min(minR, r);
            maxR = Math.max(maxR, r);
            minC = Math.min(minC, c);
            maxC = Math.max(maxC, c);
        }
        return { minRow: minR, maxRow: maxR, minCol: minC, maxCol: maxC };
    }, [piece.shape]);

    const rows = maxRow - minRow + 1;
    const cols = maxCol - minCol + 1;

    // Create a set for quick lookup
    const filledCells = useMemo(() => {
        const set = new Set<string>();
        for (const [r, c] of piece.shape) {
            set.add(`${r - minRow},${c - minCol}`);
        }
        return set;
    }, [piece.shape, minRow, minCol]);

    return (
        <div
            className={`relative transition-opacity ${isOnBoard ? 'opacity-30' : 'opacity-100'}`}
            title={`${piece.name} (${piece.size} cells)${isOnBoard ? ' - on board' : ' - available'}`}
        >
            <div
                className="grid gap-px"
                style={{
                    gridTemplateColumns: `repeat(${cols}, 6px)`,
                    gridTemplateRows: `repeat(${rows}, 6px)`,
                }}
            >
                {Array.from({ length: rows * cols }).map((_, idx) => {
                    const r = Math.floor(idx / cols);
                    const c = idx % cols;
                    const isFilled = filledCells.has(`${r},${c}`);
                    return (
                        <div
                            key={idx}
                            className="rounded-[1px]"
                            style={{
                                backgroundColor: isFilled ? color : 'transparent',
                            }}
                        />
                    );
                })}
            </div>
        </div>
    );
};

// Section for a single player color - now with used/unused grouping
const ColorSection = ({
    color,
    colorName,
    analysis,
    catalog,
}: {
    color: string;
    colorName: PlayerColor;
    analysis: { onBoard: string[]; offBoard: string[] } | null;
    catalog: PieceDefinition[];
}) => {
    // Sort pieces by size (descending) for display
    const sortedCatalog = useMemo(() =>
        [...catalog].sort((a, b) => b.size - a.size),
    [catalog]
    );

    if (!analysis) {
        return null;
    }

    const onBoardSet = new Set(analysis.onBoard);
    const usedPieces = sortedCatalog.filter(p => onBoardSet.has(p.id));
    const unusedPieces = sortedCatalog.filter(p => !onBoardSet.has(p.id));

    return (
        <div className="space-y-1.5">
            <div className="flex items-center gap-2">
                <div
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: color }}
                />
                <span className="text-xs font-medium text-foreground capitalize">{colorName}</span>
                <span className="text-[10px] text-muted">
                    {analysis.onBoard.length}/21 used
                </span>
            </div>

            {/* Used pieces */}
            {usedPieces.length > 0 && (
                <div className="ml-5 space-y-0.5">
                    <p className="text-[9px] text-muted">Used:</p>
                    <div className="flex flex-wrap gap-1.5 p-1.5 bg-surface rounded border border-border/50">
                        {usedPieces.map(piece => (
                            <PieceShape
                                key={piece.id}
                                piece={piece}
                                isOnBoard={true}
                                color={color}
                            />
                        ))}
                    </div>
                </div>
            )}

            {/* Unused pieces */}
            {unusedPieces.length > 0 && (
                <div className="ml-5 space-y-0.5">
                    <p className="text-[9px] text-muted">Available:</p>
                    <div className="flex flex-wrap gap-1.5 p-1.5 rounded border border-dashed border-border/30">
                        {unusedPieces.map(piece => (
                            <PieceShape
                                key={piece.id}
                                piece={piece}
                                isOnBoard={false}
                                color={color}
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export const PiecesPanel = ({ gridLabels, aiPrediction }: PiecesPanelProps) => {
    const [catalog, setCatalog] = useState<PieceDefinition[]>([]);
    const [humanAnalysis, setHumanAnalysis] = useState<GridAnalysisResult | null>(null);
    const [aiAnalysis, setAiAnalysis] = useState<GridAnalysisResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Load piece catalog on mount
    useEffect(() => {
        PIECES_API.getCatalog()
            .then(setCatalog)
            .catch(() => setError('Failed to load piece catalog'));
    }, []);

    // Analyze human grid when it changes
    useEffect(() => {
        if (!gridLabels) {
            setHumanAnalysis(null);
            return;
        }

        // Check if grid has any non-null cells
        const hasData = gridLabels.some(row => row.some(cell => cell !== null));
        if (!hasData) {
            setHumanAnalysis(null);
            return;
        }

        setIsLoading(true);
        setError(null);
        PIECES_API.analyzeGrid(gridLabels)
            .then(setHumanAnalysis)
            .catch(() => setError('Failed to analyze grid'))
            .finally(() => setIsLoading(false));
    }, [gridLabels]);

    // Analyze AI grid when it changes
    useEffect(() => {
        if (!aiPrediction) {
            setAiAnalysis(null);
            return;
        }

        const hasData = aiPrediction.some(row => row.some(cell => cell !== null));
        if (!hasData) {
            setAiAnalysis(null);
            return;
        }

        PIECES_API.analyzeGrid(aiPrediction)
            .then(setAiAnalysis)
            .catch(() => {}); // Silently fail for AI analysis
    }, [aiPrediction]);

    const colors: { name: PlayerColor; hex: string }[] = [
        { name: 'blue', hex: PLAYER_COLORS.blue },
        { name: 'yellow', hex: PLAYER_COLORS.yellow },
        { name: 'red', hex: PLAYER_COLORS.red },
        { name: 'green', hex: PLAYER_COLORS.green },
    ];

    if (error) {
        return (
            <div className="bg-surface-dim rounded-lg p-3 border border-border">
                <div className="flex items-center gap-2 text-error text-sm">
                    <AlertCircle className="w-4 h-4" />
                    {error}
                </div>
            </div>
        );
    }

    if (!gridLabels && !aiPrediction) {
        return (
            <div className="bg-surface-dim rounded-lg p-3 border border-border">
                <p className="text-xs text-muted text-center">
                    Label grid to see piece analysis
                </p>
            </div>
        );
    }

    return (
        <div className="bg-surface-dim rounded-lg p-3 border border-border space-y-3">
            <div className="flex items-center justify-between">
                <h4 className="text-xs font-medium text-secondary">Pieces Analysis</h4>
                {isLoading && <Loader2 className="w-3 h-3 animate-spin text-muted" />}
            </div>

            {humanAnalysis && (
                <div className="space-y-2">
                    <p className="text-[10px] text-muted uppercase tracking-wide">Human Labels</p>
                    {colors.map(({ name, hex }) => (
                        <ColorSection
                            key={name}
                            color={hex}
                            colorName={name}
                            analysis={humanAnalysis[name]}
                            catalog={catalog}
                        />
                    ))}
                </div>
            )}

            {aiAnalysis && (
                <div className="space-y-2 pt-2 border-t border-border">
                    <p className="text-[10px] text-muted uppercase tracking-wide">AI Prediction</p>
                    {colors.map(({ name, hex }) => (
                        <ColorSection
                            key={name}
                            color={hex}
                            colorName={name}
                            analysis={aiAnalysis[name]}
                            catalog={catalog}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};
