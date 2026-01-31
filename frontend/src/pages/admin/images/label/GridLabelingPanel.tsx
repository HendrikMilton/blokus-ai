import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { Sparkles, Loader2 } from 'lucide-react';
import { PLAYER_COLORS } from '@blokus-ai/shared';
import type { GridPrediction, GridPredictionRecord } from '@blokus-ai/shared';
import { Button } from '../../../../components/ui/Button';
import { ML_API } from '../../../../api/ml';
import { CELL_COLORS, createEmptyGrid, type GridLabels, type PlayerColor, type CornerDetectionResult } from './types';

interface GridLabelingPanelProps {
    sampleId: string;
    cornerDetection: CornerDetectionResult | null;
    gridLabels: GridLabels | null;
    selectedColor: PlayerColor | null;
    isGridMode: boolean;
    isGridModified: boolean;
    initialPrediction: GridPredictionRecord | null;
    isEntireBoardVisible: boolean;
    onCellClick: (row: number, col: number) => void;
    onColorSelect: (color: PlayerColor | null) => void;
    onSave: () => void;
    setGridLabels: (grid: GridLabels) => void;
}

// Calculate comparison metrics between AI prediction and human labels
const calculateMetrics = (prediction: GridLabels, humanLabels: GridLabels) => {
    let correct = 0;
    let total = 0;
    const colorStats: Record<string, { tp: number; fp: number; fn: number }> = {
        blue: { tp: 0, fp: 0, fn: 0 },
        yellow: { tp: 0, fp: 0, fn: 0 },
        red: { tp: 0, fp: 0, fn: 0 },
        green: { tp: 0, fp: 0, fn: 0 },
        empty: { tp: 0, fp: 0, fn: 0 },
    };

    for (let r = 0; r < 20; r++) {
        for (let c = 0; c < 20; c++) {
            const pred = prediction[r][c];
            const human = humanLabels[r][c];
            total++;

            if (pred === human) {
                correct++;
                const key = pred || 'empty';
                colorStats[key].tp++;
            } else {
                const predKey = pred || 'empty';
                const humanKey = human || 'empty';
                colorStats[predKey].fp++;
                colorStats[humanKey].fn++;
            }
        }
    }

    return { correct, total, accuracy: correct / total, colorStats };
};

// Grid overlay component for reuse
const GridOverlay = ({
    grid,
    croppedUrl,
    onClick,
    interactive = false,
    label,
    borderColor = 'border-border',
}: {
    grid: GridLabels;
    croppedUrl: string;
    onClick?: (row: number, col: number) => void;
    interactive?: boolean;
    label: string;
    borderColor?: string;
}) => (
    <div className="space-y-1">
        <p className="text-xs text-secondary text-center">{label}</p>
        <div className={`relative aspect-square bg-surface rounded-lg border-2 overflow-hidden ${borderColor}`}>
            <img
                src={croppedUrl}
                alt="Cropped board"
                className="absolute inset-0 w-full h-full object-cover"
                draggable={false}
            />
            <div className="absolute inset-0 grid grid-cols-20 grid-rows-20">
                {grid.map((row, ri) =>
                    row.map((cell, ci) => (
                        <button
                            key={`${ri}-${ci}`}
                            onClick={() => onClick?.(ri, ci)}
                            disabled={!interactive}
                            className={`border border-white/20 transition-colors ${interactive ? 'hover:border-white/60' : ''}`}
                            style={{
                                backgroundColor: cell ? `${PLAYER_COLORS[cell]}CC` : 'transparent',
                            }}
                        />
                    ))
                )}
            </div>
        </div>
    </div>
);

export const GridLabelingPanel = ({
    sampleId,
    cornerDetection,
    gridLabels,
    selectedColor,
    isGridMode,
    isGridModified,
    initialPrediction,
    isEntireBoardVisible,
    onCellClick,
    onColorSelect,
    onSave,
    setGridLabels,
}: GridLabelingPanelProps) => {
    const [aiPrediction, setAiPrediction] = useState<GridPrediction | null>(null);
    const [isPredicting, setIsPredicting] = useState(false);

    // Initialize aiPrediction from initialPrediction when it changes
    useEffect(() => {
        if (initialPrediction) {
            setAiPrediction({
                grid: initialPrediction.grid,
                confidence: initialPrediction.confidence,
                cellConfidences: [], // Not stored in DB, empty for display
                processingTimeMs: initialPrediction.processingTimeMs,
            });
        } else {
            setAiPrediction(null);
        }
    }, [initialPrediction]);

    const croppedUrl = cornerDetection?.best.croppedUrl || cornerDetection?.hough.croppedUrl || cornerDetection?.edge.croppedUrl;

    const handlePredict = async () => {
        if (!sampleId) {
            return;
        }
        setIsPredicting(true);
        try {
            const prediction = await ML_API.predictGrid(sampleId);
            setAiPrediction(prediction);
            toast.success(`Grid predicted in ${prediction.processingTimeMs}ms`);
        } catch (error) {
            toast.error('Failed to predict grid');
            console.error(error);
        } finally {
            setIsPredicting(false);
        }
    };

    const handleAcceptPrediction = () => {
        if (aiPrediction) {
            setGridLabels(aiPrediction.grid);
            toast.success('AI prediction accepted');
        }
    };

    const handleCellClick = (row: number, col: number) => {
        if (!gridLabels) {
            setGridLabels(createEmptyGrid());
        }
        onCellClick(row, col);
    };

    // Calculate metrics if we have both prediction and human labels
    const metrics = aiPrediction && gridLabels ? calculateMetrics(aiPrediction.grid, gridLabels) : null;

    return (
        <div className="space-y-3">
            {/* Header with color palette */}
            <div className="flex items-center justify-between">
                <h3 className={`text-sm font-medium ${isGridMode ? 'text-success' : 'text-secondary'}`}>
                    Grid Labeling {isGridMode && <span className="ml-1 px-1.5 py-0.5 bg-success/20 rounded text-[10px]">ACTIVE</span>}
                </h3>
                <div className="flex gap-1">
                    {CELL_COLORS.map(({ key, label, color }) => (
                        <button
                            key={label}
                            onClick={() => onColorSelect(key)}
                            className={`w-6 h-6 rounded border-2 transition-all ${
                                selectedColor === key ? 'border-foreground scale-110 ring-2 ring-foreground/30' : 'border-border'
                            }`}
                            style={{ backgroundColor: color }}
                            title={label}
                        />
                    ))}
                </div>
            </div>

            {!isEntireBoardVisible ? (
                <div className="aspect-square bg-surface-dim rounded-lg border border-border flex items-center justify-center">
                    <p className="text-sm text-muted text-center px-4">Grid labeling requires &quot;Entire board visible&quot; = Yes</p>
                </div>
            ) : !croppedUrl ? (
                <div className="aspect-square bg-surface-dim rounded-lg border border-border flex items-center justify-center">
                    <p className="text-sm text-muted text-center px-4">Run corner detection to enable grid labeling</p>
                </div>
            ) : (
                <>
                    {/* Grids side by side */}
                    <div className="grid grid-cols-2 gap-4">
                        {/* Human labeled grid */}
                        <GridOverlay
                            grid={gridLabels || createEmptyGrid()}
                            croppedUrl={croppedUrl}
                            onClick={handleCellClick}
                            interactive={true}
                            label="Human Labels"
                            borderColor={isGridMode ? 'border-success' : 'border-border'}
                        />

                        {/* AI prediction grid */}
                        {aiPrediction ? (
                            <GridOverlay
                                grid={aiPrediction.grid}
                                croppedUrl={croppedUrl}
                                interactive={false}
                                label="AI Prediction"
                                borderColor="border-primary"
                            />
                        ) : (
                            <div className="space-y-1">
                                <p className="text-xs text-secondary text-center">AI Prediction</p>
                                <div className="aspect-square bg-surface-dim rounded-lg border-2 border-dashed border-border flex items-center justify-center">
                                    <p className="text-xs text-muted text-center px-2">Click "Predict" to run AI</p>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* AI Prediction button and controls */}
                    <div className="flex gap-2">
                        <Button
                            onClick={handlePredict}
                            disabled={isPredicting}
                            variant="secondary"
                            size="sm"
                            className="flex-1"
                        >
                            {isPredicting ? (
                                <>
                                    <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />
                                    Predicting...
                                </>
                            ) : (
                                <>
                                    <Sparkles className="w-4 h-4 mr-1.5" />
                                    Predict Grid (AI)
                                </>
                            )}
                        </Button>
                        {aiPrediction && (
                            <Button
                                onClick={handleAcceptPrediction}
                                variant="primary"
                                size="sm"
                            >
                                Accept AI
                            </Button>
                        )}
                    </div>

                    {/* AI Prediction metrics */}
                    {aiPrediction && (
                        <div className="bg-surface-dim rounded-lg p-3 space-y-2">
                            <div className="flex items-center justify-between text-xs">
                                <span className="text-secondary">AI Confidence</span>
                                <span className="text-muted">{(aiPrediction.confidence * 100).toFixed(1)}% · {aiPrediction.processingTimeMs}ms</span>
                            </div>

                            {/* Comparison metrics */}
                            {metrics && (
                                <div className="pt-2 border-t border-border space-y-1">
                                    <div className="flex items-center justify-between text-xs">
                                        <span className="text-secondary">vs Human Labels</span>
                                        <span className={`font-medium ${metrics.accuracy >= 0.9 ? 'text-success' : metrics.accuracy >= 0.7 ? 'text-warning' : 'text-error'}`}>
                                            {(metrics.accuracy * 100).toFixed(1)}% ({metrics.correct}/{metrics.total})
                                        </span>
                                    </div>
                                    {/* Per-color precision and recall table */}
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-[10px]">
                                            <thead>
                                                <tr className="text-muted">
                                                    <th className="text-left py-1">Color</th>
                                                    <th className="text-right py-1">Precision</th>
                                                    <th className="text-right py-1">Recall</th>
                                                    <th className="text-right py-1">F1</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {Object.entries(metrics.colorStats).map(([color, stats]) => {
                                                    // If no predictions and no actual instances, that's 100% (correctly didn't predict)
                                                    // If predictions but no actual instances (all FP), precision = 0
                                                    // If actual instances but no predictions (all FN), recall = 0
                                                    const precision = stats.tp + stats.fp > 0 ? stats.tp / (stats.tp + stats.fp) : (stats.fn > 0 ? 0 : 1);
                                                    const recall = stats.tp + stats.fn > 0 ? stats.tp / (stats.tp + stats.fn) : (stats.fp > 0 ? 0 : 1);
                                                    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 1;
                                                    const bgColor = color === 'empty' ? '#9ca3af' : PLAYER_COLORS[color as PlayerColor];
                                                    return (
                                                        <tr key={color} className="border-t border-border/50">
                                                            <td className="py-1 text-foreground">
                                                                <span className="inline-flex items-center gap-1">
                                                                    <span
                                                                        className="w-2.5 h-2.5 rounded-sm"
                                                                        style={{ backgroundColor: bgColor }}
                                                                    />
                                                                    <span className="capitalize">{color}</span>
                                                                </span>
                                                            </td>
                                                            <td className={`text-right py-1 ${precision >= 0.8 ? 'text-success' : precision >= 0.5 ? 'text-warning' : 'text-error'}`}>
                                                                {(precision * 100).toFixed(0)}%
                                                            </td>
                                                            <td className={`text-right py-1 ${recall >= 0.8 ? 'text-success' : recall >= 0.5 ? 'text-warning' : 'text-error'}`}>
                                                                {(recall * 100).toFixed(0)}%
                                                            </td>
                                                            <td className={`text-right py-1 font-medium ${f1 >= 0.8 ? 'text-success' : f1 >= 0.5 ? 'text-warning' : 'text-error'}`}>
                                                                {(f1 * 100).toFixed(0)}%
                                                            </td>
                                                        </tr>
                                                    );
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </>
            )}

            {isGridModified && (
                <Button
                    variant="success"
                    size="sm"
                    className="w-full"
                    onClick={onSave}
                >
                    Save Grid Labels
                </Button>
            )}
        </div>
    );
};
