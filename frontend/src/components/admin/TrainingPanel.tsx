/**
 * TrainingPanel - Controls for training the model
 */

import { useState } from 'react';
import { TrainResponse, PLAYER_COLORS } from '@blokus-ai/shared';
import { Button } from '../ui/Button';

interface TrainingPanelProps {
    labeledCount: number;
    onTrain: (params: { validation_split: number; min_samples: number }) => Promise<TrainResponse>;
    minSamplesRequired?: number;
}

export function TrainingPanel({
    labeledCount,
    onTrain,
    minSamplesRequired = 10,
}: TrainingPanelProps) {
    const [isTraining, setIsTraining] = useState(false);
    const [result, setResult] = useState<TrainResponse | null>(null);
    const [validationSplit, setValidationSplit] = useState(0.2);

    const canTrain = labeledCount >= minSamplesRequired;

    const handleTrain = async () => {
        setIsTraining(true);
        setResult(null);

        try {
            const response = await onTrain({
                validation_split: validationSplit,
                min_samples: minSamplesRequired,
            });
            setResult(response);
        } catch (error) {
            setResult({
                success: false,
                message: error instanceof Error ? error.message : 'Training failed',
                samples_used: 0,
                validation_accuracy: null,
                train_accuracy: null,
                color_accuracies: null,
                model_path: null,
            });
        } finally {
            setIsTraining(false);
        }
    };

    return (
        <div className="space-y-4 p-4 bg-surface rounded-lg border border-border">
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-sm font-medium text-foreground">Train Model</h3>
                    <p className="text-xs text-muted-foreground mt-0.5">
                        Build a KNN classifier from labeled samples
                    </p>
                </div>

                <Button
                    onClick={handleTrain}
                    disabled={!canTrain || isTraining}
                    loading={isTraining}
                    size="sm"
                >
                    {isTraining ? 'Training...' : 'Train'}
                </Button>
            </div>

            {/* Requirements */}
            <div className="flex items-center gap-4 text-xs">
                <div className={`flex items-center gap-1 ${canTrain ? 'text-green-400' : 'text-yellow-400'}`}>
                    {canTrain ? (
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                    ) : (
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                    )}
                    <span>
                        {labeledCount}/{minSamplesRequired} labeled samples
                    </span>
                </div>

                {/* Validation split slider */}
                <div className="flex items-center gap-2">
                    <label className="text-muted-foreground">Validation:</label>
                    <input
                        type="range"
                        min="0.1"
                        max="0.4"
                        step="0.1"
                        value={validationSplit}
                        onChange={(e) => setValidationSplit(parseFloat(e.target.value))}
                        className="w-20 h-1 bg-border rounded-lg appearance-none cursor-pointer"
                    />
                    <span>{Math.round(validationSplit * 100)}%</span>
                </div>
            </div>

            {/* Results */}
            {result && (
                <div className={`p-3 rounded-md ${result.success ? 'bg-green-500/10 border border-green-500/30' : 'bg-red-500/10 border border-red-500/30'}`}>
                    <div className="flex items-start gap-2">
                        {result.success ? (
                            <svg className="w-5 h-5 text-green-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        ) : (
                            <svg className="w-5 h-5 text-red-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        )}
                        <div className="flex-1 min-w-0">
                            <p className={`text-sm font-medium ${result.success ? 'text-green-400' : 'text-red-400'}`}>
                                {result.message}
                            </p>

                            {result.success && result.validation_accuracy !== null && (
                                <div className="mt-2 space-y-2">
                                    {/* Accuracy metrics */}
                                    <div className="grid grid-cols-2 gap-2 text-xs">
                                        <div>
                                            <span className="text-muted-foreground">Train Accuracy:</span>
                                            <span className="ml-1 text-foreground font-medium">
                                                {(result.train_accuracy! * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div>
                                            <span className="text-muted-foreground">Validation Accuracy:</span>
                                            <span className="ml-1 text-foreground font-medium">
                                                {(result.validation_accuracy * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>

                                    {/* Per-color accuracy */}
                                    {result.color_accuracies && (
                                        <div className="flex flex-wrap gap-2 text-xs">
                                            {Object.entries(result.color_accuracies).map(([color, accuracy]) => (
                                                <div
                                                    key={color}
                                                    className="flex items-center gap-1 px-2 py-0.5 rounded-full"
                                                    style={{
                                                        backgroundColor: color in PLAYER_COLORS
                                                            ? `${PLAYER_COLORS[color as keyof typeof PLAYER_COLORS]}20`
                                                            : 'rgba(128,128,128,0.2)',
                                                    }}
                                                >
                                                    <span
                                                        className="w-2 h-2 rounded-full"
                                                        style={{
                                                            backgroundColor: color in PLAYER_COLORS
                                                                ? PLAYER_COLORS[color as keyof typeof PLAYER_COLORS]
                                                                : '#888',
                                                        }}
                                                    />
                                                    <span className="capitalize">{color}:</span>
                                                    <span className="font-medium">{(accuracy * 100).toFixed(0)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    <p className="text-xs text-muted-foreground">
                                        Trained with {result.samples_used} samples
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
