import type { BoundingBox, BoardDetection } from '@blokus-ai/shared';
import { Button } from '../../../../components/ui/Button';
import { isUnlabeled } from '../../../../lib/status';
import { CORNER_CONFIG, type Sample, type LabelData } from './types';

interface LabelingControlsProps {
    currentSample: Sample;
    labelData: LabelData;
    currentDetection: BoardDetection | null;
    boundingBox: BoundingBox | null;
    isDrawingBox: boolean;
    isCornerMode: boolean;
    cornerCount: number;
    onDrawModeToggle: () => void;
    onDeleteLabel: () => void;
    onSaveAndNext: () => void;
    onToggleBoardDetected: () => void;
    onToggleEntireBoardVisible: () => void;
    updateCorrection: (updates: Partial<BoardDetection>) => void;
    updateBoundingBox: (box: BoundingBox) => void;
}

export const LabelingControls = ({
    currentSample,
    labelData,
    currentDetection,
    boundingBox,
    isDrawingBox,
    isCornerMode,
    cornerCount,
    onDrawModeToggle,
    onDeleteLabel,
    onSaveAndNext,
    onToggleBoardDetected,
    onToggleEntireBoardVisible,
    updateCorrection,
    updateBoundingBox,
}: LabelingControlsProps) => {
    return (
        <div className="space-y-4">
            {/* Mode hints */}
            {isCornerMode && (
                <p className="text-xs text-warning text-center">
                    Corner Mode: Click {CORNER_CONFIG[cornerCount]?.label || 'done'} · <kbd className="px-1 py-0.5 bg-surface-dim rounded text-[10px]">C</kbd> cancel
                </p>
            )}
            {isDrawingBox && (
                <p className="text-xs text-primary text-center">
                    Draw Mode · <kbd className="px-1 py-0.5 bg-surface-dim rounded text-[10px]">Arrows</kbd> move · <kbd className="px-1 py-0.5 bg-surface-dim rounded text-[10px]">Shift</kbd> resize
                </p>
            )}

            {/* Action buttons */}
            <div className="flex gap-2">
                {isDrawingBox ? (
                    <Button onClick={onDrawModeToggle} variant="secondary" className="flex-1">
                        Cancel Drawing (Esc)
                    </Button>
                ) : (
                    <Button onClick={() => {
                        onDrawModeToggle();
                        if (!boundingBox) {
                            updateBoundingBox({ x: 0.25, y: 0.25, width: 0.5, height: 0.5 });
                        }
                    }} variant="secondary" className="flex-1">
                        Draw Bounding Box (D)
                    </Button>
                )}

                {!isUnlabeled(currentSample.status) && (
                    <Button
                        onClick={onDeleteLabel}
                        variant="danger"
                        className="shrink-0"
                    >
                        Delete (X)
                    </Button>
                )}
            </div>

            {labelData.correction && (
                <Button
                    onClick={onSaveAndNext}
                    variant="success"
                    className="w-full"
                >
                    Save & Next (Enter)
                </Button>
            )}

            {/* Label toggles */}
            <div className="bg-surface rounded-lg border border-border p-4 space-y-3">
                <div className="flex items-center justify-between gap-4">
                    <span className="text-sm text-secondary">Board detected</span>
                    <div className="flex rounded-lg overflow-hidden border border-border shrink-0">
                        <button
                            onClick={() => {
                                if (!currentDetection?.hasBoardDetected) {
                                    onToggleBoardDetected();
                                }
                            }}
                            className={`px-3 py-1 text-sm font-medium transition-colors ${
                                currentDetection?.hasBoardDetected === true ? 'bg-success text-white' : 'bg-surface hover:bg-surface-dim text-secondary'
                            }`}
                        >
                            Yes (1)
                        </button>
                        <button
                            onClick={() => {
                                if (currentDetection?.hasBoardDetected !== false) {
                                    updateCorrection({ hasBoardDetected: false, isEntireBoardVisible: false, boundingBox: null });
                                }
                            }}
                            className={`px-3 py-1 text-sm font-medium transition-colors ${
                                currentDetection?.hasBoardDetected === false ? 'bg-error text-white' : 'bg-surface hover:bg-surface-dim text-secondary'
                            }`}
                        >
                            No (2)
                        </button>
                    </div>
                </div>

                <div className="flex items-center justify-between gap-4">
                    <span className="text-sm text-secondary">Entire board visible</span>
                    <div className={`flex rounded-lg overflow-hidden border border-border shrink-0 ${!currentDetection?.hasBoardDetected ? 'opacity-50 pointer-events-none' : ''}`}>
                        <button
                            onClick={() => {
                                if (!currentDetection?.isEntireBoardVisible) {
                                    onToggleEntireBoardVisible();
                                }
                            }}
                            className={`px-3 py-1 text-sm font-medium transition-colors ${
                                currentDetection?.isEntireBoardVisible === true ? 'bg-success text-white' : 'bg-surface hover:bg-surface-dim text-secondary'
                            }`}
                        >
                            Yes (3)
                        </button>
                        <button
                            onClick={() => {
                                if (currentDetection?.isEntireBoardVisible !== false) {
                                    updateCorrection({ isEntireBoardVisible: false });
                                }
                            }}
                            className={`px-3 py-1 text-sm font-medium transition-colors ${
                                currentDetection?.hasBoardDetected && currentDetection?.isEntireBoardVisible === false ? 'bg-error text-white' : 'bg-surface hover:bg-surface-dim text-secondary'
                            }`}
                        >
                            No (4)
                        </button>
                    </div>
                </div>
            </div>

            {/* Shortcut hints */}
            <div className="bg-surface-dim rounded-lg p-3 text-xs text-muted space-y-1">
                <p className="font-medium text-secondary mb-2">Keyboard Shortcuts</p>
                <p>
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">←</kbd>
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">→</kbd> navigate ·
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">D</kbd> draw bbox ·
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">C</kbd> corners ·
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">Z</kbd> grid mode ·
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">H</kbd> hide overlays
                </p>
                <p>
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">1</kbd> board yes ·
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">2</kbd> board no ·
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">3</kbd> visible yes ·
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">4</kbd> visible no
                </p>
                <p>
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">Enter</kbd> save & next ·
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">X</kbd> delete label ·
                    <kbd className="px-1.5 py-0.5 bg-surface rounded text-[10px]">Q</kbd> clear corners
                </p>
                <p>
                    Grid colors:
                    <kbd className="px-1.5 py-0.5 bg-blue-500 text-white rounded text-[10px] ml-1">B</kbd>
                    <kbd className="px-1.5 py-0.5 bg-yellow-400 text-black rounded text-[10px]">Y</kbd>
                    <kbd className="px-1.5 py-0.5 bg-green-600 text-white rounded text-[10px]">G</kbd>
                    <kbd className="px-1.5 py-0.5 bg-red-500 text-white rounded text-[10px]">R</kbd>
                    <kbd className="px-1.5 py-0.5 bg-gray-200 text-black rounded text-[10px]">W</kbd> empty
                </p>
            </div>
        </div>
    );
};
