import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { http, API_URL } from '../../../../api/config';
import { Button } from '../../../../components/ui/Button';
import type { BoundingBox, BoardDetection } from '@blokus-ai/shared';

interface Sample {
    id: string;
    filename: string;
    status: 'unlabeled' | 'labeled' | 'verified';
    created_at: string;
    imageUrl?: string;
    thumbnailUrl?: string;
}

interface SamplesResponse {
    samples: Sample[];
}

interface LabelData {
    prediction: BoardDetection | null;
    correction: BoardDetection | null;
    isModified: boolean;
}

type DragHandle = 'nw' | 'ne' | 'sw' | 'se' | 'move' | null;

const BackIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
    </svg>
);

const ChevronLeftIcon = () => (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
    </svg>
);

const ChevronRightIcon = () => (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
    </svg>
);

export const LabelingPage = () => {
    const navigate = useNavigate();
    const [unlabeledSamples, setUnlabeledSamples] = useState<Sample[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const [labelData, setLabelData] = useState<LabelData>({
        prediction: null,
        correction: null,
        isModified: false,
    });
    const [isDrawingBox, setIsDrawingBox] = useState(false);
    const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
    const [dragHandle, setDragHandle] = useState<DragHandle>(null);
    const [dragStartBox, setDragStartBox] = useState<BoundingBox | null>(null);
    const [dragStartPos, setDragStartPos] = useState<{ x: number; y: number } | null>(null);
    const imageContainerRef = useRef<HTMLDivElement>(null);

    const currentSample = unlabeledSamples[currentIndex] || null;
    const currentDetection = labelData.correction || labelData.prediction;
    const boundingBox = currentDetection?.boundingBox;

    // Load unlabeled samples
    useEffect(() => {
        const loadSamples = async () => {
            try {
                const data = await http.get<SamplesResponse>('/api/ml/samples');
                const unlabeled = (data.samples || []).filter(s => s.status === 'unlabeled');
                setUnlabeledSamples(unlabeled);
            } catch {
                // Error toast shown by http client
            } finally {
                setIsLoading(false);
            }
        };
        loadSamples();
    }, []);

    // Reset label data when sample changes
    useEffect(() => {
        setLabelData({
            prediction: null,
            correction: null,
            isModified: false,
        });
        setIsDrawingBox(false);
    }, [currentIndex]);

    // Keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Don't handle if user is typing in an input
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
                return;
            }

            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                goToPrevious();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                goToNext();
            } else if (e.key === 's' || e.key === 'S') {
                e.preventDefault();
                if (currentSample && labelData.correction) {
                    handleSaveAndNext();
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [currentIndex, unlabeledSamples.length, labelData.correction, currentSample]);

    const goToPrevious = () => {
        if (currentIndex > 0) {
            setCurrentIndex(currentIndex - 1);
        }
    };

    const goToNext = () => {
        if (currentIndex < unlabeledSamples.length - 1) {
            setCurrentIndex(currentIndex + 1);
        }
    };

    const handleSaveAndNext = async () => {
        if (!currentSample || !labelData.correction) return;

        try {
            // Save the manual label
            await http.post(`/api/ml/samples/${currentSample.id}/label`, {
                hasBoardDetected: labelData.correction.hasBoardDetected,
                isEntireBoardVisible: labelData.correction.isEntireBoardVisible,
                boundingBox: labelData.correction.boundingBox,
            });
            toast.success('Saved!');

            // Remove from list and move to next
            setUnlabeledSamples(prev => prev.filter(s => s.id !== currentSample.id));
            // Index stays the same (next item slides into current position)
            // Unless we were at the end
            if (currentIndex >= unlabeledSamples.length - 1 && currentIndex > 0) {
                setCurrentIndex(currentIndex - 1);
            }
        } catch {
            toast.error('Failed to save');
        }
    };

    const updateCorrection = useCallback((updates: Partial<BoardDetection>) => {
        setLabelData(prev => {
            const base = prev.correction || prev.prediction || {
                hasBoardDetected: false,
                isEntireBoardVisible: false,
                boundingBox: null,
            };
            return {
                ...prev,
                correction: { ...base, ...updates },
                isModified: true,
            };
        });
    }, []);

    const updateBoundingBox = useCallback((newBox: BoundingBox) => {
        const minSize = 0.02;
        const clampedBox: BoundingBox = {
            x: Math.max(0, Math.min(1 - minSize, newBox.x)),
            y: Math.max(0, Math.min(1 - minSize, newBox.y)),
            width: Math.max(minSize, Math.min(1 - newBox.x, newBox.width)),
            height: Math.max(minSize, Math.min(1 - newBox.y, newBox.height)),
        };
        updateCorrection({
            boundingBox: clampedBox,
            hasBoardDetected: true,
        });
    }, [updateCorrection]);

    const toggleBoardDetected = () => {
        const current = currentDetection?.hasBoardDetected ?? false;
        updateCorrection({
            hasBoardDetected: !current,
            ...(!current ? {} : { isEntireBoardVisible: false, boundingBox: null }),
        });
    };

    const toggleEntireBoardVisible = () => {
        const current = currentDetection?.isEntireBoardVisible ?? false;
        updateCorrection({ isEntireBoardVisible: !current });
    };

    const getRelativePosition = useCallback((clientX: number, clientY: number) => {
        if (!imageContainerRef.current) return null;
        const rect = imageContainerRef.current.getBoundingClientRect();
        return {
            x: Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
            y: Math.max(0, Math.min(1, (clientY - rect.top) / rect.height)),
        };
    }, []);

    const handlePointerDown = (e: React.PointerEvent) => {
        const pos = getRelativePosition(e.clientX, e.clientY);
        if (!pos) return;

        if (isDrawingBox) {
            e.preventDefault();
            setDrawStart(pos);
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            return;
        }

        if (boundingBox) {
            const handleSize = 0.03;
            const box = boundingBox;
            const corners: { handle: DragHandle; x: number; y: number }[] = [
                { handle: 'nw', x: box.x, y: box.y },
                { handle: 'ne', x: box.x + box.width, y: box.y },
                { handle: 'sw', x: box.x, y: box.y + box.height },
                { handle: 'se', x: box.x + box.width, y: box.y + box.height },
            ];

            for (const corner of corners) {
                if (Math.abs(pos.x - corner.x) < handleSize && Math.abs(pos.y - corner.y) < handleSize) {
                    e.preventDefault();
                    setDragHandle(corner.handle);
                    setDragStartBox({ ...box });
                    setDragStartPos(pos);
                    (e.target as HTMLElement).setPointerCapture(e.pointerId);
                    return;
                }
            }

            if (pos.x >= box.x && pos.x <= box.x + box.width &&
                pos.y >= box.y && pos.y <= box.y + box.height) {
                e.preventDefault();
                setDragHandle('move');
                setDragStartBox({ ...box });
                setDragStartPos(pos);
                (e.target as HTMLElement).setPointerCapture(e.pointerId);
                return;
            }
        }
    };

    const handlePointerMove = (e: React.PointerEvent) => {
        const pos = getRelativePosition(e.clientX, e.clientY);
        if (!pos) return;

        if (isDrawingBox && drawStart) {
            e.preventDefault();
            const x = Math.min(drawStart.x, pos.x);
            const y = Math.min(drawStart.y, pos.y);
            const width = Math.abs(pos.x - drawStart.x);
            const height = Math.abs(pos.y - drawStart.y);
            updateBoundingBox({ x, y, width, height });
            return;
        }

        if (dragHandle && dragStartBox && dragStartPos) {
            e.preventDefault();
            const dx = pos.x - dragStartPos.x;
            const dy = pos.y - dragStartPos.y;
            let newBox: BoundingBox;

            switch (dragHandle) {
            case 'move':
                newBox = {
                    x: Math.max(0, Math.min(1 - dragStartBox.width, dragStartBox.x + dx)),
                    y: Math.max(0, Math.min(1 - dragStartBox.height, dragStartBox.y + dy)),
                    width: dragStartBox.width,
                    height: dragStartBox.height,
                };
                break;
            case 'nw':
                newBox = { x: dragStartBox.x + dx, y: dragStartBox.y + dy, width: dragStartBox.width - dx, height: dragStartBox.height - dy };
                break;
            case 'ne':
                newBox = { x: dragStartBox.x, y: dragStartBox.y + dy, width: dragStartBox.width + dx, height: dragStartBox.height - dy };
                break;
            case 'sw':
                newBox = { x: dragStartBox.x + dx, y: dragStartBox.y, width: dragStartBox.width - dx, height: dragStartBox.height + dy };
                break;
            case 'se':
                newBox = { x: dragStartBox.x, y: dragStartBox.y, width: dragStartBox.width + dx, height: dragStartBox.height + dy };
                break;
            default:
                return;
            }
            updateBoundingBox(newBox);
        }
    };

    const handlePointerUp = (e: React.PointerEvent) => {
        if (isDrawingBox && drawStart) {
            (e.target as HTMLElement).releasePointerCapture(e.pointerId);
            setDrawStart(null);
            setIsDrawingBox(false);
        }
        if (dragHandle) {
            (e.target as HTMLElement).releasePointerCapture(e.pointerId);
            setDragHandle(null);
            setDragStartBox(null);
            setDragStartPos(null);
        }
    };

    const getCursor = () => {
        if (isDrawingBox) return 'crosshair';
        if (dragHandle === 'move') return 'move';
        if (dragHandle === 'nw' || dragHandle === 'se') return 'nwse-resize';
        if (dragHandle === 'ne' || dragHandle === 'sw') return 'nesw-resize';
        return 'default';
    };

    if (isLoading) {
        return (
            <div className="space-y-6">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-surface-dim rounded-lg animate-pulse" />
                    <div className="h-6 w-48 bg-surface-dim rounded animate-pulse" />
                </div>
                <div className="aspect-square max-w-2xl mx-auto bg-surface-dim rounded-lg animate-pulse" />
            </div>
        );
    }

    if (unlabeledSamples.length === 0) {
        return (
            <div className="space-y-6">
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => navigate('/admin/images')}
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <BackIcon />
                    </button>
                    <h1 className="text-xl font-medium text-foreground">Data Labeling</h1>
                </div>
                <div className="flex flex-col items-center justify-center py-16 text-center">
                    <div className="w-16 h-16 bg-success/20 rounded-full flex items-center justify-center mb-4">
                        <svg className="w-8 h-8 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                    </div>
                    <p className="text-foreground font-medium">All images labeled!</p>
                    <p className="text-sm text-muted mt-1">No unlabeled images remaining</p>
                </div>
            </div>
        );
    }

    const isInteracting = isDrawingBox || dragHandle !== null || drawStart !== null;
    const isEditable = !isDrawingBox && boundingBox;

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => navigate('/admin/images')}
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <BackIcon />
                    </button>
                    <h1 className="text-xl font-medium text-foreground">Data Labeling</h1>
                </div>
                <span className="text-sm text-secondary">
                    {currentIndex + 1} / {unlabeledSamples.length}
                </span>
            </div>

            {/* Main content */}
            <div className="max-w-2xl mx-auto space-y-4">
                {/* Navigation and image */}
                <div className="flex items-center gap-2">
                    <button
                        onClick={goToPrevious}
                        disabled={currentIndex === 0}
                        className="p-2 rounded-lg bg-surface-dim text-foreground hover:bg-primary hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        <ChevronLeftIcon />
                    </button>

                    <div
                        ref={imageContainerRef}
                        className="flex-1 relative bg-surface rounded-lg border border-border overflow-hidden"
                        onPointerDown={handlePointerDown}
                        onPointerMove={handlePointerMove}
                        onPointerUp={handlePointerUp}
                        onPointerCancel={handlePointerUp}
                        style={{
                            touchAction: isInteracting || isDrawingBox || boundingBox ? 'none' : 'auto',
                            cursor: getCursor(),
                        }}
                    >
                        <img
                            src={currentSample.thumbnailUrl || currentSample.imageUrl || `${API_URL}/api/ml/samples/${currentSample.id}/image`}
                            alt={`Sample ${currentSample.id}`}
                            className="w-full h-auto select-none"
                            draggable={false}
                        />
                        {/* Bounding box overlay */}
                        {boundingBox && (
                            <div
                                className={`absolute rounded-sm ${
                                    labelData.correction?.boundingBox
                                        ? 'border-2 border-success'
                                        : 'border-2 border-primary border-dashed'
                                } ${isEditable ? 'cursor-move' : ''}`}
                                style={{
                                    left: `${boundingBox.x * 100}%`,
                                    top: `${boundingBox.y * 100}%`,
                                    width: `${boundingBox.width * 100}%`,
                                    height: `${boundingBox.height * 100}%`,
                                    pointerEvents: isDrawingBox ? 'none' : 'auto',
                                }}
                            >
                                {isEditable && (
                                    <>
                                        <div className="absolute -left-2 -top-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nwse-resize shadow-sm" />
                                        <div className="absolute -right-2 -top-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nesw-resize shadow-sm" />
                                        <div className="absolute -left-2 -bottom-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nesw-resize shadow-sm" />
                                        <div className="absolute -right-2 -bottom-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nwse-resize shadow-sm" />
                                    </>
                                )}
                            </div>
                        )}
                        {isDrawingBox && (
                            <div className="absolute inset-0 bg-black/10 pointer-events-none" />
                        )}
                    </div>

                    <button
                        onClick={goToNext}
                        disabled={currentIndex === unlabeledSamples.length - 1}
                        className="p-2 rounded-lg bg-surface-dim text-foreground hover:bg-primary hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        <ChevronRightIcon />
                    </button>
                </div>

                {/* Keyboard hints */}
                <p className="text-xs text-muted text-center">
                    Use <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">←</kbd> <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">→</kbd> to navigate, <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">S</kbd> to save
                </p>

                {/* Action button */}
                {isDrawingBox ? (
                    <Button onClick={() => setIsDrawingBox(false)} variant="secondary" className="w-full">
                        Cancel Drawing
                    </Button>
                ) : (
                    <Button onClick={() => setIsDrawingBox(true)} variant="secondary" className="w-full">
                        Draw Bounding Box
                    </Button>
                )}

                {labelData.correction && (
                    <Button
                        onClick={handleSaveAndNext}
                        variant="success"
                        className="w-full"
                    >
                        Save & Next (S)
                    </Button>
                )}

                {/* Label toggles */}
                {labelData.correction && (
                    <div className="bg-surface rounded-lg border border-border p-4 space-y-3">
                        <div className="flex items-center justify-between gap-4">
                            <span className="text-sm text-secondary">Board detected</span>
                            <div className="flex rounded-lg overflow-hidden border border-border shrink-0">
                                <button
                                    onClick={() => { if (!currentDetection?.hasBoardDetected) toggleBoardDetected(); }}
                                    className={`px-3 py-1 text-sm font-medium transition-colors ${
                                        currentDetection?.hasBoardDetected ? 'bg-success text-white' : 'bg-surface hover:bg-surface-dim text-secondary'
                                    }`}
                                >
                                    Yes
                                </button>
                                <button
                                    onClick={() => { if (currentDetection?.hasBoardDetected) toggleBoardDetected(); }}
                                    className={`px-3 py-1 text-sm font-medium transition-colors ${
                                        !currentDetection?.hasBoardDetected ? 'bg-error text-white' : 'bg-surface hover:bg-surface-dim text-secondary'
                                    }`}
                                >
                                    No
                                </button>
                            </div>
                        </div>

                        <div className="flex items-center justify-between gap-4">
                            <span className="text-sm text-secondary">Entire board visible</span>
                            <div className={`flex rounded-lg overflow-hidden border border-border shrink-0 ${!currentDetection?.hasBoardDetected ? 'opacity-50 pointer-events-none' : ''}`}>
                                <button
                                    onClick={() => { if (!currentDetection?.isEntireBoardVisible) toggleEntireBoardVisible(); }}
                                    className={`px-3 py-1 text-sm font-medium transition-colors ${
                                        currentDetection?.isEntireBoardVisible ? 'bg-success text-white' : 'bg-surface hover:bg-surface-dim text-secondary'
                                    }`}
                                >
                                    Yes
                                </button>
                                <button
                                    onClick={() => { if (currentDetection?.isEntireBoardVisible) toggleEntireBoardVisible(); }}
                                    className={`px-3 py-1 text-sm font-medium transition-colors ${
                                        currentDetection?.hasBoardDetected && !currentDetection?.isEntireBoardVisible ? 'bg-error text-white' : 'bg-surface hover:bg-surface-dim text-secondary'
                                    }`}
                                >
                                    No
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
