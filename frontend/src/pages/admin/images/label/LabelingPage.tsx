import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { http, API_URL } from '../../../../api/config';
import { Button } from '../../../../components/ui/Button';
import type { BoundingBox, BoardDetection, SampleStatusFlags } from '@blokus-ai/shared';
import { getStatusLabels } from '@blokus-ai/shared';
import { statusColors, statusLabels, isUnlabeled } from '../../../../lib/status';

interface Sample {
    id: string;
    filename: string;
    status: SampleStatusFlags;
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
    const [samples, setSamples] = useState<Sample[]>([]);
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

    const currentSample = samples[currentIndex] || null;
    const currentDetection = labelData.correction || labelData.prediction;
    const boundingBox = currentDetection?.boundingBox;

    // Load all samples
    useEffect(() => {
        const loadSamples = async () => {
            try {
                const data = await http.get<SamplesResponse>('/api/ml/samples');
                setSamples(data.samples || []);
            } catch {
                // Error toast shown by http client
            } finally {
                setIsLoading(false);
            }
        };
        loadSamples();
    }, []);

    // Load existing label when sample changes
    useEffect(() => {
        const loadExistingLabel = async () => {
            if (!currentSample) {
                return;
            }

            // Reset first
            setLabelData({
                prediction: null,
                correction: null,
                isModified: false,
            });
            setIsDrawingBox(false);

            // If sample has any labels, load the existing detection
            if (!isUnlabeled(currentSample.status)) {
                try {
                    const data = await http.get<{ detection: BoardDetection; correction: BoardDetection | null } | null>(`/api/ml/detect/${currentSample.id}`);
                    if (data) {
                        setLabelData({
                            prediction: data.detection,
                            correction: data.correction,
                            isModified: false,
                        });
                    }
                } catch {
                    // Ignore errors
                }
            }
        };
        loadExistingLabel();
    }, [currentSample]);

    const goToPrevious = () => {
        if (currentIndex > 0) {
            setCurrentIndex(currentIndex - 1);
        }
    };

    const goToNext = () => {
        if (currentIndex < samples.length - 1) {
            setCurrentIndex(currentIndex + 1);
        }
    };

    const handleSaveAndNext = async () => {
        if (!currentSample || !labelData.correction) {
            return;
        }

        try {
            // Save the manual label
            await http.post(`/api/ml/samples/${currentSample.id}/label`, {
                hasBoardDetected: labelData.correction.hasBoardDetected,
                isEntireBoardVisible: labelData.correction.isEntireBoardVisible,
                boundingBox: labelData.correction.boundingBox,
            });
            toast.success('Saved!');

            // Update sample status locally (human labeled)
            setSamples(prev => prev.map(s =>
                s.id === currentSample.id ? { ...s, status: { ...s.status, humanLabeled: true } } : s
            ));

            // Reset label data and move to next image
            setLabelData({
                prediction: null,
                correction: null,
                isModified: false,
            });
            setIsDrawingBox(false);

            // Move to next if not at the end
            if (currentIndex < samples.length - 1) {
                setCurrentIndex(currentIndex + 1);
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
            } else if (e.key === 'd' || e.key === 'D') {
                e.preventDefault();
                setIsDrawingBox(prev => !prev);
            } else if (e.key === '1') {
                e.preventDefault();
                if (currentDetection?.hasBoardDetected !== true) {
                    updateCorrection({ hasBoardDetected: true });
                }
            } else if (e.key === '2') {
                e.preventDefault();
                if (currentDetection?.hasBoardDetected !== false) {
                    updateCorrection({ hasBoardDetected: false, isEntireBoardVisible: false, boundingBox: null });
                }
            } else if (e.key === '3') {
                e.preventDefault();
                if (currentDetection?.hasBoardDetected && currentDetection?.isEntireBoardVisible !== true) {
                    updateCorrection({ isEntireBoardVisible: true });
                }
            } else if (e.key === '4') {
                e.preventDefault();
                if (currentDetection?.hasBoardDetected && currentDetection?.isEntireBoardVisible !== false) {
                    updateCorrection({ isEntireBoardVisible: false });
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [currentIndex, samples.length, labelData.correction, currentSample, currentDetection, updateCorrection, goToPrevious, goToNext, handleSaveAndNext]);

    const getRelativePosition = useCallback((clientX: number, clientY: number) => {
        if (!imageContainerRef.current) {
            return null;
        }
        const rect = imageContainerRef.current.getBoundingClientRect();
        return {
            x: Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
            y: Math.max(0, Math.min(1, (clientY - rect.top) / rect.height)),
        };
    }, []);

    const handlePointerDown = (e: React.PointerEvent) => {
        const pos = getRelativePosition(e.clientX, e.clientY);
        if (!pos) {
            return;
        }

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
        if (!pos) {
            return;
        }

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
        if (isDrawingBox) {
            return 'crosshair';
        }
        if (dragHandle === 'move') {
            return 'move';
        }
        if (dragHandle === 'nw' || dragHandle === 'se') {
            return 'nwse-resize';
        }
        if (dragHandle === 'ne' || dragHandle === 'sw') {
            return 'nesw-resize';
        }
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

    if (samples.length === 0) {
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
                    <div className="w-16 h-16 bg-surface-dim rounded-full flex items-center justify-center mb-4">
                        <svg className="w-8 h-8 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                    </div>
                    <p className="text-foreground font-medium">No images</p>
                    <p className="text-sm text-muted mt-1">Upload images to get started</p>
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
                <div className="flex items-center gap-2">
                    {getStatusLabels(currentSample.status).map(label => (
                        <span
                            key={label}
                            className={`px-2 py-0.5 text-xs font-medium rounded-full ${statusColors[label]}`}
                        >
                            {statusLabels[label]}
                        </span>
                    ))}
                    <span className="text-sm text-secondary ml-1">
                        {currentIndex + 1} / {samples.length}
                    </span>
                </div>
            </div>

            {/* Main content */}
            <div className="max-w-2xl mx-auto space-y-4">
                {/* Navigation and image */}
                <div className="flex items-center gap-2">
                    <button
                        onClick={goToPrevious}
                        disabled={currentIndex === 0}
                        className="p-2 rounded-lg bg-surface-dim text-foreground enabled:hover:bg-primary enabled:hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
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
                        disabled={currentIndex === samples.length - 1}
                        className="p-2 rounded-lg bg-surface-dim text-foreground enabled:hover:bg-primary enabled:hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    >
                        <ChevronRightIcon />
                    </button>
                </div>

                {/* Keyboard hints */}
                <p className="text-xs text-muted text-center">
                    <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">←</kbd> <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">→</kbd> navigate · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">D</kbd> draw · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">1</kbd><kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">2</kbd> board · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">3</kbd><kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">4</kbd> visible · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">S</kbd> save
                </p>

                {/* Action button */}
                {isDrawingBox ? (
                    <Button onClick={() => setIsDrawingBox(false)} variant="secondary" className="w-full">
                        Cancel Drawing (D)
                    </Button>
                ) : (
                    <Button onClick={() => setIsDrawingBox(true)} variant="secondary" className="w-full">
                        Draw Bounding Box (D)
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
                <div className="bg-surface rounded-lg border border-border p-4 space-y-3">
                    <div className="flex items-center justify-between gap-4">
                        <span className="text-sm text-secondary">Board detected</span>
                        <div className="flex rounded-lg overflow-hidden border border-border shrink-0">
                            <button
                                onClick={() => {
                                    if (!currentDetection?.hasBoardDetected) {
                                        toggleBoardDetected();
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
                                        toggleEntireBoardVisible();
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
            </div>
        </div>
    );
};
