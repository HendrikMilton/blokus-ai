import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import toast from 'react-hot-toast';
import { http, API_URL } from '../../../../api/config';
import { Button } from '../../../../components/ui/Button';

interface Sample {
    id: string;
    filename: string;
    status: 'pending' | 'labeled' | 'verified';
    created_at: string;
}

interface BoundingBox {
    x: number;
    y: number;
    width: number;
    height: number;
}

interface BoardDetection {
    hasBoardDetected: boolean;
    isEntireBoardVisible: boolean;
    boundingBox: BoundingBox | null;
}

interface LabelData {
    prediction: BoardDetection | null;
    correction: BoardDetection | null;
    isModified: boolean;
}

type DragHandle = 'nw' | 'ne' | 'sw' | 'se' | 'move' | null;

// Mock detection data - will be replaced with real API call
const getMockDetection = (sampleId: string): BoardDetection => {
    const hash = sampleId.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
    const hasBoardDetected = hash % 3 !== 0;
    const isEntireBoardVisible = hasBoardDetected && hash % 2 === 0;

    return {
        hasBoardDetected,
        isEntireBoardVisible,
        boundingBox: hasBoardDetected ? {
            x: 0.1 + (hash % 10) / 100,
            y: 0.05 + (hash % 15) / 100,
            width: 0.7 + (hash % 10) / 100,
            height: 0.7 + (hash % 10) / 100,
        } : null,
    };
};

const BackIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
    </svg>
);

const PlayIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);

const RefreshIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
);

const statusColors: Record<string, string> = {
    pending: 'bg-warning/20 text-warning',
    labeled: 'bg-primary/20 text-primary',
    verified: 'bg-success/20 text-success',
};

export const ImageDetailPage = () => {
    const { id } = useParams<{ id: string }>();
    const [sample, setSample] = useState<Sample | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isRunningDetection, setIsRunningDetection] = useState(false);
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
    const [isSaving, setIsSaving] = useState(false);
    const imageContainerRef = useRef<HTMLDivElement>(null);

    // Get the current detection values (correction overrides prediction)
    const currentDetection = labelData.correction || labelData.prediction;
    const boundingBox = currentDetection?.boundingBox;

    useEffect(() => {
        const loadSample = async () => {
            try {
                const data = await http.get<Sample>(`/api/ml/samples/${id}`);
                setSample(data);
                // Load mock detection data as prediction
                if (id) {
                    setLabelData({
                        prediction: getMockDetection(id),
                        correction: null,
                        isModified: false,
                    });
                }
            } catch {
                // Error toast shown by http client
            } finally {
                setIsLoading(false);
            }
        };
        loadSample();
    }, [id]);

    const handleRunDetection = async () => {
        setIsRunningDetection(true);
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1500));
        if (id) {
            const newPrediction = getMockDetection(id + Date.now());
            setLabelData({
                prediction: newPrediction,
                correction: null,
                isModified: false,
            });
        }
        setIsRunningDetection(false);
        toast.success('Detection complete');
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
        // Clamp values to 0-1 range and ensure min size
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
            // If setting to false, also clear bounding box and entire board visible
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

        // Drawing new box mode
        if (isDrawingBox) {
            e.preventDefault();
            setDrawStart(pos);
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            return;
        }

        // Check if clicking on existing box handles or center
        if (boundingBox) {
            const handleSize = 0.03; // 3% of image size for touch targets
            const box = boundingBox;

            // Check corners (larger touch targets)
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

            // Check if inside box for move
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

        // Drawing new box
        if (isDrawingBox && drawStart) {
            e.preventDefault();
            const x = Math.min(drawStart.x, pos.x);
            const y = Math.min(drawStart.y, pos.y);
            const width = Math.abs(pos.x - drawStart.x);
            const height = Math.abs(pos.y - drawStart.y);
            updateBoundingBox({ x, y, width, height });
            return;
        }

        // Dragging existing box
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
                    newBox = {
                        x: dragStartBox.x + dx,
                        y: dragStartBox.y + dy,
                        width: dragStartBox.width - dx,
                        height: dragStartBox.height - dy,
                    };
                    break;
                case 'ne':
                    newBox = {
                        x: dragStartBox.x,
                        y: dragStartBox.y + dy,
                        width: dragStartBox.width + dx,
                        height: dragStartBox.height - dy,
                    };
                    break;
                case 'sw':
                    newBox = {
                        x: dragStartBox.x + dx,
                        y: dragStartBox.y,
                        width: dragStartBox.width - dx,
                        height: dragStartBox.height + dy,
                    };
                    break;
                case 'se':
                    newBox = {
                        x: dragStartBox.x,
                        y: dragStartBox.y,
                        width: dragStartBox.width + dx,
                        height: dragStartBox.height + dy,
                    };
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

    const handleSaveCorrections = async () => {
        setIsSaving(true);
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 800));
        setLabelData(prev => ({ ...prev, isModified: false }));
        setIsSaving(false);
        toast.success('Corrections saved');
    };

    const handleResetCorrections = () => {
        setLabelData(prev => ({
            ...prev,
            correction: null,
            isModified: false,
        }));
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    // Determine cursor based on position
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
                <div className="aspect-video max-w-4xl bg-surface-dim rounded-lg animate-pulse" />
            </div>
        );
    }

    if (!sample) {
        return (
            <div className="space-y-6">
                <div className="flex items-center gap-3">
                    <Link
                        to="/admin/images"
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <BackIcon />
                    </Link>
                    <h1 className="text-xl font-medium text-foreground">Image not found</h1>
                </div>
            </div>
        );
    }

    const hasPrediction = labelData.prediction !== null;
    const isInteracting = isDrawingBox || dragHandle !== null || drawStart !== null;
    const isEditable = !isDrawingBox && boundingBox;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="space-y-2">
                <div className="flex items-center gap-2 flex-wrap">
                    <Link
                        to="/admin/images"
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <BackIcon />
                    </Link>
                    <h1 className="text-lg sm:text-xl font-medium text-foreground font-mono">{sample.id.slice(0, 8)}</h1>
                    <span className={`px-2 py-0.5 text-xs font-medium rounded-full capitalize ${statusColors[sample.status]}`}>
                        {sample.status}
                    </span>
                    {labelData.isModified && (
                        <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-warning/20 text-warning">
                            Unsaved
                        </span>
                    )}
                </div>
                <p className="text-xs sm:text-sm text-secondary">
                    {formatDate(sample.created_at)}
                </p>
            </div>

            {/* Image with bounding box overlay */}
            <div
                ref={imageContainerRef}
                className="relative bg-surface rounded-lg border border-border overflow-hidden"
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
                    src={`${API_URL}/api/ml/samples/${sample.id}/image`}
                    alt={`Sample ${sample.id}`}
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
                        {/* Label */}
                        <div className={`absolute -top-6 left-0 text-white text-xs px-1.5 py-0.5 rounded pointer-events-none ${
                            labelData.correction?.boundingBox ? 'bg-success' : 'bg-primary'
                        }`}>
                            {labelData.correction?.boundingBox ? 'Corrected' : 'Predicted'}
                        </div>

                        {/* Corner handles - only show when not in draw mode */}
                        {isEditable && (
                            <>
                                {/* NW corner */}
                                <div
                                    className="absolute -left-2 -top-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nwse-resize shadow-sm"
                                    style={{ pointerEvents: 'auto' }}
                                />
                                {/* NE corner */}
                                <div
                                    className="absolute -right-2 -top-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nesw-resize shadow-sm"
                                    style={{ pointerEvents: 'auto' }}
                                />
                                {/* SW corner */}
                                <div
                                    className="absolute -left-2 -bottom-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nesw-resize shadow-sm"
                                    style={{ pointerEvents: 'auto' }}
                                />
                                {/* SE corner */}
                                <div
                                    className="absolute -right-2 -bottom-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nwse-resize shadow-sm"
                                    style={{ pointerEvents: 'auto' }}
                                />
                            </>
                        )}
                    </div>
                )}
                {/* Drawing mode overlay */}
                {isDrawingBox && (
                    <div className="absolute inset-0 bg-black/10 pointer-events-none" />
                )}
            </div>

            {/* Hint text */}
            {isEditable && (
                <p className="text-xs text-muted">
                    Drag corners to resize, drag center to move
                </p>
            )}

            {/* Action buttons */}
            <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
                <div className="flex gap-2 sm:gap-3">
                    <Button
                        onClick={handleRunDetection}
                        loading={isRunningDetection}
                        leftIcon={hasPrediction ? <RefreshIcon /> : <PlayIcon />}
                        variant="secondary"
                        className="flex-1 sm:flex-none"
                    >
                        {hasPrediction ? 'Re-run' : 'Detect'}
                    </Button>
                    {isDrawingBox ? (
                        <Button
                            onClick={() => setIsDrawingBox(false)}
                            variant="secondary"
                            className="flex-1 sm:flex-none"
                        >
                            Cancel
                        </Button>
                    ) : (
                        <Button
                            onClick={() => setIsDrawingBox(true)}
                            variant="secondary"
                            className="flex-1 sm:flex-none"
                        >
                            Draw Box
                        </Button>
                    )}
                </div>
                {labelData.isModified && (
                    <div className="flex gap-2 sm:gap-3">
                        <Button onClick={handleSaveCorrections} loading={isSaving} className="flex-1 sm:flex-none">
                            Save
                        </Button>
                        <Button onClick={handleResetCorrections} variant="secondary" className="flex-1 sm:flex-none">
                            Reset
                        </Button>
                    </div>
                )}
            </div>

            {/* Detection Info */}
            <div className="bg-surface rounded-lg border border-border p-4">
                <div className="flex items-center justify-between mb-3">
                    <h2 className="text-sm font-medium text-foreground">Board Detection</h2>
                    {labelData.correction && (
                        <span className="text-xs text-success font-medium">Corrected</span>
                    )}
                </div>
                <div className="space-y-3">
                    {/* Board detected */}
                    <div className="flex items-center justify-between gap-4">
                        <span className="text-sm text-secondary">Board detected</span>
                        <div className="flex rounded-lg overflow-hidden border border-border shrink-0">
                            <button
                                onClick={() => {
                                    if (!currentDetection?.hasBoardDetected) toggleBoardDetected();
                                }}
                                className={`px-3 py-1 text-sm font-medium transition-colors ${
                                    currentDetection?.hasBoardDetected
                                        ? 'bg-success text-white'
                                        : 'bg-surface hover:bg-surface-dim text-secondary'
                                }`}
                            >
                                Yes
                            </button>
                            <button
                                onClick={() => {
                                    if (currentDetection?.hasBoardDetected) toggleBoardDetected();
                                }}
                                className={`px-3 py-1 text-sm font-medium transition-colors ${
                                    !currentDetection?.hasBoardDetected
                                        ? 'bg-error text-white'
                                        : 'bg-surface hover:bg-surface-dim text-secondary'
                                }`}
                            >
                                No
                            </button>
                        </div>
                    </div>

                    {/* Entire board visible */}
                    <div className="flex items-center justify-between gap-4">
                        <span className="text-sm text-secondary">Entire board visible</span>
                        <div className="flex rounded-lg overflow-hidden border border-border shrink-0">
                            <button
                                onClick={() => {
                                    if (!currentDetection?.isEntireBoardVisible) toggleEntireBoardVisible();
                                }}
                                disabled={!currentDetection?.hasBoardDetected}
                                className={`px-3 py-1 text-sm font-medium transition-colors ${
                                    currentDetection?.isEntireBoardVisible
                                        ? 'bg-success text-white'
                                        : 'bg-surface hover:bg-surface-dim text-secondary disabled:opacity-50 disabled:cursor-not-allowed'
                                }`}
                            >
                                Yes
                            </button>
                            <button
                                onClick={() => {
                                    if (currentDetection?.isEntireBoardVisible) toggleEntireBoardVisible();
                                }}
                                disabled={!currentDetection?.hasBoardDetected}
                                className={`px-3 py-1 text-sm font-medium transition-colors ${
                                    currentDetection?.hasBoardDetected && !currentDetection?.isEntireBoardVisible
                                        ? 'bg-error text-white'
                                        : 'bg-surface hover:bg-surface-dim text-secondary disabled:opacity-50 disabled:cursor-not-allowed'
                                }`}
                            >
                                No
                            </button>
                        </div>
                    </div>

                    {/* Bounding box coordinates */}
                    {boundingBox && (
                        <div className="pt-2 border-t border-border">
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-muted">Bounding box</span>
                                {labelData.correction?.boundingBox && (
                                    <button
                                        onClick={() => updateCorrection({ boundingBox: null })}
                                        className="text-xs text-error hover:underline"
                                    >
                                        Clear
                                    </button>
                                )}
                            </div>
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                                <div className="bg-surface-dim rounded px-2 py-1">
                                    <span className="text-muted">x:</span>{' '}
                                    <span className="text-foreground">{(boundingBox.x * 100).toFixed(1)}%</span>
                                </div>
                                <div className="bg-surface-dim rounded px-2 py-1">
                                    <span className="text-muted">y:</span>{' '}
                                    <span className="text-foreground">{(boundingBox.y * 100).toFixed(1)}%</span>
                                </div>
                                <div className="bg-surface-dim rounded px-2 py-1">
                                    <span className="text-muted">w:</span>{' '}
                                    <span className="text-foreground">{(boundingBox.width * 100).toFixed(1)}%</span>
                                </div>
                                <div className="bg-surface-dim rounded px-2 py-1">
                                    <span className="text-muted">h:</span>{' '}
                                    <span className="text-foreground">{(boundingBox.height * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
