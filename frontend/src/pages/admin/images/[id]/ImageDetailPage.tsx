import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { http, API_URL } from '../../../../api/config';
import { ML_API } from '../../../../api/ml';
import { Button } from '../../../../components/ui/Button';
import { Select } from '../../../../components/ui/Select';
import type { BoundingBox, BoardDetection, DetectResponse, DetectResponseWithCorrection, SampleStatusFlags, TrainingRun } from '@blokus-ai/shared';
import { getStatusLabels } from '@blokus-ai/shared';
import { statusColors, statusLabels } from '../../../../lib/status';

interface Sample {
    id: string;
    filename: string;
    status: SampleStatusFlags;
    created_at: string;
    imageUrl?: string;
    thumbnailUrl?: string;
}

interface LabelData {
    prediction: BoardDetection | null;
    correction: BoardDetection | null;
    isModified: boolean;
    isHumanLabeled: boolean;
}

type DragHandle = 'nw' | 'ne' | 'sw' | 'se' | 'move' | null;

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

export const ImageDetailPage = () => {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const [sample, setSample] = useState<Sample | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isRunningDetection, setIsRunningDetection] = useState(false);
    const [labelData, setLabelData] = useState<LabelData>({
        prediction: null,
        correction: null,
        isModified: false,
        isHumanLabeled: false,
    });
    const [isDrawingBox, setIsDrawingBox] = useState(false);
    const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
    const [dragHandle, setDragHandle] = useState<DragHandle>(null);
    const [dragStartBox, setDragStartBox] = useState<BoundingBox | null>(null);
    const [dragStartPos, setDragStartPos] = useState<{ x: number; y: number } | null>(null);
    const [isSaving, setIsSaving] = useState(false);
    const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
    const [selectedModelId, setSelectedModelId] = useState<string>('');
    const imageContainerRef = useRef<HTMLDivElement>(null);

    // Get the current detection values (correction overrides prediction)
    const currentDetection = labelData.correction || labelData.prediction;
    const boundingBox = currentDetection?.boundingBox;

    useEffect(() => {
        const loadSample = async () => {
            try {
                const data = await http.get<Sample>(`/api/ml/samples/${id}`);
                setSample(data);
                // Try to load existing detection result
                if (id) {
                    const detectResult = await http.get<DetectResponseWithCorrection | null>(`/api/ml/detect/${id}`);
                    setLabelData({
                        prediction: detectResult?.detection ?? null,
                        correction: detectResult?.correction ?? null,
                        isModified: false,
                        isHumanLabeled: detectResult?.isHumanLabeled ?? false,
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

    // Load completed training runs for model selector
    useEffect(() => {
        const loadTrainingRuns = async () => {
            try {
                const runs = await ML_API.getTrainingRuns();
                const completedRuns = runs.filter(r => r.status === 'completed');
                setTrainingRuns(completedRuns);
                // Select the most recent completed run by default
                if (completedRuns.length > 0 && !selectedModelId) {
                    setSelectedModelId(completedRuns[0].id);
                }
            } catch {
                // Ignore errors
            }
        };
        loadTrainingRuns();
    }, [selectedModelId]);

    const handleRunDetection = async () => {
        if (!id) {
            return;
        }
        if (!selectedModelId) {
            toast.error('Please select a model first');
            return;
        }
        setIsRunningDetection(true);
        try {
            const result = await http.post<DetectResponse>(`/api/ml/detect/${id}`, { model_id: selectedModelId });
            setLabelData({
                prediction: result.detection,
                correction: null,
                isModified: false,
                isHumanLabeled: false,
            });
            setSample(prev => prev ? { ...prev, status: { ...prev.status, aiLabeled: true } } : null);
            const timeStr = result.processing_time_ms.toFixed(0);
            toast.success(`Detection complete (${timeStr}ms)`);
        } catch {
            toast.error('Detection failed');
        } finally {
            setIsRunningDetection(false);
        }
    };

    const handleDeleteDetection = async () => {
        if (!id) {
            return;
        }
        try {
            await http.delete(`/api/ml/detect/${id}`);
            setLabelData({
                prediction: null,
                correction: null,
                isModified: false,
            });
            setSample(prev => prev ? { ...prev, status: { aiLabeled: false, humanLabeled: false, verified: false } } : null);
            toast.success('Detection deleted');
        } catch {
            toast.error('Failed to delete detection');
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
        if (!pos) {
            return;
        }

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
        if (!id || !labelData.correction) {
            return;
        }
        setIsSaving(true);
        try {
            await http.patch(`/api/ml/detect/${id}/correction`, {
                hasBoardDetected: labelData.correction.hasBoardDetected,
                isEntireBoardVisible: labelData.correction.isEntireBoardVisible,
                boundingBox: labelData.correction.boundingBox,
            });
            setLabelData(prev => ({ ...prev, isModified: false }));
            toast.success('Corrections saved');
        } catch {
            toast.error('Failed to save corrections');
        } finally {
            setIsSaving(false);
        }
    };

    const handleResetCorrections = () => {
        setLabelData(prev => ({
            ...prev,
            correction: null,
            isModified: false,
        }));
    };

    const handleVerify = async () => {
        if (!id) {
            return;
        }
        try {
            await http.post(`/api/ml/detect/${id}/verify`, {});
            setSample(prev => prev ? { ...prev, status: { ...prev.status, verified: true } } : null);
            toast.success('Image verified');
        } catch {
            toast.error('Failed to verify image');
        }
    };

    const handleUnverify = async () => {
        if (!id) {
            return;
        }
        try {
            await http.delete(`/api/ml/detect/${id}/verify`);
            setSample(prev => prev ? { ...prev, status: { ...prev.status, verified: false } } : null);
            toast.success('Verification removed');
        } catch {
            toast.error('Failed to remove verification');
        }
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
                <div className="aspect-video max-w-4xl bg-surface-dim rounded-lg animate-pulse" />
            </div>
        );
    }

    if (!sample) {
        return (
            <div className="space-y-6">
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => navigate(-1)}
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <BackIcon />
                    </button>
                    <h1 className="text-xl font-medium text-foreground">Image not found</h1>
                </div>
            </div>
        );
    }

    // For human-labeled images, prediction contains human label, not AI prediction
    const hasAiPrediction = labelData.prediction !== null && !labelData.isHumanLabeled;
    const hasAnyLabel = labelData.prediction !== null || labelData.correction !== null;
    const isInteracting = isDrawingBox || dragHandle !== null || drawStart !== null;
    const isEditable = !isDrawingBox && boundingBox;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="space-y-2">
                <div className="flex items-center gap-2 flex-wrap">
                    <button
                        onClick={() => navigate(-1)}
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <BackIcon />
                    </button>
                    <h1 className="text-lg sm:text-xl font-medium text-foreground font-mono">{sample.id.slice(0, 8)}</h1>
                    {getStatusLabels(sample.status).map(label => (
                        <span
                            key={label}
                            className={`px-2 py-0.5 text-xs font-medium rounded-full ${statusColors[label]}`}
                        >
                            {statusLabels[label]}
                        </span>
                    ))}
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

            {/* Content container - centered with max width */}
            <div className="max-w-2xl mx-auto space-y-6">

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
                        src={sample.thumbnailUrl || sample.imageUrl || `${API_URL}/api/ml/samples/${sample.id}/image`}
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
                                labelData.isHumanLabeled ? 'bg-info' : labelData.correction?.boundingBox ? 'bg-success' : 'bg-primary'
                            }`}>
                                {labelData.isHumanLabeled ? 'Human Labeled' : labelData.correction?.boundingBox ? 'Corrected' : 'Predicted'}
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

                {/* Model selector and action buttons */}
                <div className="flex items-end gap-2">
                    <div className="flex-1">
                        <Select
                            label="Detection Model"
                            value={selectedModelId}
                            options={trainingRuns.map(run => ({
                                value: run.id,
                                label: `${new Date(run.createdAt).toLocaleDateString()} - ${(run.accuracy! * 100).toFixed(1)}% acc`,
                            }))}
                            onChange={(val) => setSelectedModelId(val as string)}
                            placeholder={trainingRuns.length === 0 ? 'No trained models' : 'Select model...'}
                            disabled={trainingRuns.length === 0}
                        />
                    </div>
                    <Button
                        onClick={handleRunDetection}
                        loading={isRunningDetection}
                        leftIcon={hasAiPrediction ? <RefreshIcon /> : <PlayIcon />}
                        variant="secondary"
                        className="shrink-0"
                        disabled={trainingRuns.length === 0}
                    >
                        {hasAiPrediction ? 'Re-run' : 'Detect'}
                    </Button>
                </div>

                {/* Action buttons */}
                <div className="grid grid-cols-2 gap-2">
                    {isDrawingBox ? (
                        <Button
                            onClick={() => setIsDrawingBox(false)}
                            variant="secondary"
                        >
                        Cancel
                        </Button>
                    ) : (
                        <Button
                            onClick={() => setIsDrawingBox(true)}
                            variant="secondary"
                        >
                        Draw Box
                        </Button>
                    )}
                    {hasAnyLabel && (
                        sample.status.verified ? (
                            <Button
                                onClick={handleUnverify}
                                variant="secondary"
                            >
                            Unverify
                            </Button>
                        ) : (
                            <Button
                                onClick={handleVerify}
                                variant="success"
                            >
                            Verify
                            </Button>
                        )
                    )}
                    {hasAnyLabel && (
                        <Button
                            onClick={handleDeleteDetection}
                            variant="danger"
                        >
                        Delete
                        </Button>
                    )}
                    {labelData.isModified && (
                        <>
                            <Button onClick={handleSaveCorrections} loading={isSaving}>
                            Save
                            </Button>
                            <Button onClick={handleResetCorrections} variant="secondary">
                            Reset
                            </Button>
                        </>
                    )}
                </div>

                {/* Detection Info - only show if we have detection data */}
                {(labelData.prediction || labelData.correction) && (
                    <div className="bg-surface rounded-lg border border-border p-4">
                        <div className="flex items-center justify-between mb-3">
                            <h2 className="text-sm font-medium text-foreground">
                                {labelData.isHumanLabeled ? 'Human Label' : 'Board Detection'}
                            </h2>
                            {!labelData.isHumanLabeled && labelData.correction && (
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
                                            if (!currentDetection?.hasBoardDetected) {
                                                toggleBoardDetected();
                                            }
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
                                            if (currentDetection?.hasBoardDetected) {
                                                toggleBoardDetected();
                                            }
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
                                <div className={`flex rounded-lg overflow-hidden border border-border shrink-0 select-none ${!currentDetection?.hasBoardDetected ? 'opacity-50 pointer-events-none' : ''}`}>
                                    <button
                                        onClick={() => {
                                            if (!currentDetection?.isEntireBoardVisible) {
                                                toggleEntireBoardVisible();
                                            }
                                        }}
                                        className={`px-3 py-1 text-sm font-medium transition-colors ${
                                            currentDetection?.isEntireBoardVisible
                                                ? 'bg-success text-white'
                                                : 'bg-surface hover:bg-surface-dim text-secondary'
                                        }`}
                                    >
                                Yes
                                    </button>
                                    <button
                                        onClick={() => {
                                            if (currentDetection?.isEntireBoardVisible) {
                                                toggleEntireBoardVisible();
                                            }
                                        }}
                                        className={`px-3 py-1 text-sm font-medium transition-colors ${
                                            currentDetection?.hasBoardDetected && !currentDetection?.isEntireBoardVisible
                                                ? 'bg-error text-white'
                                                : 'bg-surface hover:bg-surface-dim text-secondary'
                                        }`}
                                    >
                                No
                                    </button>
                                </div>
                            </div>

                            {/* Confidence - only show when board is detected */}
                            {labelData.prediction?.hasBoardDetected && labelData.prediction?.confidence !== undefined && (
                                <div className="flex items-center justify-between gap-4">
                                    <span className="text-sm text-secondary">Confidence</span>
                                    <span className="text-sm font-mono text-foreground">
                                        {(labelData.prediction.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                            )}

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
                )}

            </div>
        </div>
    );
};
