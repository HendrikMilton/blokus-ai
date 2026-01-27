import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { ChevronLeft, ChevronRight, Image } from 'lucide-react';
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

export const LabelingPage = () => {
    const navigate = useNavigate();
    const { id: initialId } = useParams<{ id: string }>();
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
    const [hideBox, setHideBox] = useState(false);
    const fastModeRef = useRef(false);
    const imageContainerRef = useRef<HTMLDivElement>(null);

    const currentSample = samples[currentIndex] || null;
    const currentDetection = labelData.correction || labelData.prediction;
    const boundingBox = currentDetection?.boundingBox;

    // Load all samples once on mount
    useEffect(() => {
        const loadSamples = async () => {
            try {
                const data = await http.get<SamplesResponse>('/api/ml/samples');
                const loadedSamples = data.samples || [];
                // Sort: unlabeled first, then not human-labeled, then human-labeled
                // Within each group, sort by created_at descending
                loadedSamples.sort((a, b) => {
                    const aUnlabeled = isUnlabeled(a.status);
                    const bUnlabeled = isUnlabeled(b.status);
                    if (aUnlabeled !== bUnlabeled) {
                        return aUnlabeled ? -1 : 1;
                    }
                    const aHumanLabeled = a.status.humanLabeled;
                    const bHumanLabeled = b.status.humanLabeled;
                    if (aHumanLabeled !== bHumanLabeled) {
                        return aHumanLabeled ? 1 : -1;
                    }
                    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
                });
                setSamples(loadedSamples);

                // Set initial index based on URL param
                if (loadedSamples.length > 0) {
                    if (initialId) {
                        const index = loadedSamples.findIndex(s => s.id === initialId);
                        if (index >= 0) {
                            setCurrentIndex(index);
                        }
                    } else {
                        // No ID in URL, update URL to first sample
                        navigate(`/admin/images/label/${loadedSamples[0].id}`, { replace: true });
                    }
                }
            } catch {
                // Error toast shown by http client
            } finally {
                setIsLoading(false);
            }
        };
        loadSamples();
     
    }, []); // Only run once on mount

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

    const goToPrevious = useCallback(() => {
        if (currentIndex > 0) {
            const newIndex = currentIndex - 1;
            setCurrentIndex(newIndex);
            navigate(`/admin/images/label/${samples[newIndex].id}`, { replace: true });
        }
    }, [currentIndex, samples, navigate]);

    const goToNext = useCallback(() => {
        if (currentIndex < samples.length - 1) {
            const newIndex = currentIndex + 1;
            setCurrentIndex(newIndex);
            navigate(`/admin/images/label/${samples[newIndex].id}`, { replace: true });
        }
    }, [currentIndex, samples, navigate]);

    const saveAndNext = useCallback(async (labelToSave: BoardDetection) => {
        if (!currentSample) {
            return;
        }

        try {
            // Save the manual label
            await http.post(`/api/ml/samples/${currentSample.id}/label`, {
                hasBoardDetected: labelToSave.hasBoardDetected,
                isEntireBoardVisible: labelToSave.isEntireBoardVisible,
                boundingBox: labelToSave.boundingBox,
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
                const newIndex = currentIndex + 1;
                setCurrentIndex(newIndex);
                navigate(`/admin/images/label/${samples[newIndex].id}`, { replace: true });
            }
        } catch {
            toast.error('Failed to save');
        }
    }, [currentSample, currentIndex, samples, navigate]);

    const handleSaveAndNext = useCallback(async () => {
        if (!labelData.correction) {
            return;
        }
        await saveAndNext(labelData.correction);
    }, [labelData.correction, saveAndNext]);

    const handleDeleteLabel = async () => {
        if (!currentSample) {
            return;
        }

        // Only delete if there's existing label data
        if (isUnlabeled(currentSample.status)) {
            toast.error('No label to delete');
            return;
        }

        try {
            await http.delete(`/api/ml/detect/${currentSample.id}`);
            toast.success('Label deleted');

            // Update sample status locally (reset to unlabeled)
            setSamples(prev => prev.map(s =>
                s.id === currentSample.id ? { ...s, status: { aiLabeled: false, humanLabeled: false, verified: false } } : s
            ));

            // Reset label data
            setLabelData({
                prediction: null,
                correction: null,
                isModified: false,
            });
            setIsDrawingBox(false);
        } catch {
            toast.error('Failed to delete label');
        }
    };

    const updateCorrection = useCallback((updates: Partial<BoardDetection>) => {
        setLabelData(prev => {
            const base: BoardDetection = prev.correction || prev.prediction || {
                hasBoardDetected: false,
                isEntireBoardVisible: false,
                boundingBox: null,
                confidence: 0,
                classifierConfidence: null,
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

            const key = e.key.toLowerCase();
            const step = fastModeRef.current ? 0.04 : 0.01; // 4x with F held, 1% normally

            // Track F key for fast mode
            if (key === 'f') {
                fastModeRef.current = true;
                return;
            }

            // Draw mode: Arrow keys control box
            if (isDrawingBox) {
                if (key === 'escape') {
                    e.preventDefault();
                    setIsDrawingBox(false);
                    return;
                }

                if (key === 'enter') {
                    e.preventDefault();
                    setIsDrawingBox(false);
                    // If we have a correction with bbox, save
                    if (currentSample && labelData.correction) {
                        handleSaveAndNext();
                    }
                    return;
                }

                // Arrow keys for move/resize
                if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)) {
                    e.preventDefault();
                    const currentBox = boundingBox || { x: 0.25, y: 0.25, width: 0.5, height: 0.5 };

                    if (e.shiftKey) {
                        // Shift + Arrows = resize (bottom/right edges)
                        // Shift + Ctrl + Arrows = resize (top/left edges)
                        const newBox = { ...currentBox };
                        if (e.ctrlKey || e.metaKey) {
                            // Ctrl/Cmd: move top/left edges
                            if (key === 'arrowup') {
                                // Grow by moving top edge up
                                const newY = Math.max(0, currentBox.y - step);
                                newBox.height = currentBox.height + (currentBox.y - newY);
                                newBox.y = newY;
                            } else if (key === 'arrowdown') {
                                // Shrink by moving top edge down
                                const maxMove = currentBox.height - 0.05;
                                const move = Math.min(step, maxMove);
                                newBox.y = currentBox.y + move;
                                newBox.height = currentBox.height - move;
                            } else if (key === 'arrowleft') {
                                // Grow by moving left edge left
                                const newX = Math.max(0, currentBox.x - step);
                                newBox.width = currentBox.width + (currentBox.x - newX);
                                newBox.x = newX;
                            } else if (key === 'arrowright') {
                                // Shrink by moving left edge right
                                const maxMove = currentBox.width - 0.05;
                                const move = Math.min(step, maxMove);
                                newBox.x = currentBox.x + move;
                                newBox.width = currentBox.width - move;
                            }
                        } else {
                            // No Ctrl: move bottom/right edges
                            if (key === 'arrowup') {
                                // Shrink height from bottom
                                newBox.height = Math.max(0.05, currentBox.height - step);
                            } else if (key === 'arrowdown') {
                                // Grow height
                                newBox.height = Math.min(1 - currentBox.y, currentBox.height + step);
                            } else if (key === 'arrowleft') {
                                // Shrink width from right
                                newBox.width = Math.max(0.05, currentBox.width - step);
                            } else if (key === 'arrowright') {
                                // Grow width
                                newBox.width = Math.min(1 - currentBox.x, currentBox.width + step);
                            }
                        }
                        updateBoundingBox(newBox);
                    } else {
                        // Arrows = move
                        const newBox = { ...currentBox };
                        if (key === 'arrowup') {
                            newBox.y = Math.max(0, currentBox.y - step);
                        } else if (key === 'arrowdown') {
                            newBox.y = Math.min(1 - currentBox.height, currentBox.y + step);
                        } else if (key === 'arrowleft') {
                            newBox.x = Math.max(0, currentBox.x - step);
                        } else if (key === 'arrowright') {
                            newBox.x = Math.min(1 - currentBox.width, currentBox.x + step);
                        }
                        updateBoundingBox(newBox);
                    }
                    return;
                }
            }

            // Normal mode shortcuts (arrows navigate images when not in draw mode)
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                goToPrevious();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                goToNext();
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (currentSample && labelData.correction) {
                    handleSaveAndNext();
                }
            } else if (key === 'd') {
                e.preventDefault();
                // Enter draw mode and create initial box if none exists
                if (!isDrawingBox) {
                    setIsDrawingBox(true);
                    if (!boundingBox) {
                        // Create centered default box
                        updateBoundingBox({ x: 0.25, y: 0.25, width: 0.5, height: 0.5 });
                    }
                }
            } else if (e.key === '1') {
                e.preventDefault();
                if (currentDetection?.hasBoardDetected !== true) {
                    updateCorrection({ hasBoardDetected: true });
                }
            } else if (e.key === '2') {
                e.preventDefault();
                // No board detected - auto save and move to next
                saveAndNext({
                    hasBoardDetected: false,
                    isEntireBoardVisible: false,
                    boundingBox: null,
                    confidence: 1.0,
                });
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
            } else if (key === 'x') {
                e.preventDefault();
                if (currentSample && !isUnlabeled(currentSample.status)) {
                    handleDeleteLabel();
                }
            } else if (key === 'h') {
                e.preventDefault();
                setHideBox(prev => !prev);
            }
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            if (e.key.toLowerCase() === 'f') {
                fastModeRef.current = false;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        };
    }, [currentIndex, samples.length, labelData.correction, currentSample, currentDetection, updateCorrection, goToPrevious, goToNext, handleSaveAndNext, handleDeleteLabel, isDrawingBox, boundingBox, updateBoundingBox, saveAndNext]);

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

        // Check for handle interactions first (even in draw mode)
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

            // Check for move (clicking inside the box)
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

        // If in draw mode and clicked outside existing box, start drawing new box
        if (isDrawingBox) {
            e.preventDefault();
            setDrawStart(pos);
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            return;
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
            // Stay in draw mode so user can adjust with arrows
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
                        <ChevronLeft className="w-4 h-4" />
                    </button>
                    <h1 className="text-xl font-medium text-foreground">Data Labeling</h1>
                </div>
                <div className="flex flex-col items-center justify-center py-16 text-center">
                    <div className="w-16 h-16 bg-surface-dim rounded-full flex items-center justify-center mb-4">
                        <Image className="w-8 h-8 text-muted" />
                    </div>
                    <p className="text-foreground font-medium">No images</p>
                    <p className="text-sm text-muted mt-1">Upload images to get started</p>
                </div>
            </div>
        );
    }

    const isInteracting = isDrawingBox || dragHandle !== null || drawStart !== null;
    // Show handles when there's a box and we're not actively drawing a new one with mouse
    const isEditable = boundingBox && !drawStart;

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => navigate('/admin/images')}
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <ChevronLeft className="w-4 h-4" />
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
                        <ChevronLeft className="w-5 h-5" />
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
                        {boundingBox && !hideBox && (
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
                                    pointerEvents: drawStart ? 'none' : 'auto',
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
                        <ChevronRight className="w-5 h-5" />
                    </button>
                </div>

                {/* Keyboard hints */}
                {isDrawingBox ? (
                    <div className="text-xs text-center space-y-1">
                        <p className="text-primary font-medium">Draw Mode</p>
                        <p className="text-muted">
                            <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">Arrows</kbd> move · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">Shift</kbd> resize · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">Ctrl+Shift</kbd> resize top/left · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">Enter</kbd> confirm · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">Esc</kbd> cancel
                        </p>
                    </div>
                ) : (
                    <p className="text-xs text-muted text-center">
                        <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">←</kbd> <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">→</kbd> navigate · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">D</kbd> draw · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">H</kbd> hide · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">1</kbd><kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">2</kbd> board · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">3</kbd><kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">4</kbd> visible · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">Enter</kbd> save · <kbd className="px-1.5 py-0.5 bg-surface-dim rounded text-[10px]">X</kbd> delete
                    </p>
                )}

                {/* Action buttons */}
                <div className="flex gap-2">
                    {isDrawingBox ? (
                        <Button onClick={() => setIsDrawingBox(false)} variant="secondary" className="flex-1">
                            Cancel Drawing (Esc)
                        </Button>
                    ) : (
                        <Button onClick={() => {
                            setIsDrawingBox(true);
                            if (!boundingBox) {
                                updateBoundingBox({ x: 0.25, y: 0.25, width: 0.5, height: 0.5 });
                            }
                        }} variant="secondary" className="flex-1">
                            Draw Bounding Box (D)
                        </Button>
                    )}

                    {!isUnlabeled(currentSample.status) && (
                        <Button
                            onClick={handleDeleteLabel}
                            variant="danger"
                            className="shrink-0"
                        >
                            Delete (X)
                        </Button>
                    )}
                </div>

                {labelData.correction && (
                    <Button
                        onClick={handleSaveAndNext}
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
