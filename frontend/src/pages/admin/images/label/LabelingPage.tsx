import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { ChevronLeft, ChevronRight, Image, ExternalLink } from 'lucide-react';
import { http } from '../../../../api/config';
import { ML_API } from '../../../../api/ml';
import type { BoundingBox, BoardDetection, CornerDetectionResult, GridLabels, PlayerColor, GridPredictionRecord, Tag, BoardCorners } from '@blokus-ai/shared';
import { getStatusLabels } from '@blokus-ai/shared';
import { statusColors, statusLabels, isUnlabeled } from '../../../../lib/status';
import { OriginalImagePanel } from './OriginalImagePanel';
import { GridLabelingPanel } from './GridLabelingPanel';
import { LabelingControls } from './LabelingControls';
import { PiecesPanel } from './PiecesPanel';
import { SampleTagsSection } from '../[id]/SampleTagsSection';
import { CORNER_CONFIG, createEmptyGrid, type Sample, type SamplesResponse, type LabelData, type DragHandle, type PartialCorners } from './types';

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

    // Corner labeling state
    const [isCornerMode, setIsCornerMode] = useState(false);
    const [pendingCorners, setPendingCorners] = useState<PartialCorners>({});
    const [savedCorners, setSavedCorners] = useState<BoardCorners | null>(null);
    const [cornerCount, setCornerCount] = useState(0);

    // Corner detection result with cropped images
    const [cornerDetection, setCornerDetection] = useState<CornerDetectionResult | null>(null);

    // Grid labeling state
    const [isGridMode, setIsGridMode] = useState(false);
    const [gridLabels, setGridLabels] = useState<GridLabels | null>(null);
    const [selectedColor, setSelectedColor] = useState<PlayerColor | null>(null);
    const [isGridModified, setIsGridModified] = useState(false);
    const [gridPrediction, setGridPrediction] = useState<GridPredictionRecord | null>(null);

    // Tags state
    const [allTags, setAllTags] = useState<Tag[]>([]);

    const currentSample = samples[currentIndex] || null;
    const currentDetection = labelData.correction || labelData.prediction;
    const boundingBox = currentDetection?.boundingBox;

    // Load all samples once on mount
    useEffect(() => {
        const loadSamples = async () => {
            try {
                const data = await http.get<SamplesResponse>('/api/samples');
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

        // Load all tags for tag selector
        const loadTags = async () => {
            try {
                const tags = await ML_API.getTags();
                setAllTags(tags);
            } catch {
                // Ignore error
            }
        };
        loadTags();

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
            // Reset corner mode
            setIsCornerMode(false);
            setPendingCorners({});
            setCornerCount(0);
            // Reset grid mode
            setIsGridMode(false);
            setGridLabels(null);
            setIsGridModified(false);
            setGridPrediction(null);

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

            // Load corners
            try {
                const result = await ML_API.getCorners(currentSample.id);
                setSavedCorners(result.corners);
            } catch {
                setSavedCorners(null);
            }

            // Load corner detection result (with cropped images)
            try {
                const results = await ML_API.getCornerDetectionsForSamples([currentSample.id]);
                setCornerDetection(results.length > 0 ? results[0] : null);
            } catch {
                setCornerDetection(null);
            }

            // Load grid labels
            try {
                const grid = await ML_API.getGridLabel(currentSample.id);
                setGridLabels(grid);
            } catch {
                setGridLabels(null);
            }

            // Load grid prediction (AI)
            try {
                const predictions = await ML_API.getGridPredictions([currentSample.id]);
                setGridPrediction(predictions.length > 0 ? predictions[0] : null);
            } catch {
                setGridPrediction(null);
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
            await http.post(`/api/samples/${currentSample.id}/label`, {
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

    // Corner mode toggle
    const toggleCornerMode = useCallback(() => {
        if (isCornerMode) {
            // Reset if already in corner mode
            setIsCornerMode(false);
            setPendingCorners({});
            setCornerCount(0);
            setHideBox(false); // Show bounding box again when exiting corner mode
        } else {
            // Only allow corner mode if entire board is visible
            if (!currentDetection?.isEntireBoardVisible) {
                toast.error('Corners can only be labeled when entire board is visible');
                return;
            }
            // Enter corner mode, exit draw mode, hide bounding box
            setIsCornerMode(true);
            setIsDrawingBox(false);
            setHideBox(true); // Hide bounding box for easier corner labeling
            setPendingCorners({});
            setCornerCount(0);
        }
    }, [isCornerMode, currentDetection?.isEntireBoardVisible]);

    // Grid mode toggle - saves when exiting if modified
    const toggleGridMode = useCallback(async () => {
        if (isGridMode) {
            // Exit grid mode - save if modified
            if (isGridModified && currentSample && gridLabels) {
                try {
                    await ML_API.saveGridLabel(currentSample.id, gridLabels);
                    toast.success('Grid labels saved!');
                    setIsGridModified(false);
                } catch {
                    toast.error('Failed to save grid labels');
                }
            }
            setIsGridMode(false);
        } else {
            // Check if we have a cropped image to show
            const croppedUrl = cornerDetection?.best.croppedUrl || cornerDetection?.hough.croppedUrl || cornerDetection?.edge.croppedUrl;
            if (!croppedUrl) {
                toast.error('Run corner detection first to get cropped board image');
                return;
            }
            // Enter grid mode
            setIsGridMode(true);
            setIsDrawingBox(false);
            setIsCornerMode(false);
            // Initialize grid if not loaded
            if (!gridLabels) {
                setGridLabels(createEmptyGrid());
            }
        }
    }, [isGridMode, cornerDetection, gridLabels, isGridModified, currentSample]);

    // Handle grid cell click - toggle: if cell has selected color, clear it; otherwise set selected color
    const handleGridCellClick = useCallback((row: number, col: number) => {
        if (!gridLabels) {
            return;
        }
        const currentCell = gridLabels[row][col];
        // If clicking a cell that already has the selected color, clear it (set to null)
        // Otherwise, set it to the selected color
        const newColor = currentCell === selectedColor ? null : selectedColor;
        const newGrid = gridLabels.map((r, ri) =>
            ri === row ? r.map((c, ci) => ci === col ? newColor : c) : r
        );
        setGridLabels(newGrid);
        setIsGridModified(true);
        // Auto-enable grid mode when clicking
        if (!isGridMode) {
            setIsGridMode(true);
        }
    }, [gridLabels, selectedColor, isGridMode]);

    // Save grid labels
    const saveGridLabels = useCallback(async () => {
        if (!currentSample || !gridLabels) {
            return;
        }
        try {
            await ML_API.saveGridLabel(currentSample.id, gridLabels);
            toast.success('Grid labels saved!');
            setIsGridModified(false);
        } catch {
            toast.error('Failed to save grid labels');
        }
    }, [currentSample, gridLabels]);

    // Handle corner click
    const handleCornerClick = useCallback((pos: { x: number; y: number }) => {
        if (!isCornerMode || !currentSample) {
            return;
        }

        const cornerKey = CORNER_CONFIG[cornerCount].key;
        const newCorners = { ...pendingCorners, [cornerKey]: pos };
        setPendingCorners(newCorners);

        const newCount = cornerCount + 1;
        setCornerCount(newCount);

        // Auto-save after 4th corner
        if (newCount === 4) {
            const fullCorners: BoardCorners = {
                topLeft: newCorners.topLeft!,
                topRight: newCorners.topRight!,
                bottomRight: newCorners.bottomRight!,
                bottomLeft: newCorners.bottomLeft!,
            };
            ML_API.saveCorners(currentSample.id, fullCorners).then(() => {
                setSavedCorners(fullCorners);
                setIsCornerMode(false);
                setPendingCorners({});
                setCornerCount(0);
                toast.success('Corners saved!');

                // Auto-navigate to next if bbox exists
                const hasBbox = labelData.correction?.boundingBox || labelData.prediction?.boundingBox;
                if (hasBbox && currentIndex < samples.length - 1) {
                    const newIndex = currentIndex + 1;
                    setCurrentIndex(newIndex);
                    navigate(`/admin/images/label/${samples[newIndex].id}`, { replace: true });
                }
            });
        }
    }, [isCornerMode, cornerCount, pendingCorners, currentSample, labelData, currentIndex, samples, navigate]);

    // Get relative position helper for pointer events
    const getRelativePosition = useCallback((clientX: number, clientY: number, containerRef: React.RefObject<HTMLDivElement>) => {
        if (!containerRef.current) {
            return null;
        }
        const rect = containerRef.current.getBoundingClientRect();
        return {
            x: Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
            y: Math.max(0, Math.min(1, (clientY - rect.top) / rect.height)),
        };
    }, []);

    const handlePointerDown = useCallback((e: React.PointerEvent, containerRef: React.RefObject<HTMLDivElement>) => {
        const pos = getRelativePosition(e.clientX, e.clientY, containerRef);
        if (!pos) {
            return;
        }

        // Handle corner mode clicks
        if (isCornerMode) {
            e.preventDefault();
            handleCornerClick(pos);
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
    }, [isCornerMode, isDrawingBox, boundingBox, handleCornerClick, getRelativePosition]);

    const handlePointerMove = useCallback((e: React.PointerEvent, containerRef: React.RefObject<HTMLDivElement>) => {
        const pos = getRelativePosition(e.clientX, e.clientY, containerRef);
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
    }, [isDrawingBox, drawStart, dragHandle, dragStartBox, dragStartPos, updateBoundingBox, getRelativePosition]);

    const handlePointerUp = useCallback((e: React.PointerEvent) => {
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
    }, [isDrawingBox, drawStart, dragHandle]);

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
            } else if (key === 'c') {
                e.preventDefault();
                toggleCornerMode();
            } else if (key === 'q') {
                e.preventDefault();
                // Clear corners if they exist
                if (currentSample && savedCorners) {
                    ML_API.clearCorners(currentSample.id).then(() => {
                        setSavedCorners(null);
                        toast.success('Corners cleared');
                    }).catch(() => {
                        toast.error('Failed to clear corners');
                    });
                }
            } else if (key === 'z') {
                e.preventDefault();
                toggleGridMode();
            } else if (key === 'b') {
                e.preventDefault();
                setSelectedColor('blue');
            } else if (key === 'y') {
                e.preventDefault();
                setSelectedColor('yellow');
            } else if (key === 'g') {
                e.preventDefault();
                setSelectedColor('green');
            } else if (key === 'r') {
                e.preventDefault();
                setSelectedColor('red');
            } else if (key === 'w') {
                e.preventDefault();
                setSelectedColor(null); // White/empty
            } else if (key === 'a') {
                e.preventDefault();
                // Accept AI prediction and save as human labels
                if (gridPrediction && currentSample) {
                    setGridLabels(gridPrediction.grid);
                    ML_API.saveGridLabel(currentSample.id, gridPrediction.grid).then(() => {
                        toast.success('AI labels accepted and saved!');
                        setIsGridModified(false);
                    }).catch(() => {
                        toast.error('Failed to save AI labels');
                    });
                }
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
    }, [currentIndex, samples.length, labelData.correction, currentSample, currentDetection, updateCorrection, goToPrevious, goToNext, handleSaveAndNext, handleDeleteLabel, isDrawingBox, boundingBox, updateBoundingBox, saveAndNext, toggleCornerMode, savedCorners, toggleGridMode, gridPrediction]);

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
                    <button
                        onClick={() => navigate(`/admin/images/${currentSample.id}`)}
                        className="p-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                        title="View image details"
                    >
                        <ExternalLink className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Main content with navigation arrows outside, centered */}
            <div className="flex items-center justify-center gap-4">
                {/* Left navigation arrow */}
                <button
                    onClick={goToPrevious}
                    disabled={currentIndex === 0}
                    className="p-2 rounded-lg bg-surface-dim text-foreground enabled:hover:bg-primary enabled:hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
                >
                    <ChevronLeft className="w-5 h-5" />
                </button>

                {/* Content area */}
                <div className="space-y-4 w-full max-w-6xl">
                    {/* Row 1: Original image (1fr) + Grid panel with 2 grids (2fr) */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        {/* Original image - takes 1 column */}
                        <div className="space-y-2">
                            <h3 className="text-sm font-medium text-secondary">Original Image</h3>
                            <OriginalImagePanel
                                currentSample={currentSample}
                                labelData={labelData}
                                boundingBox={boundingBox || null}
                                hideBox={hideBox}
                                isDrawingBox={isDrawingBox}
                                drawStart={drawStart}
                                dragHandle={dragHandle}
                                isCornerMode={isCornerMode}
                                cornerCount={cornerCount}
                                pendingCorners={pendingCorners}
                                savedCorners={savedCorners}
                                onPointerDown={handlePointerDown}
                                onPointerMove={handlePointerMove}
                                onPointerUp={handlePointerUp}
                            />
                        </div>

                        {/* Grid labeling panel - takes 2 columns for the 2 grids */}
                        <div className="lg:col-span-2">
                            <GridLabelingPanel
                                sampleId={currentSample.id}
                                cornerDetection={cornerDetection}
                                gridLabels={gridLabels}
                                selectedColor={selectedColor}
                                isGridMode={isGridMode}
                                isGridModified={isGridModified}
                                initialPrediction={gridPrediction}
                                isEntireBoardVisible={currentDetection?.isEntireBoardVisible ?? false}
                                onCellClick={handleGridCellClick}
                                onColorSelect={setSelectedColor}
                                onSave={saveGridLabels}
                                setGridLabels={setGridLabels}
                            />
                        </div>
                    </div>

                    {/* Row 2: Controls, Tags, and Pieces */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        <div>
                            <LabelingControls
                                currentSample={currentSample}
                                labelData={labelData}
                                currentDetection={currentDetection || null}
                                boundingBox={boundingBox || null}
                                isDrawingBox={isDrawingBox}
                                isCornerMode={isCornerMode}
                                cornerCount={cornerCount}
                                onDrawModeToggle={() => setIsDrawingBox(!isDrawingBox)}
                                onDeleteLabel={handleDeleteLabel}
                                onSaveAndNext={handleSaveAndNext}
                                onToggleBoardDetected={toggleBoardDetected}
                                onToggleEntireBoardVisible={toggleEntireBoardVisible}
                                updateCorrection={updateCorrection}
                                updateBoundingBox={updateBoundingBox}
                            />
                        </div>
                        <div>
                            <SampleTagsSection sampleId={currentSample.id} allTags={allTags} />
                        </div>
                        <div>
                            <PiecesPanel
                                gridLabels={gridLabels}
                                aiPrediction={gridPrediction?.grid ?? null}
                            />
                        </div>
                    </div>
                </div>

                {/* Right navigation arrow */}
                <button
                    onClick={goToNext}
                    disabled={currentIndex === samples.length - 1}
                    className="p-2 rounded-lg bg-surface-dim text-foreground enabled:hover:bg-primary enabled:hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
                >
                    <ChevronRight className="w-5 h-5" />
                </button>
            </div>
        </div>
    );
};
