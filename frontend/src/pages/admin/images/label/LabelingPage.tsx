import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { ChevronLeft, ChevronRight, Image } from 'lucide-react';
import { http, API_URL } from '../../../../api/config';
import { ML_API, BoardCorners } from '../../../../api/ml';
import { Button } from '../../../../components/ui/Button';
import type { BoundingBox, BoardDetection, SampleStatusFlags, CornerDetectionResult, GridLabels, PlayerColor } from '@blokus-ai/shared';
import { getStatusLabels, PLAYER_COLORS } from '@blokus-ai/shared';
import { statusColors, statusLabels, isUnlabeled } from '../../../../lib/status';

// Grid cell colors for labeling
const CELL_COLORS: { key: PlayerColor | null; label: string; color: string }[] = [
    { key: null, label: 'Empty', color: '#e5e7eb' },      // gray-200
    { key: 'blue', label: 'Blue', color: PLAYER_COLORS.blue },
    { key: 'yellow', label: 'Yellow', color: PLAYER_COLORS.yellow },
    { key: 'red', label: 'Red', color: PLAYER_COLORS.red },
    { key: 'green', label: 'Green', color: PLAYER_COLORS.green },
];

// Create empty 20x20 grid
const createEmptyGrid = (): GridLabels =>
    Array.from({ length: 20 }, () => Array.from({ length: 20 }, () => null));

// Corner labels and colors
const CORNER_CONFIG = [
    { key: 'topLeft' as const, label: 'TL', color: '#ef4444' },     // red
    { key: 'topRight' as const, label: 'TR', color: '#22c55e' },    // green
    { key: 'bottomRight' as const, label: 'BR', color: '#3b82f6' }, // blue
    { key: 'bottomLeft' as const, label: 'BL', color: '#eab308' },  // yellow
];

type CornerKey = 'topLeft' | 'topRight' | 'bottomRight' | 'bottomLeft';
type PartialCorners = Partial<Record<CornerKey, { x: number; y: number }>>;

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
    const gridContainerRef = useRef<HTMLDivElement>(null);

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
            // Reset corner mode
            setIsCornerMode(false);
            setPendingCorners({});
            setCornerCount(0);
            // Reset grid mode
            setIsGridMode(false);
            setGridLabels(null);
            setIsGridModified(false);

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
    }, [currentIndex, samples.length, labelData.correction, currentSample, currentDetection, updateCorrection, goToPrevious, goToNext, handleSaveAndNext, handleDeleteLabel, isDrawingBox, boundingBox, updateBoundingBox, saveAndNext, toggleCornerMode, savedCorners, toggleGridMode]);

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
        if (isCornerMode) {
            return 'crosshair';
        }
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
            <div className="space-y-4 max-w-5xl mx-auto">
                {/* Row 1: Images side by side */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Original image with navigation */}
                    <div className="space-y-2">
                        <h3 className="text-sm font-medium text-secondary">Original Image</h3>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={goToPrevious}
                                disabled={currentIndex === 0}
                                className="p-1.5 rounded-lg bg-surface-dim text-foreground enabled:hover:bg-primary enabled:hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                            >
                                <ChevronLeft className="w-4 h-4" />
                            </button>

                            <div
                                ref={imageContainerRef}
                                className="flex-1 relative bg-surface rounded-lg border border-border overflow-hidden max-w-md"
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

                        {/* Saved corners overlay */}
                        {savedCorners && !isCornerMode && (
                            <>
                                {/* Dashed lines connecting corners */}
                                <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ overflow: 'visible' }}>
                                    <polygon
                                        points={`${savedCorners.topLeft.x * 100}%,${savedCorners.topLeft.y * 100}% ${savedCorners.topRight.x * 100}%,${savedCorners.topRight.y * 100}% ${savedCorners.bottomRight.x * 100}%,${savedCorners.bottomRight.y * 100}% ${savedCorners.bottomLeft.x * 100}%,${savedCorners.bottomLeft.y * 100}%`}
                                        fill="none"
                                        stroke="cyan"
                                        strokeWidth="2"
                                        strokeDasharray="6,4"
                                    />
                                </svg>
                                {/* Corner dots */}
                                {CORNER_CONFIG.map(({ key, label, color }) => {
                                    const corner = savedCorners[key];
                                    return (
                                        <div
                                            key={key}
                                            className="absolute pointer-events-none"
                                            style={{
                                                left: `${corner.x * 100}%`,
                                                top: `${corner.y * 100}%`,
                                                transform: 'translate(-50%, -50%)',
                                            }}
                                        >
                                            <div
                                                className="w-4 h-4 rounded-full border-2 border-white"
                                                style={{ backgroundColor: color }}
                                            />
                                            <div
                                                className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold px-1 rounded"
                                                style={{ backgroundColor: color, color: 'white' }}
                                            >
                                                {label}
                                            </div>
                                        </div>
                                    );
                                })}
                            </>
                        )}

                        {/* Pending corners overlay (while labeling) */}
                        {isCornerMode && Object.keys(pendingCorners).length > 0 && (
                            <>
                                {/* Partial lines connecting placed corners */}
                                {cornerCount >= 2 && (
                                    <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ overflow: 'visible' }}>
                                        {cornerCount >= 2 && pendingCorners.topLeft && pendingCorners.topRight && (
                                            <line
                                                x1={`${pendingCorners.topLeft.x * 100}%`}
                                                y1={`${pendingCorners.topLeft.y * 100}%`}
                                                x2={`${pendingCorners.topRight.x * 100}%`}
                                                y2={`${pendingCorners.topRight.y * 100}%`}
                                                stroke="cyan"
                                                strokeWidth="2"
                                                strokeDasharray="6,4"
                                            />
                                        )}
                                        {cornerCount >= 3 && pendingCorners.topRight && pendingCorners.bottomRight && (
                                            <line
                                                x1={`${pendingCorners.topRight.x * 100}%`}
                                                y1={`${pendingCorners.topRight.y * 100}%`}
                                                x2={`${pendingCorners.bottomRight.x * 100}%`}
                                                y2={`${pendingCorners.bottomRight.y * 100}%`}
                                                stroke="cyan"
                                                strokeWidth="2"
                                                strokeDasharray="6,4"
                                            />
                                        )}
                                        {cornerCount >= 4 && pendingCorners.bottomRight && pendingCorners.bottomLeft && (
                                            <line
                                                x1={`${pendingCorners.bottomRight.x * 100}%`}
                                                y1={`${pendingCorners.bottomRight.y * 100}%`}
                                                x2={`${pendingCorners.bottomLeft.x * 100}%`}
                                                y2={`${pendingCorners.bottomLeft.y * 100}%`}
                                                stroke="cyan"
                                                strokeWidth="2"
                                                strokeDasharray="6,4"
                                            />
                                        )}
                                        {cornerCount >= 4 && pendingCorners.bottomLeft && pendingCorners.topLeft && (
                                            <line
                                                x1={`${pendingCorners.bottomLeft.x * 100}%`}
                                                y1={`${pendingCorners.bottomLeft.y * 100}%`}
                                                x2={`${pendingCorners.topLeft.x * 100}%`}
                                                y2={`${pendingCorners.topLeft.y * 100}%`}
                                                stroke="cyan"
                                                strokeWidth="2"
                                                strokeDasharray="6,4"
                                            />
                                        )}
                                    </svg>
                                )}
                                {/* Pending corner dots */}
                                {CORNER_CONFIG.slice(0, cornerCount).map(({ key, label, color }) => {
                                    const corner = pendingCorners[key];
                                    if (!corner) {
                                        return null;
                                    }
                                    return (
                                        <div
                                            key={key}
                                            className="absolute pointer-events-none"
                                            style={{
                                                left: `${corner.x * 100}%`,
                                                top: `${corner.y * 100}%`,
                                                transform: 'translate(-50%, -50%)',
                                            }}
                                        >
                                            <div
                                                className="w-4 h-4 rounded-full border-2 border-white"
                                                style={{ backgroundColor: color }}
                                            />
                                            <div
                                                className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold px-1 rounded"
                                                style={{ backgroundColor: color, color: 'white' }}
                                            >
                                                {label}
                                            </div>
                                        </div>
                                    );
                                })}
                            </>
                        )}

                        {/* Corner mode overlay */}
                        {isCornerMode && (
                            <div className="absolute inset-0 bg-warning/10 pointer-events-none" />
                        )}
                            </div>

                            <button
                                onClick={goToNext}
                                disabled={currentIndex === samples.length - 1}
                                className="p-1.5 rounded-lg bg-surface-dim text-foreground enabled:hover:bg-primary enabled:hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                            >
                                <ChevronRight className="w-4 h-4" />
                            </button>
                        </div>
                        {/* Mode hints under image */}
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
                    </div>

                    {/* Grid labeling image - always shown */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <h3 className={`text-sm font-medium ${isGridMode ? 'text-success' : 'text-secondary'}`}>
                                Grid Labeling {isGridMode && <span className="ml-1 px-1.5 py-0.5 bg-success/20 rounded text-[10px]">ACTIVE</span>}
                            </h3>
                            {/* Color palette */}
                            <div className="flex gap-1">
                                {CELL_COLORS.map(({ key, label, color }) => (
                                    <button
                                        key={label}
                                        onClick={() => setSelectedColor(key)}
                                        className={`w-6 h-6 rounded border-2 transition-all ${
                                            selectedColor === key ? 'border-foreground scale-110 ring-2 ring-foreground/30' : 'border-border'
                                        }`}
                                        style={{ backgroundColor: color }}
                                        title={label}
                                    />
                                ))}
                            </div>
                        </div>
                        {(() => {
                            const croppedUrl = cornerDetection?.best.croppedUrl || cornerDetection?.hough.croppedUrl || cornerDetection?.edge.croppedUrl;
                            if (!croppedUrl) {
                                return (
                                    <div className="aspect-square max-w-md bg-surface-dim rounded-lg border border-border flex items-center justify-center">
                                        <p className="text-sm text-muted text-center px-4">Run corner detection to enable grid labeling</p>
                                    </div>
                                );
                            }
                            // Initialize grid if needed
                            const grid = gridLabels || createEmptyGrid();
                            return (
                                <div
                                    ref={gridContainerRef}
                                    className={`relative aspect-square max-w-md bg-surface rounded-lg border-2 overflow-hidden ${isGridMode ? 'border-success' : 'border-border'}`}
                                >
                                    <img
                                        src={croppedUrl}
                                        alt="Cropped board"
                                        className="absolute inset-0 w-full h-full object-cover"
                                        draggable={false}
                                    />
                                    {/* 20x20 grid overlay */}
                                    <div className="absolute inset-0 grid grid-cols-20 grid-rows-20">
                                        {grid.map((row, ri) =>
                                            row.map((cell, ci) => (
                                                <button
                                                    key={`${ri}-${ci}`}
                                                    onClick={() => {
                                                        if (!gridLabels) {
                                                            setGridLabels(createEmptyGrid());
                                                        }
                                                        handleGridCellClick(ri, ci);
                                                    }}
                                                    className="border border-white/20 hover:border-white/60 transition-colors"
                                                    style={{
                                                        backgroundColor: cell ? `${PLAYER_COLORS[cell]}99` : 'transparent',
                                                    }}
                                                />
                                            ))
                                        )}
                                    </div>
                                </div>
                            );
                        })()}
                        {isGridModified && (
                            <Button
                                variant="success"
                                size="sm"
                                className="w-full max-w-md"
                                onClick={saveGridLabels}
                            >
                                Save Grid Labels
                            </Button>
                        )}
                    </div>
                </div>

                {/* Row 2: Controls */}
                <div className="space-y-4">
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
            </div>
        </div>
    );
};
