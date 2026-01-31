import { useRef, useState, useEffect, useMemo } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { Plus, ChevronLeft, ChevronRight, Tag, Lightbulb, Trash2, Search, Image, LayoutGrid, List, X, Focus, Grid3X3 } from 'lucide-react';
import { cn } from '../../../utils/cn';
import { formatShortId } from '../../../utils/format';
import { http, API_URL } from '../../../api/config';
import { ML_API } from '../../../api/ml';
import { Button } from '../../../components/ui/Button';
import { Input } from '../../../components/ui/Input';
import { Select } from '../../../components/ui/Select';
import { MultiSelect } from '../../../components/ui/MultiSelect';
import { Table, Column, SortState } from '../../../components/ui/Table';
import { getStatusLabels, Sample } from '@blokus-ai/shared';
import { statusColors, statusLabels } from '../../../lib/status';
import { TrainedModelsTable } from './TrainedModelsTable';
import { DetectionRunsTable } from './DetectionRunsTable';
import { RunDetectionModal } from './RunDetectionModal';
import { TestModelTab } from './TestModelTab';
import { PipelinesSection } from './PipelinesSection';
import { CreatePipelineModal } from './CreatePipelineModal';
import { TagsSection } from './TagsSection';
import { CreateTagModal } from './CreateTagModal';
import { ConfirmModal } from '../../../components/ui/ConfirmModal';
import { useAdmin } from '../../../contexts/AdminContext';
import { useSocketEvent } from '../../../hooks/useSocketEvent';
import type { ModelPipeline, CornerDetectionResult, GridPredictionRecord, GridLabels, GridLabel, CellColor } from '@blokus-ai/shared';

type Tab = 'images' | 'models' | 'detections' | 'test' | 'config';
type FilterPreset = 'all' | 'selected' | 'unlabeled' | 'false-positives' | 'false-negatives' | 'low-iou' | 'disagreements';
type SortOption = 'default' | 'lowest-iou' | 'lowest-confidence';
type ViewType = 'grid' | 'list';
type MultiModelMode = 'intersection' | 'union';

const filterPresetLabels: Record<FilterPreset, string> = {
    'all': 'All',
    'selected': 'Selected',
    'unlabeled': 'Unlabeled',
    'false-positives': 'False Positives',
    'false-negatives': 'False Negatives',
    'low-iou': 'Low IoU (<80%)',
    'disagreements': 'Disagreements',
};

const sortLabels: Record<SortOption, string> = {
    'default': 'Default',
    'lowest-iou': 'Lowest IoU',
    'lowest-confidence': 'Lowest Confidence',
};

interface GridMetrics {
    accuracy: number;  // Overall accuracy (correct/400)
    minF1: number;     // Minimum F1 score across colors
    minRecall: number; // Minimum recall across colors
}

/**
 * Calculate grid metrics by comparing AI prediction vs human label.
 * Returns null if either is missing.
 */
const calculateGridMetrics = (
    aiGrid: GridLabels | null | undefined,
    humanGrid: GridLabels | null | undefined
): GridMetrics | null => {
    if (!aiGrid || !humanGrid) {
        return null;
    }

    // Colors to track (null = empty)
    const colors: (CellColor)[] = ['blue', 'yellow', 'red', 'green', null];

    // Count TP, FP, FN for each color
    const stats: Record<string, { tp: number; fp: number; fn: number }> = {};
    for (const color of colors) {
        const key = color ?? 'empty';
        stats[key] = { tp: 0, fp: 0, fn: 0 };
    }

    let correct = 0;
    const total = 400;

    for (let row = 0; row < 20; row++) {
        for (let col = 0; col < 20; col++) {
            const ai = aiGrid[row]?.[col] ?? null;
            const human = humanGrid[row]?.[col] ?? null;

            if (ai === human) {
                correct++;
                const key = ai ?? 'empty';
                stats[key].tp++;
            } else {
                // AI predicted `ai` but should have been `human`
                const aiKey = ai ?? 'empty';
                const humanKey = human ?? 'empty';
                stats[aiKey].fp++;
                stats[humanKey].fn++;
            }
        }
    }

    const accuracy = correct / total;

    // Calculate F1 and recall for each color
    const f1Scores: number[] = [];
    const recalls: number[] = [];

    for (const color of colors) {
        const key = color ?? 'empty';
        const { tp, fp, fn } = stats[key];

        const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
        const recall = tp + fn > 0 ? tp / (tp + fn) : 1; // If no positives, recall is perfect
        const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

        f1Scores.push(f1);
        recalls.push(recall);
    }

    return {
        accuracy,
        minF1: Math.min(...f1Scores),
        minRecall: Math.min(...recalls),
    };
};

interface PresignResponse {
    id: string;
    uploadUrl: string;
    key: string;
    imageUrl: string;
}


const matchesPreset = (sample: Sample, preset: FilterPreset, selectedIds?: Set<string>): boolean => {
    switch (preset) {
    case 'all':
        return true;
    case 'selected':
        return selectedIds?.has(sample.id) ?? false;
    case 'unlabeled':
        return !sample.status.humanLabeled;
    case 'false-positives':
        // AI says board, human says no board
        return sample.ai?.hasBoardDetected === true && sample.human?.hasBoardDetected === false;
    case 'false-negatives':
        // AI says no board, human says board
        return sample.ai?.hasBoardDetected === false && sample.human?.hasBoardDetected === true;
    case 'low-iou':
        // Both have bboxes but IoU < 80%
        return sample.iou !== null && sample.iou < 0.8;
    case 'disagreements':
        // AI and human disagree on board detection
        return sample.ai !== null && sample.human !== null &&
                sample.ai.hasBoardDetected !== sample.human.hasBoardDetected;
    default:
        return true;
    }
};

const sortSamples = (samples: Sample[], sort: SortOption): Sample[] => {
    const sorted = [...samples];
    switch (sort) {
    case 'lowest-iou':
        return sorted.sort((a, b) => {
            // Nulls go to the end
            if (a.iou === null && b.iou === null) {
                return 0;
            }
            if (a.iou === null) {
                return 1;
            }
            if (b.iou === null) {
                return -1;
            }
            return a.iou - b.iou;
        });
    case 'lowest-confidence':
        return sorted.sort((a, b) => {
            const aConf = a.ai?.confidence ?? null;
            const bConf = b.ai?.confidence ?? null;
            // Nulls and 0% go to the end
            const aIsEmpty = aConf === null || aConf === 0;
            const bIsEmpty = bConf === null || bConf === 0;
            if (aIsEmpty && bIsEmpty) {
                return 0;
            }
            if (aIsEmpty) {
                return 1;
            }
            if (bIsEmpty) {
                return -1;
            }
            return aConf - bConf;
        });
    case 'default':
    default:
        return sorted;
    }
};

export const ImagesPage = () => {
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Get data from context
    const {
        samples,
        isLoadingSamples: isLoading,
        fetchSamples,
        removeSamples,
        selectedModelIds,
        setSelectedModelIds,
        trainingRuns,
        isLoadingTrainingRuns: isLoadingRuns,
        trainingProgress,
        removeTrainingRun,
        detectionRuns,
        isLoadingDetectionRuns: isLoadingDetections,
        detectionProgress,
        removeDetectionRun,
        pipelines,
        isLoadingPipelines,
        addPipeline,
        updatePipelineState,
        removePipeline,
        tags,
        isLoadingTags,
        addTag,
        removeTag,
        cornerProgress,
        gridProgress,
    } = useAdmin();

    // Local UI state
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState({ completed: 0, total: 0 });
    const [isTraining, setIsTraining] = useState(false);
    const [isRunningDetection, setIsRunningDetection] = useState(false);
    const [showDetectionModal, setShowDetectionModal] = useState(false);
    const [showPipelineModal, setShowPipelineModal] = useState(false);
    const [editingPipeline, setEditingPipeline] = useState<ModelPipeline | null>(null);
    const [showTagModal, setShowTagModal] = useState(false);
    const [isRunningCorners, setIsRunningCorners] = useState(false);
    const [activeCornerRunId, setActiveCornerRunId] = useState<string | null>(null);
    const [cornerDetections, setCornerDetections] = useState<Map<string, CornerDetectionResult>>(new Map());
    const [isRunningGridPrediction, setIsRunningGridPrediction] = useState(false);
    const [gridPredictions, setGridPredictions] = useState<Map<string, GridPredictionRecord>>(new Map());
    const [gridLabels, setGridLabels] = useState<Map<string, GridLabel>>(new Map());

    // Overlay visibility toggles
    const [overlayVisibility, setOverlayVisibility] = useState({
        humanBox: true,
        aiBox: true,
        humanCorners: false,
        aiCorners: false,
        bestCorners: false,
        houghCorners: false,
        edgeCorners: false,
    });

    // Selection state
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [lastClickedIndex, setLastClickedIndex] = useState<number | null>(null);
    const [isDeleting, setIsDeleting] = useState(false);
    const [showDeleteModal, setShowDeleteModal] = useState(false);

    // Pagination state
    const [currentPage, setCurrentPage] = useState(1);
    const pageSize = 50;

    // Get tab from URL or default to 'images'
    const urlTab = searchParams.get('tab') as Tab | null;
    const tab: Tab = urlTab === 'models' ? 'models' : urlTab === 'detections' ? 'detections' : urlTab === 'test' ? 'test' : urlTab === 'config' ? 'config' : 'images';

    const setTab = (newTab: Tab) => {
        if (newTab === 'images') {
            searchParams.delete('tab');
        } else {
            searchParams.set('tab', newTab);
        }
        // Clear filters when switching tabs
        searchParams.delete('preset');
        searchParams.delete('sort');
        setSearchParams(searchParams);
    };

    // Get preset filter from URL or default to 'all'
    const urlPreset = searchParams.get('preset') as FilterPreset | null;
    const validPresets: FilterPreset[] = ['all', 'selected', 'unlabeled', 'false-positives', 'false-negatives', 'low-iou', 'disagreements'];
    const preset: FilterPreset = urlPreset && validPresets.includes(urlPreset) ? urlPreset : 'all';

    const setPreset = (newPreset: FilterPreset) => {
        if (newPreset === 'all') {
            searchParams.delete('preset');
        } else {
            searchParams.set('preset', newPreset);
        }
        setSearchParams(searchParams);
    };

    // Get sort from URL or default to 'default'
    const urlSort = searchParams.get('sort') as SortOption | null;
    const validSorts: SortOption[] = ['default', 'lowest-iou', 'lowest-confidence'];
    const sort: SortOption = urlSort && validSorts.includes(urlSort) ? urlSort : 'default';

    const setSort = (newSort: SortOption) => {
        if (newSort === 'default') {
            searchParams.delete('sort');
        } else {
            searchParams.set('sort', newSort);
        }
        setSearchParams(searchParams);
    };

    // Get view type from URL or default to 'grid'
    const urlView = searchParams.get('view') as ViewType | null;
    const viewType: ViewType = urlView === 'list' ? 'list' : 'grid';

    const setViewType = (newView: ViewType) => {
        if (newView === 'grid') {
            searchParams.delete('view');
        } else {
            searchParams.set('view', newView);
        }
        setSearchParams(searchParams);
    };

    // Search state
    const [searchQuery, setSearchQuery] = useState('');

    // Tag filter from URL
    const urlTagId = searchParams.get('tagId');
    const selectedTagId = urlTagId || '';

    const setSelectedTagId = (tagId: string) => {
        if (!tagId) {
            searchParams.delete('tagId');
        } else {
            searchParams.set('tagId', tagId);
        }
        setSearchParams(searchParams);
    };

    // Multi-model mode (intersection vs union)
    const [multiModelMode, setMultiModelMode] = useState<MultiModelMode>('intersection');

    // Multi-model samples cache
    const [multiModelSamples, setMultiModelSamples] = useState<Map<string, Set<string>>>(new Map());
    const [isLoadingMultiModel, setIsLoadingMultiModel] = useState(false);

    // List view sorting state
    const [listSortState, setListSortState] = useState<SortState>({ column: 'created_at', direction: 'desc' });

    const handleListSort = (columnKey: string) => {
        setListSortState(prev => {
            if (prev.column !== columnKey) {
                // New column: start with desc
                return { column: columnKey, direction: 'desc' };
            }
            // Same column: cycle desc → asc → null
            if (prev.direction === 'desc') {
                return { column: columnKey, direction: 'asc' };
            }
            if (prev.direction === 'asc') {
                return { column: null, direction: null };
            }
            // null → desc
            return { column: columnKey, direction: 'desc' };
        });
    };

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) {
            return;
        }

        const fileArray = Array.from(files);
        setIsUploading(true);
        setUploadProgress({ completed: 0, total: fileArray.length });

        let completedCount = 0;

        const uploadFile = async (file: File): Promise<boolean> => {
            try {
                // Get presigned URL from API
                const presign = await http.post<PresignResponse>('/api/samples/upload/presign', {});

                // Upload directly to S3
                const response = await fetch(presign.uploadUrl, {
                    method: 'PUT',
                    body: file,
                    headers: {
                        'Content-Type': file.type || 'image/png',
                    },
                });

                if (!response.ok) {
                    throw new Error('Upload failed');
                }

                // Generate thumbnail
                await http.post('/api/samples/upload/complete', { id: presign.id });

                completedCount++;
                setUploadProgress(prev => ({ ...prev, completed: completedCount }));
                return true;
            } catch {
                completedCount++;
                setUploadProgress(prev => ({ ...prev, completed: completedCount }));
                return false;
            }
        };

        // Process uploads with concurrency limit of 5
        const concurrencyLimit = 5;
        const results: boolean[] = [];

        for (let i = 0; i < fileArray.length; i += concurrencyLimit) {
            const batch = fileArray.slice(i, i + concurrencyLimit);
            const batchResults = await Promise.all(batch.map(uploadFile));
            results.push(...batchResults);
        }

        const successCount = results.filter(Boolean).length;
        const failCount = results.length - successCount;

        if (successCount > 0) {
            toast.success(`Uploaded ${successCount} image${successCount > 1 ? 's' : ''}`);
            fetchSamples(true);
        }
        if (failCount > 0) {
            toast.error(`Failed to upload ${failCount} image${failCount > 1 ? 's' : ''}`);
        }

        setIsUploading(false);
        setUploadProgress({ completed: 0, total: 0 });
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleTrain = async () => {
        if (selectedIds.size === 0) {
            return;
        }
        setIsTraining(true);
        try {
            // Don't need to addTrainingRun here - WebSocket 'training-run-started' event handles it
            await ML_API.startTraining(Array.from(selectedIds));
            toast.success('Training started');
        } catch {
            toast.error('Failed to start training');
            setIsTraining(false);
        }
    };

    const handleStartDetection = async (modelId: string, includeExistingAiLabeled: boolean) => {
        setIsRunningDetection(true);
        try {
            await ML_API.startBulkDetection(modelId, includeExistingAiLabeled);
            // Don't add run here - socket event 'detection-run-started' will handle it
            setShowDetectionModal(false);
            toast.success('Detection started');
        } catch {
            toast.error('Failed to start detection');
        } finally {
            setIsRunningDetection(false);
        }
    };

    // Pipeline handlers
    const handleCreatePipeline = () => {
        setEditingPipeline(null);
        setShowPipelineModal(true);
    };

    const handleEditPipeline = (pipeline: ModelPipeline) => {
        setEditingPipeline(pipeline);
        setShowPipelineModal(true);
    };

    const handleSavePipeline = async (name: string, componentIds: string[]) => {
        if (editingPipeline) {
            const updated = await ML_API.updatePipeline(editingPipeline.id, { name, componentIds });
            updatePipelineState(editingPipeline.id, updated);
            toast.success('Pipeline updated');
        } else {
            const pipeline = await ML_API.createPipeline(name, componentIds);
            addPipeline(pipeline);
            toast.success('Pipeline created');
        }
    };

    const handleDeletePipeline = async (id: string) => {
        try {
            await ML_API.deletePipeline(id);
            removePipeline(id);
            toast.success('Pipeline deleted');
        } catch {
            toast.error('Failed to delete pipeline');
        }
    };

    // Tag handlers
    const handleCreateTag = async (name: string) => {
        const tag = await ML_API.createTag(name);
        addTag(tag);
        toast.success('Tag created');
    };

    const handleDeleteTag = async (id: string) => {
        try {
            await ML_API.deleteTag(id);
            removeTag(id);
            toast.success('Tag deleted');
        } catch {
            toast.error('Failed to delete tag');
        }
    };

    // Corner detection handler
    const handleRunCorners = async () => {
        if (selectedIds.size === 0) {
            return;
        }
        setIsRunningCorners(true);
        try {
            const run = await ML_API.startCornerDetection(Array.from(selectedIds));
            setActiveCornerRunId(run.id);
            toast.success('Corner detection started');
        } catch {
            toast.error('Failed to start corner detection');
            setIsRunningCorners(false);
        }
    };

    // Grid prediction handler
    const handleRunGridPrediction = async () => {
        setIsRunningGridPrediction(true);
        try {
            const result = await ML_API.startGridRun(selectedIds.size > 0 ? Array.from(selectedIds) : undefined);
            toast.success(`Grid prediction started for ${result.started} samples`);
            // Progress and completion handled by socket events
        } catch {
            toast.error('Failed to start grid prediction');
            setIsRunningGridPrediction(false);
        }
    };

    const handleSelect = (id: string, shiftKey: boolean = false, ctrlKey: boolean = false) => {
        const currentIndex = filteredSamples.findIndex(s => s.id === id);

        if (shiftKey && lastClickedIndex !== null && currentIndex !== -1) {
            // Shift+click: select range (add to existing selection)
            const start = Math.min(lastClickedIndex, currentIndex);
            const end = Math.max(lastClickedIndex, currentIndex);
            const rangeIds = filteredSamples.slice(start, end + 1).map(s => s.id);

            setSelectedIds(prev => {
                const next = new Set(prev);
                for (const rangeId of rangeIds) {
                    next.add(rangeId);
                }
                return next;
            });
        } else if (ctrlKey) {
            // Ctrl+click: toggle selection (add or remove from existing selection)
            setSelectedIds(prev => {
                const next = new Set(prev);
                if (next.has(id)) {
                    next.delete(id);
                } else {
                    next.add(id);
                }
                return next;
            });
            setLastClickedIndex(currentIndex);
        } else {
            // Normal click: select single item (clear others)
            setSelectedIds(new Set([id]));
            setLastClickedIndex(currentIndex);
        }
    };

    const handleDoubleClick = (id: string) => {
        // Open in new tab on double-click, passing first selected model
        const modelParam = selectedModelIds.length > 0 ? `?model=${selectedModelIds[0]}` : '';
        window.open(`/admin/images/${id}${modelParam}`, '_blank');
    };

    const selectAll = () => {
        setSelectedIds(new Set(filteredSamples.map(s => s.id)));
    };

    const clearSelection = () => {
        setSelectedIds(new Set());
    };

    const handleDeleteSelected = async () => {
        if (selectedIds.size === 0) {
            return;
        }

        setIsDeleting(true);
        try {
            await http.delete('/api/samples', { ids: Array.from(selectedIds) });
            toast.success(`Deleted ${selectedIds.size} image${selectedIds.size > 1 ? 's' : ''}`);
            removeSamples(Array.from(selectedIds));
            setSelectedIds(new Set());
            setShowDeleteModal(false);
        } catch {
            toast.error('Failed to delete images');
        } finally {
            setIsDeleting(false);
        }
    };

    // Socket event handlers for toast notifications (state is managed by AdminContext)
    useSocketEvent('training-run-completed', () => {
        setIsTraining(false);
        toast.success('Training completed successfully');
    }, []);

    useSocketEvent('training-run-failed', (data) => {
        setIsTraining(false);
        toast.error(data.run.errorMessage || 'Training failed');
    }, []);

    useSocketEvent('detection-run-completed', () => {
        toast.success('Detection completed successfully');
    }, []);

    useSocketEvent('detection-run-failed', (data) => {
        toast.error(data.run.errorMessage || 'Detection failed');
    }, []);

    useSocketEvent('detection-run-deleted', () => {
        // Reload samples to reflect cleared AI labels
        const modelId = selectedModelIds.length === 1 ? selectedModelIds[0] : null;
        fetchSamples(true, modelId, selectedTagId || null);
    }, [fetchSamples, selectedModelIds, selectedTagId]);

    // Socket event handlers for corner detection
    useSocketEvent('corner-run-completed', async (data) => {
        setIsRunningCorners(false);
        toast.success('Corner detection completed');
        // Load corner detection results for current samples
        if (data.run.id === activeCornerRunId) {
            try {
                const results = await ML_API.getCornerDetections(data.run.id);
                const newMap = new Map<string, CornerDetectionResult>();
                for (const result of results) {
                    newMap.set(result.sampleId, result);
                }
                setCornerDetections(newMap);
            } catch {
                // Error toast shown by http client
            }
        }
    }, [activeCornerRunId]);

    useSocketEvent('corner-run-failed', (data) => {
        setIsRunningCorners(false);
        toast.error(data.run.errorMessage || 'Corner detection failed');
    }, []);

    // Socket event handlers for grid prediction
    useSocketEvent('grid-run-completed', async (data) => {
        setIsRunningGridPrediction(false);
        toast.success(`Grid prediction completed: ${data.successCount} succeeded, ${data.failedCount} failed`);
        // Reload grid predictions
        const sampleIds = samples.map(s => s.id);
        if (sampleIds.length > 0) {
            try {
                const predictions = await ML_API.getGridPredictions(sampleIds);
                const newMap = new Map<string, GridPredictionRecord>();
                for (const pred of predictions) {
                    newMap.set(pred.sampleId, pred);
                }
                setGridPredictions(newMap);
            } catch {
                // Silently ignore
            }
        }
    }, [samples]);

    useSocketEvent('grid-run-failed', (data) => {
        setIsRunningGridPrediction(false);
        toast.error(data.errorMessage || 'Grid prediction failed');
    }, []);

    // Refetch samples when selected model or tag changes (single model mode)
    useEffect(() => {
        if (selectedModelIds.length <= 1) {
            const modelId = selectedModelIds.length === 1 ? selectedModelIds[0] : null;
            fetchSamples(true, modelId, selectedTagId || null);
        }
    }, [selectedModelIds, selectedTagId, fetchSamples]);

    // Load corner detections from database when samples change
    useEffect(() => {
        const sampleIds = samples.map(s => s.id);
        if (sampleIds.length === 0) {
            return;
        }
        ML_API.getCornerDetectionsForSamples(sampleIds)
            .then(results => {
                const newMap = new Map<string, CornerDetectionResult>();
                for (const result of results) {
                    newMap.set(result.sampleId, result);
                }
                setCornerDetections(newMap);
            })
            .catch(() => {
                // Silently ignore errors - corner detections are optional
            });
    }, [samples]);

    // Load grid predictions when samples change
    useEffect(() => {
        const sampleIds = samples.map(s => s.id);
        if (sampleIds.length === 0) {
            return;
        }
        ML_API.getGridPredictions(sampleIds)
            .then(predictions => {
                const newMap = new Map<string, GridPredictionRecord>();
                for (const pred of predictions) {
                    newMap.set(pred.sampleId, pred);
                }
                setGridPredictions(newMap);
            })
            .catch(() => {
                // Silently ignore errors
            });
    }, [samples]);

    // Load grid labels when samples change
    useEffect(() => {
        const sampleIds = samples.map(s => s.id);
        if (sampleIds.length === 0) {
            return;
        }
        ML_API.getGridLabels(sampleIds)
            .then(labels => {
                const newMap = new Map<string, GridLabel>();
                for (const label of labels) {
                    newMap.set(label.sampleId, label);
                }
                setGridLabels(newMap);
            })
            .catch(() => {
                // Silently ignore errors
            });
    }, [samples]);

    // Fetch samples for each model when multiple models selected
    useEffect(() => {
        if (selectedModelIds.length <= 1 || preset === 'all') {
            setMultiModelSamples(new Map());
            return;
        }

        const fetchMultiModelSamples = async () => {
            setIsLoadingMultiModel(true);
            const newMap = new Map<string, Set<string>>();

            for (const modelId of selectedModelIds) {
                try {
                    const response = await http.get<{ samples: Sample[] }>(`/api/samples?modelId=${modelId}&filter=${preset}`);
                    newMap.set(modelId, new Set(response.samples.map(s => s.id)));
                } catch {
                    // Ignore errors
                }
            }

            setMultiModelSamples(newMap);
            setIsLoadingMultiModel(false);
        };

        fetchMultiModelSamples();
    }, [selectedModelIds, preset]);

    // Get completed models for the dropdown
    const completedModels = useMemo(() =>
        trainingRuns.filter(r => r.status === 'completed'),
    [trainingRuns]
    );

    // Compute filtered samples based on multi-model mode
    const filteredSamples = useMemo(() => {
        let filtered = samples;

        // Apply multi-model filtering when multiple models selected and not "all" filter
        if (selectedModelIds.length > 1 && preset !== 'all' && preset !== 'selected' && multiModelSamples.size > 0) {
            const sampleSets = Array.from(multiModelSamples.values());

            if (multiModelMode === 'intersection') {
                // Sample must be in ALL selected models
                filtered = samples.filter(s => sampleSets.every(set => set.has(s.id)));
            } else {
                // Sample must be in ANY selected model
                filtered = samples.filter(s => sampleSets.some(set => set.has(s.id)));
            }
        } else {
            // Single model or "all"/"selected" filter - use normal filtering
            filtered = samples.filter(s => {
                if (!matchesPreset(s, preset, selectedIds)) {
                    return false;
                }
                return true;
            });
        }

        // Apply search filter
        if (searchQuery) {
            filtered = filtered.filter(s => s.id.toLowerCase().includes(searchQuery.toLowerCase()));
        }

        return sortSamples(filtered, sort);
    }, [samples, selectedModelIds, preset, multiModelMode, multiModelSamples, searchQuery, sort, selectedIds]);

    // Reset page when filters change
    useEffect(() => {
        setCurrentPage(1);
    }, [preset, sort, samples.length, searchQuery]);

    // Paginate grid view samples
    const paginatedGridSamples = useMemo(() => {
        const startIndex = (currentPage - 1) * pageSize;
        return filteredSamples.slice(startIndex, startIndex + pageSize);
    }, [filteredSamples, currentPage, pageSize]);

    // Sort samples for list view
    const listSortedSamples = useMemo(() => {
        if (viewType !== 'list' || !listSortState.column) {
            return filteredSamples;
        }
        const getStatusPriority = (s: Sample) => {
            if (s.status.humanLabeled) {
                return 2;
            }
            if (s.status.aiLabeled) {
                return 1;
            }
            return 0;
        };
        const getBoardAgreed = (s: Sample): number => {
            if (!s.ai || !s.human) {
                return -1;
            }
            return s.ai.hasBoardDetected === s.human.hasBoardDetected ? 1 : 0;
        };
        const getEntireAgreed = (s: Sample): number => {
            if (!s.ai || !s.human) {
                return -1;
            }
            return s.ai.isEntireBoardVisible === s.human.isEntireBoardVisible ? 1 : 0;
        };
        return [...filteredSamples].sort((a, b) => {
            let comparison = 0;
            switch (listSortState.column) {
            case 'id':
                comparison = a.id.localeCompare(b.id);
                break;
            case 'created_at':
                comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
                break;
            case 'iou':
                comparison = (a.iou ?? -1) - (b.iou ?? -1);
                break;
            case 'confidence':
                comparison = (a.ai?.confidence ?? -1) - (b.ai?.confidence ?? -1);
                break;
            case 'status':
                comparison = getStatusPriority(a) - getStatusPriority(b);
                break;
            case 'boardAgreed':
                comparison = getBoardAgreed(a) - getBoardAgreed(b);
                break;
            case 'entireAgreed':
                comparison = getEntireAgreed(a) - getEntireAgreed(b);
                break;
            case 'bestErr':
                comparison = (cornerDetections.get(a.id)?.best.error ?? -1) - (cornerDetections.get(b.id)?.best.error ?? -1);
                break;
            case 'houghErr':
                comparison = (cornerDetections.get(a.id)?.hough.error ?? -1) - (cornerDetections.get(b.id)?.hough.error ?? -1);
                break;
            case 'edgeErr':
                comparison = (cornerDetections.get(a.id)?.edge.error ?? -1) - (cornerDetections.get(b.id)?.edge.error ?? -1);
                break;
            case 'gridAcc': {
                const metricsA = calculateGridMetrics(gridPredictions.get(a.id)?.grid, gridLabels.get(a.id)?.grid);
                const metricsB = calculateGridMetrics(gridPredictions.get(b.id)?.grid, gridLabels.get(b.id)?.grid);
                comparison = (metricsA?.accuracy ?? -1) - (metricsB?.accuracy ?? -1);
                break;
            }
            case 'gridF1Min': {
                const metricsA = calculateGridMetrics(gridPredictions.get(a.id)?.grid, gridLabels.get(a.id)?.grid);
                const metricsB = calculateGridMetrics(gridPredictions.get(b.id)?.grid, gridLabels.get(b.id)?.grid);
                comparison = (metricsA?.minF1 ?? -1) - (metricsB?.minF1 ?? -1);
                break;
            }
            case 'gridRecallMin': {
                const metricsA = calculateGridMetrics(gridPredictions.get(a.id)?.grid, gridLabels.get(a.id)?.grid);
                const metricsB = calculateGridMetrics(gridPredictions.get(b.id)?.grid, gridLabels.get(b.id)?.grid);
                comparison = (metricsA?.minRecall ?? -1) - (metricsB?.minRecall ?? -1);
                break;
            }
            }
            return listSortState.direction === 'asc' ? comparison : -comparison;
        });
    }, [filteredSamples, viewType, listSortState, cornerDetections, gridPredictions, gridLabels]);

    // Calculate average corner errors for column headers
    const avgCornerErrors = useMemo(() => {
        const samplesWithCorners = filteredSamples.filter(s => cornerDetections.has(s.id));
        if (samplesWithCorners.length === 0) {
            return { best: null, hough: null, edge: null };
        }
        const calcAvg = (getter: (cd: CornerDetectionResult) => number | null) => {
            const values = samplesWithCorners
                .map(s => getter(cornerDetections.get(s.id)!))
                .filter((v): v is number => v !== null);
            if (values.length === 0) {
                return null;
            }
            return values.reduce((a, b) => a + b, 0) / values.length;
        };
        return {
            best: calcAvg(cd => cd.best.error),
            hough: calcAvg(cd => cd.hough.error),
            edge: calcAvg(cd => cd.edge.error),
        };
    }, [filteredSamples, cornerDetections]);

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleDateString(undefined, {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    // List view columns
    const listColumns: Column<Sample>[] = [
        {
            key: 'thumbnail',
            header: '',
            width: '64px',
            render: (sample) => (
                <img
                    src={sample.thumbnailUrl || sample.imageUrl || `${API_URL}/api/samples/${sample.id}/image`}
                    alt={`Sample ${sample.id}`}
                    className="w-12 h-12 object-cover rounded"
                    loading="lazy"
                />
            ),
        },
        {
            key: 'id',
            header: 'ID',
            sortable: true,
            width: '100px',
            render: (sample) => <span className="font-mono text-muted">{formatShortId(sample.id)}</span>,
        },
        {
            key: 'status',
            header: 'Status',
            sortable: true,
            noTruncate: true,
            render: (sample) => {
                const allLabels = getStatusLabels(sample.status);
                return (
                    <div className="flex gap-1">
                        {allLabels.map(label => (
                            <span key={label} className={`px-1.5 py-0.5 text-xs rounded ${statusColors[label]}`}>
                                {statusLabels[label]}
                            </span>
                        ))}
                    </div>
                );
            },
        },
        {
            key: 'boardAgreed',
            header: 'Board visible agreed',
            sortable: true,
            render: (sample) => {
                if (!sample.ai || !sample.human) {
                    return '—';
                }
                const agreed = sample.ai.hasBoardDetected === sample.human.hasBoardDetected;
                return <span className={agreed ? 'text-success' : 'text-error'}>{agreed ? 'Yes' : 'No'}</span>;
            },
        },
        {
            key: 'entireAgreed',
            header: 'Entire board visible agreed',
            sortable: true,
            render: (sample) => {
                if (!sample.ai || !sample.human) {
                    return '—';
                }
                const agreed = sample.ai.isEntireBoardVisible === sample.human.isEntireBoardVisible;
                return <span className={agreed ? 'text-success' : 'text-error'}>{agreed ? 'Yes' : 'No'}</span>;
            },
        },
        {
            key: 'confidence',
            header: 'Confidence',
            sortable: true,
            width: '100px',
            render: (sample) => sample.ai ? `${Math.round(sample.ai.confidence * 100)}%` : '—',
        },
        {
            key: 'iou',
            header: 'IoU',
            sortable: true,
            width: '80px',
            render: (sample) => sample.iou !== null ? (
                <span className={
                    sample.iou >= 0.75 ? 'text-success' :
                        sample.iou >= 0.5 ? 'text-warning' : 'text-error'
                }>
                    {Math.round(sample.iou * 100)}%
                </span>
            ) : '—',
        },
        {
            key: 'tags',
            header: 'Tags',
            noTruncate: true,
            render: (sample) => sample.tags && sample.tags.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                    {sample.tags.map(tag => (
                        <span key={tag.id} className="px-1.5 py-0.5 text-xs bg-primary/10 text-primary rounded">
                            {tag.name}
                        </span>
                    ))}
                </div>
            ) : <span className="text-muted">—</span>,
        },
        {
            key: 'gridAcc',
            header: 'Grid Acc',
            sortable: true,
            width: '80px',
            render: (sample) => {
                const aiPred = gridPredictions.get(sample.id);
                const humanLabel = gridLabels.get(sample.id);
                const metrics = calculateGridMetrics(aiPred?.grid, humanLabel?.grid);
                if (!metrics) {
                    return <span className="text-muted">—</span>;
                }
                const pct = metrics.accuracy * 100;
                const cls = pct >= 90 ? 'text-success' : pct >= 70 ? 'text-warning' : 'text-error';
                return <span className={cls}>{pct.toFixed(0)}%</span>;
            },
        },
        {
            key: 'gridF1Min',
            header: 'Grid F1 Min',
            sortable: true,
            width: '90px',
            render: (sample) => {
                const aiPred = gridPredictions.get(sample.id);
                const humanLabel = gridLabels.get(sample.id);
                const metrics = calculateGridMetrics(aiPred?.grid, humanLabel?.grid);
                if (!metrics) {
                    return <span className="text-muted">—</span>;
                }
                const pct = metrics.minF1 * 100;
                const cls = pct >= 90 ? 'text-success' : pct >= 70 ? 'text-warning' : 'text-error';
                return <span className={cls}>{pct.toFixed(0)}%</span>;
            },
        },
        {
            key: 'gridRecallMin',
            header: 'Grid Recall Min',
            sortable: true,
            width: '110px',
            render: (sample) => {
                const aiPred = gridPredictions.get(sample.id);
                const humanLabel = gridLabels.get(sample.id);
                const metrics = calculateGridMetrics(aiPred?.grid, humanLabel?.grid);
                if (!metrics) {
                    return <span className="text-muted">—</span>;
                }
                const pct = metrics.minRecall * 100;
                const cls = pct >= 90 ? 'text-success' : pct >= 70 ? 'text-warning' : 'text-error';
                return <span className={cls}>{pct.toFixed(0)}%</span>;
            },
        },
        {
            key: 'bestErr',
            header: avgCornerErrors.best !== null
                ? `Best (${(avgCornerErrors.best * 100).toFixed(1)}%)`
                : 'Best Err',
            sortable: true,
            width: '180px',
            render: (sample) => {
                const cd = cornerDetections.get(sample.id);
                if (!cd || cd.best.error === null) {
                    return <span className="text-muted">—</span>;
                }
                const pct = cd.best.error * 100;
                const cls = pct < 5 ? 'text-success' : pct < 10 ? 'text-warning' : 'text-error';
                const q = cd.best.quality;
                return (
                    <span className={cls}>
                        {pct.toFixed(1)}%
                        {q && <span className="text-muted text-[10px] ml-1">({q.aspectRatio?.toFixed(2)}, {q.areaRatio?.toFixed(2)}, {q.avgAngle?.toFixed(0)}°)</span>}
                    </span>
                );
            },
        },
        {
            key: 'created_at',
            header: 'Created',
            sortable: true,
            width: '140px',
            className: 'text-muted',
            render: (sample) => formatDate(sample.created_at),
        },
    ];

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-3">
                <Link
                    to="/admin"
                    className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                >
                    <ChevronLeft className="w-4 h-4" />
                </Link>
                <h1 className="text-xl font-medium text-foreground">Training</h1>
            </div>

            {/* Action buttons - responsive grid */}
            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleFileChange}
                className="hidden"
            />
            <div className="grid grid-cols-2 sm:flex sm:flex-wrap gap-2">
                <Button
                    onClick={() => navigate('/admin/images/label')}
                    variant="secondary"
                    leftIcon={<Tag className="w-4 h-4" />}
                    size="sm"
                    className="justify-center"
                >
                    <span className="hidden sm:inline">Data labeling</span>
                    <span className="sm:hidden">Label</span>
                </Button>
                <Button
                    onClick={() => setShowDetectionModal(true)}
                    variant="secondary"
                    leftIcon={<Search className="w-4 h-4" />}
                    size="sm"
                    className="justify-center"
                >
                    <span className="hidden sm:inline">Run AI Detection</span>
                    <span className="sm:hidden">Detect</span>
                </Button>
                <Button
                    onClick={handleTrain}
                    loading={isTraining}
                    disabled={selectedIds.size === 0}
                    leftIcon={<Lightbulb className="w-4 h-4" />}
                    size="sm"
                    className="justify-center"
                    title={selectedIds.size === 0 ? 'Select images to train on' : `Train on ${selectedIds.size} selected images`}
                >
                    Train{selectedIds.size > 0 ? ` (${selectedIds.size})` : ''}
                </Button>
                <Button
                    onClick={handleRunCorners}
                    loading={isRunningCorners}
                    disabled={selectedIds.size === 0}
                    leftIcon={<Focus className="w-4 h-4" />}
                    size="sm"
                    variant="secondary"
                    className="justify-center"
                    title={selectedIds.size === 0 ? 'Select images to run corner detection' : `Run corner detection on ${selectedIds.size} selected images`}
                >
                    {isRunningCorners && activeCornerRunId && cornerProgress[activeCornerRunId]
                        ? `${cornerProgress[activeCornerRunId].processedCount}/${cornerProgress[activeCornerRunId].totalCount}`
                        : `Corners${selectedIds.size > 0 ? ` (${selectedIds.size})` : ''}`}
                </Button>
                <Button
                    onClick={handleRunGridPrediction}
                    loading={isRunningGridPrediction}
                    disabled={selectedIds.size === 0 && !isRunningGridPrediction}
                    leftIcon={<Grid3X3 className="w-4 h-4" />}
                    size="sm"
                    variant="secondary"
                    className="justify-center"
                    title={selectedIds.size === 0 ? 'Select samples first' : 'Run AI grid cell prediction on selected samples'}
                >
                    {isRunningGridPrediction && gridProgress
                        ? `${gridProgress.processedCount}/${gridProgress.totalCount}`
                        : isRunningGridPrediction
                            ? 'Starting...'
                            : `Cells${selectedIds.size > 0 ? ` (${selectedIds.size})` : ''}`}
                </Button>
                {tab === 'images' && (
                    <Button
                        onClick={handleUploadClick}
                        loading={isUploading}
                        leftIcon={<Plus className="w-4 h-4" />}
                        size="sm"
                        className="justify-center"
                    >
                        {isUploading && uploadProgress.total > 1
                            ? `${uploadProgress.completed}/${uploadProgress.total}`
                            : 'Upload'}
                    </Button>
                )}
            </div>

            {/* Tabs */}
            <div className="flex gap-1 border-b border-border overflow-x-auto overflow-y-hidden scrollbar-hide">
                <button
                    onClick={() => setTab('images')}
                    className={`
                        px-4 py-2 text-sm font-medium transition-colors -mb-px whitespace-nowrap
                        ${tab === 'images'
            ? 'border-b-2 border-primary text-foreground'
            : 'text-secondary hover:text-foreground'
        }
                    `}
                >
                    Training Images
                    <span className="ml-2 px-1.5 py-0.5 text-xs bg-surface-dim rounded-full">
                        {samples.length}
                    </span>
                </button>
                <button
                    onClick={() => setTab('models')}
                    className={`
                        px-4 py-2 text-sm font-medium transition-colors -mb-px whitespace-nowrap
                        ${tab === 'models'
            ? 'border-b-2 border-primary text-foreground'
            : 'text-secondary hover:text-foreground'
        }
                    `}
                >
                    Trained Models
                    {trainingRuns.length > 0 && (
                        <span className="ml-2 px-1.5 py-0.5 text-xs bg-surface-dim rounded-full">
                            {trainingRuns.length}
                        </span>
                    )}
                </button>
                <button
                    onClick={() => setTab('detections')}
                    className={`
                        px-4 py-2 text-sm font-medium transition-colors -mb-px whitespace-nowrap
                        ${tab === 'detections'
            ? 'border-b-2 border-primary text-foreground'
            : 'text-secondary hover:text-foreground'
        }
                    `}
                >
                    Detection Runs
                    {detectionRuns.length > 0 && (
                        <span className="ml-2 px-1.5 py-0.5 text-xs bg-surface-dim rounded-full">
                            {detectionRuns.length}
                        </span>
                    )}
                </button>
                <button
                    onClick={() => setTab('test')}
                    className={`
                        px-4 py-2 text-sm font-medium transition-colors -mb-px whitespace-nowrap
                        ${tab === 'test'
            ? 'border-b-2 border-primary text-foreground'
            : 'text-secondary hover:text-foreground'
        }
                    `}
                >
                    Test Model
                </button>
                <button
                    onClick={() => setTab('config')}
                    className={`
                        px-4 py-2 text-sm font-medium transition-colors -mb-px whitespace-nowrap
                        ${tab === 'config'
            ? 'border-b-2 border-primary text-foreground'
            : 'text-secondary hover:text-foreground'
        }
                    `}
                >
                    Configurations
                </button>
            </div>

            {tab === 'images' ? (
                <>
                    {/* Filter, Sort, Model Dropdowns, Selection and View Toggle */}
                    <div className="flex flex-wrap items-end gap-3">
                        {/* Search input */}
                        <Input
                            label="Search"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search by ID..."
                            icon={<Search className="w-4 h-4" />}
                            rightIcon={searchQuery ? (
                                <button
                                    onClick={() => setSearchQuery('')}
                                    className="text-muted hover:text-foreground"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            ) : undefined}
                            className="w-40"
                        />
                        <Select
                            label="Filter"
                            value={preset}
                            onChange={(v) => setPreset(v as FilterPreset)}
                            options={Object.entries(filterPresetLabels).map(([value, label]) => ({ value, label }))}
                            className="w-40"
                        />
                        {viewType === 'grid' && (
                            <Select
                                label="Sort"
                                value={sort}
                                onChange={(v) => setSort(v as SortOption)}
                                options={Object.entries(sortLabels).map(([value, label]) => ({ value, label }))}
                                className="w-40"
                            />
                        )}
                        {(completedModels.length > 0 || pipelines.length > 0) && (
                            <MultiSelect
                                label="Models"
                                options={[
                                    ...completedModels.map(run => ({
                                        value: run.id,
                                        label: `${formatShortId(run.id)} - ${formatDate(run.createdAt)} (${run.imageCount} imgs)`,
                                    })),
                                    ...pipelines.map(pipeline => ({
                                        value: `pipeline:${pipeline.id}`,
                                        label: `\u{1F517} ${pipeline.name}`,
                                    })),
                                ]}
                                selected={selectedModelIds}
                                onChange={setSelectedModelIds}
                                placeholder="All models"
                                allLabel="All models"
                                className="w-64"
                            />
                        )}
                        {tags.length > 0 && (
                            <Select
                                label="Tags"
                                value={selectedTagId}
                                onChange={(v) => setSelectedTagId(String(v))}
                                options={[
                                    { value: '', label: 'All tags' },
                                    ...tags.map(tag => ({ value: tag.id, label: tag.name })),
                                ]}
                                className="w-40"
                            />
                        )}
                        {/* Visibility toggles for overlays */}
                        {viewType === 'grid' && (
                            <div className="flex flex-col gap-1">
                                <label className="block text-sm font-medium text-foreground">Overlays</label>
                                <div className="flex items-center gap-3 h-9 px-2 border border-border rounded-md bg-surface">
                                    <span className="text-xs text-muted">Box:</span>
                                    <label className="flex items-center gap-1 cursor-pointer" title="Human bounding box">
                                        <input
                                            type="checkbox"
                                            checked={overlayVisibility.humanBox}
                                            onChange={(e) => setOverlayVisibility(prev => ({ ...prev, humanBox: e.target.checked }))}
                                            className="w-3 h-3 rounded border-border"
                                        />
                                        <span className="text-xs text-green-500">H</span>
                                    </label>
                                    <label className="flex items-center gap-1 cursor-pointer" title="AI bounding box">
                                        <input
                                            type="checkbox"
                                            checked={overlayVisibility.aiBox}
                                            onChange={(e) => setOverlayVisibility(prev => ({ ...prev, aiBox: e.target.checked }))}
                                            className="w-3 h-3 rounded border-border"
                                        />
                                        <span className="text-xs text-blue-500">AI</span>
                                    </label>
                                    <div className="w-px h-4 bg-border" />
                                    <span className="text-xs text-muted">Corners:</span>
                                    <label className="flex items-center gap-1 cursor-pointer" title="Human corners">
                                        <input
                                            type="checkbox"
                                            checked={overlayVisibility.humanCorners}
                                            onChange={(e) => setOverlayVisibility(prev => ({ ...prev, humanCorners: e.target.checked }))}
                                            className="w-3 h-3 rounded border-border"
                                        />
                                        <span className="text-xs text-green-500">H</span>
                                    </label>
                                    <label className="flex items-center gap-1 cursor-pointer" title="AI corners">
                                        <input
                                            type="checkbox"
                                            checked={overlayVisibility.aiCorners}
                                            onChange={(e) => setOverlayVisibility(prev => ({ ...prev, aiCorners: e.target.checked }))}
                                            className="w-3 h-3 rounded border-border"
                                        />
                                        <span className="text-xs text-blue-500">AI</span>
                                    </label>
                                    <label className="flex items-center gap-1 cursor-pointer" title="Best (auto-selected)">
                                        <input
                                            type="checkbox"
                                            checked={overlayVisibility.bestCorners}
                                            onChange={(e) => setOverlayVisibility(prev => ({ ...prev, bestCorners: e.target.checked }))}
                                            className="w-3 h-3 rounded border-border"
                                        />
                                        <span className="text-xs text-yellow-500">B</span>
                                    </label>
                                </div>
                            </div>
                        )}
                        {/* Intersection/Union toggle - show when multiple models selected and filter is not "all" */}
                        {selectedModelIds.length > 1 && preset !== 'all' && (
                            <div className="flex flex-col gap-1">
                                <label className="block text-sm font-medium text-foreground mb-1">Mode</label>
                                <div className="flex items-center border border-border rounded-md overflow-hidden h-9">
                                    <button
                                        onClick={() => setMultiModelMode('intersection')}
                                        className={cn(
                                            'px-3 h-full text-sm transition-colors',
                                            multiModelMode === 'intersection'
                                                ? 'bg-primary text-white'
                                                : 'bg-surface text-secondary hover:text-foreground hover:bg-surface-dim'
                                        )}
                                        title="Show samples matching filter in ALL selected models"
                                    >
                                        All
                                    </button>
                                    <button
                                        onClick={() => setMultiModelMode('union')}
                                        className={cn(
                                            'px-3 h-full text-sm transition-colors',
                                            multiModelMode === 'union'
                                                ? 'bg-primary text-white'
                                                : 'bg-surface text-secondary hover:text-foreground hover:bg-surface-dim'
                                        )}
                                        title="Show samples matching filter in ANY selected model"
                                    >
                                        Any
                                    </button>
                                </div>
                            </div>
                        )}
                        {/* Spacer to push right side elements */}
                        <div className="flex-1" />
                        {/* Selection info and actions */}
                        {selectedIds.size > 0 && (
                            <>
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-medium text-primary">
                                        {selectedIds.size} selected
                                    </span>
                                    <button
                                        onClick={selectAll}
                                        className="text-xs text-primary hover:underline"
                                    >
                                        Select all ({filteredSamples.length})
                                    </button>
                                    <button
                                        onClick={clearSelection}
                                        className="text-xs text-secondary hover:underline"
                                    >
                                        Clear
                                    </button>
                                </div>
                                <Button
                                    onClick={() => setShowDeleteModal(true)}
                                    variant="danger"
                                    size="sm"
                                    leftIcon={<Trash2 className="w-4 h-4" />}
                                >
                                    Delete
                                </Button>
                                <div className="h-8 w-px bg-border" />
                            </>
                        )}
                        <span className="text-xs text-muted">
                            {isLoadingMultiModel ? 'Loading...' : `${filteredSamples.length} of ${samples.length} images`}
                        </span>
                        {/* View Toggle */}
                        <div className="flex items-center border border-border rounded-md overflow-hidden">
                            <button
                                onClick={() => setViewType('list')}
                                className={`p-2 transition-colors ${viewType === 'list' ? 'bg-surface-dim text-foreground' : 'text-muted hover:text-foreground hover:bg-surface-dim/50'}`}
                                title="List view"
                            >
                                <List className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => setViewType('grid')}
                                className={`p-2 transition-colors ${viewType === 'grid' ? 'bg-surface-dim text-foreground' : 'text-muted hover:text-foreground hover:bg-surface-dim/50'}`}
                                title="Grid view"
                            >
                                <LayoutGrid className="w-4 h-4" />
                            </button>
                        </div>
                    </div>

                    {/* Images Grid/List */}
                    {isLoading ? (
                        viewType === 'grid' ? (
                            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                                {[...Array(10)].map((_, i) => (
                                    <div key={i} className="bg-surface rounded-lg border border-border overflow-hidden">
                                        <div className="aspect-square bg-surface-dim animate-pulse" />
                                        <div className="p-2 space-y-1.5">
                                            <div className="h-3 bg-surface-dim rounded animate-pulse w-2/3" />
                                            <div className="h-2.5 bg-surface-dim rounded animate-pulse w-1/3" />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="bg-surface rounded-lg border border-border overflow-hidden">
                                <div className="animate-pulse p-4 space-y-3">
                                    {[...Array(10)].map((_, i) => (
                                        <div key={i} className="h-12 bg-surface-dim rounded" />
                                    ))}
                                </div>
                            </div>
                        )
                    ) : filteredSamples.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-16 text-center">
                            <div className="w-16 h-16 bg-surface-dim rounded-full flex items-center justify-center mb-4">
                                <Image className="w-8 h-8 text-muted" />
                            </div>
                            <p className="text-secondary font-medium">No images found</p>
                            <p className="text-sm text-muted mt-1">Upload board images to get started</p>
                        </div>
                    ) : viewType === 'list' ? (
                        /* List View */
                        <>
                            <Table<Sample>
                                columns={listColumns}
                                data={listSortedSamples}
                                getRowKey={(sample) => sample.id}
                                sortState={listSortState}
                                onSort={handleListSort}
                                selectable
                                selectedKeys={selectedIds}
                                onSelectionChange={(key, shiftKey, ctrlKey) => handleSelect(key as string, shiftKey, ctrlKey)}
                                onRowDoubleClick={(sample) => handleDoubleClick(sample.id)}
                                emptyMessage="No images found"
                                paginated
                                currentPage={currentPage}
                                pageSize={pageSize}
                                totalItems={filteredSamples.length}
                                onPageChange={setCurrentPage}
                            />
                        </>
                    ) : (
                        /* Grid View */
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                            {paginatedGridSamples.map((sample) => {
                                const allLabels = getStatusLabels(sample.status);
                                const isSelected = selectedIds.has(sample.id);

                                return (
                                    <div
                                        key={sample.id}
                                        onClick={(e) => handleSelect(sample.id, e.shiftKey, e.ctrlKey || e.metaKey)}
                                        onDoubleClick={() => handleDoubleClick(sample.id)}
                                        className={`relative bg-surface rounded-lg border overflow-hidden text-left transition-all cursor-pointer select-none ${
                                            isSelected
                                                ? 'border-primary ring-2 ring-primary/30 bg-primary/5 hover:bg-primary/10'
                                                : 'border-border hover:bg-surface-dim'
                                        }`}
                                    >
                                        <div className="relative">
                                            <img
                                                src={sample.thumbnailUrl || sample.imageUrl || `${API_URL}/api/samples/${sample.id}/image`}
                                                alt={`Sample ${sample.id}`}
                                                className="w-full h-auto"
                                                loading="lazy"
                                                draggable={false}
                                            />
                                            {/* Tags overlay */}
                                            {sample.tags && sample.tags.length > 0 && (
                                                <div className="absolute top-1 left-1/2 -translate-x-1/2 flex flex-wrap justify-center gap-1 max-w-[90%]">
                                                    {sample.tags.map(tag => (
                                                        <span
                                                            key={tag.id}
                                                            className="px-1.5 py-0.5 text-[10px] bg-surface/80 text-foreground rounded backdrop-blur-sm"
                                                        >
                                                            {tag.name}
                                                        </span>
                                                    ))}
                                                </div>
                                            )}
                                            {/* AI bounding box (dashed, blue) */}
                                            {overlayVisibility.aiBox && sample.ai?.boundingBox && (
                                                <div
                                                    className="absolute border-2 border-blue-500 border-dashed pointer-events-none"
                                                    style={{
                                                        left: `${sample.ai.boundingBox.x * 100}%`,
                                                        top: `${sample.ai.boundingBox.y * 100}%`,
                                                        width: `${sample.ai.boundingBox.width * 100}%`,
                                                        height: `${sample.ai.boundingBox.height * 100}%`,
                                                    }}
                                                />
                                            )}
                                            {/* Human bounding box (solid, green) */}
                                            {overlayVisibility.humanBox && sample.human?.boundingBox && (
                                                <div
                                                    className="absolute border-2 border-green-500 pointer-events-none"
                                                    style={{
                                                        left: `${sample.human.boundingBox.x * 100}%`,
                                                        top: `${sample.human.boundingBox.y * 100}%`,
                                                        width: `${sample.human.boundingBox.width * 100}%`,
                                                        height: `${sample.human.boundingBox.height * 100}%`,
                                                    }}
                                                />
                                            )}
                                            {/* Corner overlays (SVG polygons) */}
                                            <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                                                {/* Human corners - green */}
                                                {overlayVisibility.humanCorners && sample.human?.corners && (
                                                    <polygon
                                                        points={`${sample.human.corners.topLeft.x * 100},${sample.human.corners.topLeft.y * 100} ${sample.human.corners.topRight.x * 100},${sample.human.corners.topRight.y * 100} ${sample.human.corners.bottomRight.x * 100},${sample.human.corners.bottomRight.y * 100} ${sample.human.corners.bottomLeft.x * 100},${sample.human.corners.bottomLeft.y * 100}`}
                                                        fill="none"
                                                        stroke="#22c55e"
                                                        strokeWidth="1"
                                                        vectorEffect="non-scaling-stroke"
                                                    />
                                                )}
                                                {/* AI corners - blue */}
                                                {overlayVisibility.aiCorners && sample.ai?.corners && (
                                                    <polygon
                                                        points={`${sample.ai.corners.topLeft.x * 100},${sample.ai.corners.topLeft.y * 100} ${sample.ai.corners.topRight.x * 100},${sample.ai.corners.topRight.y * 100} ${sample.ai.corners.bottomRight.x * 100},${sample.ai.corners.bottomRight.y * 100} ${sample.ai.corners.bottomLeft.x * 100},${sample.ai.corners.bottomLeft.y * 100}`}
                                                        fill="none"
                                                        stroke="#3b82f6"
                                                        strokeWidth="1"
                                                        strokeDasharray="2 1"
                                                        vectorEffect="non-scaling-stroke"
                                                    />
                                                )}
                                                {/* Best corners - yellow */}
                                                {overlayVisibility.bestCorners && cornerDetections.get(sample.id)?.best.corners && (() => {
                                                    const c = cornerDetections.get(sample.id)!.best.corners!;
                                                    return (
                                                        <polygon
                                                            points={`${c.topLeft.x * 100},${c.topLeft.y * 100} ${c.topRight.x * 100},${c.topRight.y * 100} ${c.bottomRight.x * 100},${c.bottomRight.y * 100} ${c.bottomLeft.x * 100},${c.bottomLeft.y * 100}`}
                                                            fill="none"
                                                            stroke="#eab308"
                                                            strokeWidth="1"
                                                            vectorEffect="non-scaling-stroke"
                                                        />
                                                    );
                                                })()}
                                            </svg>
                                        </div>
                                        <div className="px-2 py-1.5 text-xs">
                                            {/* Header: ID + badges */}
                                            <div className="flex items-center justify-between gap-1 mb-1">
                                                <span className="text-muted font-mono">{formatShortId(sample.id)}</span>
                                                <div className="flex gap-0.5">
                                                    {allLabels.map(label => (
                                                        <span key={label} className={`px-1.5 py-0.5 text-[10px] rounded ${statusColors[label]}`}>
                                                            {statusLabels[label]}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* Detection table with metrics */}
                                            <table>
                                                <thead>
                                                    <tr className="text-muted">
                                                        <th className="text-left font-medium pr-2"></th>
                                                        <th className="text-left font-medium pr-1.5">H</th>
                                                        <th className="text-left font-medium pr-3">AI</th>
                                                        <th className="text-left font-medium text-secondary"></th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    <tr>
                                                        <td className="text-muted pr-2">Board</td>
                                                        <td className={`pr-1.5 ${sample.human ? (sample.human.hasBoardDetected ? 'text-success' : 'text-error') : 'text-muted'}`}>
                                                            {sample.human ? (sample.human.hasBoardDetected ? 'Y' : 'N') : '—'}
                                                        </td>
                                                        <td className={`pr-3 ${sample.ai ? (sample.ai.hasBoardDetected ? 'text-success' : 'text-error') : 'text-muted'}`}>
                                                            {sample.ai ? (sample.ai.hasBoardDetected ? 'Y' : 'N') : '—'}
                                                        </td>
                                                        <td className="text-secondary">
                                                            {sample.ai && (
                                                                <span>
                                                                    Confidence: {Math.round(sample.ai.confidence * 100)}%
                                                                    {sample.iou !== null && (
                                                                        <span className={
                                                                            sample.iou >= 0.75 ? ' text-success' :
                                                                                sample.iou >= 0.5 ? ' text-warning' : ' text-error'
                                                                        }> IoU: {Math.round(sample.iou * 100)}%</span>
                                                                    )}
                                                                </span>
                                                            )}
                                                        </td>
                                                    </tr>
                                                    <tr>
                                                        <td className="text-muted pr-2">Entire</td>
                                                        <td className={`pr-1.5 ${sample.human ? (sample.human.isEntireBoardVisible ? 'text-success' : 'text-error') : 'text-muted'}`}>
                                                            {sample.human ? (sample.human.isEntireBoardVisible ? 'Y' : 'N') : '—'}
                                                        </td>
                                                        <td className={`pr-3 ${sample.ai ? (sample.ai.isEntireBoardVisible ? 'text-success' : 'text-error') : 'text-muted'}`}>
                                                            {sample.ai ? (sample.ai.isEntireBoardVisible ? 'Y' : 'N') : '—'}
                                                        </td>
                                                        <td className="text-secondary">
                                                            {sample.ai?.classifierConfidence !== null && sample.ai?.classifierConfidence !== undefined && (
                                                                <span>Classifier: {Math.round(sample.ai.classifierConfidence * 100)}%</span>
                                                            )}
                                                        </td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                            {/* Corner detection errors */}
                                            {cornerDetections.has(sample.id) && (() => {
                                                const cd = cornerDetections.get(sample.id)!;
                                                const formatErr = (err: number | null) => {
                                                    if (err === null) {
                                                        return '—';
                                                    }
                                                    const pct = err * 100;
                                                    const cls = pct < 5 ? 'text-success' : pct < 10 ? 'text-warning' : 'text-error';
                                                    return <span className={cls}>{pct.toFixed(1)}%</span>;
                                                };
                                                return (
                                                    <div className="mt-1 pt-1 border-t border-border/50 flex gap-2 text-[10px]">
                                                        <span className="text-muted">Corners:</span>
                                                        <span>B:{formatErr(cd.best.error)}</span>
                                                        <span>H:{formatErr(cd.hough.error)}</span>
                                                        <span>E:{formatErr(cd.edge.error)}</span>
                                                    </div>
                                                );
                                            })()}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}

                    {/* Grid Pagination */}
                    {viewType === 'grid' && filteredSamples.length > pageSize && (
                        <div className="flex items-center justify-between px-4 py-3 border border-border rounded-md bg-surface">
                            <div className="text-xs text-secondary">
                                {paginatedGridSamples.length === 0 ? 0 : ((currentPage - 1) * pageSize) + 1}–{Math.min(currentPage * pageSize, filteredSamples.length)} of {filteredSamples.length}
                            </div>
                            <div className="flex items-center gap-1">
                                <button
                                    onClick={() => setCurrentPage(currentPage - 1)}
                                    disabled={currentPage === 1}
                                    className={cn(
                                        'p-1.5 rounded transition-colors text-secondary',
                                        currentPage === 1
                                            ? 'opacity-30 cursor-not-allowed'
                                            : 'hover:bg-surface-dim hover:text-foreground'
                                    )}
                                >
                                    <ChevronLeft className="w-4 h-4" />
                                </button>
                                <div className="flex items-center gap-0.5">
                                    {(() => {
                                        const totalPages = Math.ceil(filteredSamples.length / pageSize);
                                        const pages = [];
                                        const maxVisible = 5;

                                        let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
                                        const endPage = Math.min(totalPages, startPage + maxVisible - 1);

                                        if (endPage - startPage < maxVisible - 1) {
                                            startPage = Math.max(1, endPage - maxVisible + 1);
                                        }

                                        for (let i = startPage; i <= endPage; i++) {
                                            pages.push(
                                                <button
                                                    key={i}
                                                    onClick={() => setCurrentPage(i)}
                                                    className={cn(
                                                        'min-w-[28px] h-7 px-2 rounded text-xs font-medium transition-colors',
                                                        i === currentPage
                                                            ? 'bg-primary text-white'
                                                            : 'text-secondary hover:bg-surface-dim hover:text-foreground'
                                                    )}
                                                >
                                                    {i}
                                                </button>
                                            );
                                        }
                                        return pages;
                                    })()}
                                </div>
                                <button
                                    onClick={() => setCurrentPage(currentPage + 1)}
                                    disabled={currentPage >= Math.ceil(filteredSamples.length / pageSize)}
                                    className={cn(
                                        'p-1.5 rounded transition-colors text-secondary',
                                        currentPage >= Math.ceil(filteredSamples.length / pageSize)
                                            ? 'opacity-30 cursor-not-allowed'
                                            : 'hover:bg-surface-dim hover:text-foreground'
                                    )}
                                >
                                    <ChevronRight className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    )}
                </>
            ) : tab === 'models' ? (
                <div className="space-y-6">
                    <TrainedModelsTable
                        runs={trainingRuns}
                        loading={isLoadingRuns}
                        progress={trainingProgress}
                        onDelete={removeTrainingRun}
                    />
                    <PipelinesSection
                        pipelines={pipelines}
                        trainingRuns={trainingRuns}
                        loading={isLoadingPipelines}
                        onDelete={handleDeletePipeline}
                        onEdit={handleEditPipeline}
                        onCreate={handleCreatePipeline}
                    />
                </div>
            ) : tab === 'detections' ? (
                <DetectionRunsTable
                    runs={detectionRuns}
                    loading={isLoadingDetections}
                    progress={detectionProgress}
                    onDelete={(id) => {
                        removeDetectionRun(id);
                        fetchSamples(true); // Reload to reflect cleared AI labels
                    }}
                    trainingRuns={trainingRuns}
                    pipelines={pipelines}
                />
            ) : tab === 'test' ? (
                <TestModelTab trainingRuns={trainingRuns} pipelines={pipelines} />
            ) : (
                <div className="space-y-6">
                    <TagsSection
                        tags={tags}
                        loading={isLoadingTags}
                        onDelete={handleDeleteTag}
                        onCreate={() => setShowTagModal(true)}
                    />
                </div>
            )}

            {/* Detection Modal */}
            <RunDetectionModal
                isOpen={showDetectionModal}
                onClose={() => setShowDetectionModal(false)}
                onStart={handleStartDetection}
                trainingRuns={trainingRuns}
                pipelines={pipelines}
                isLoading={isRunningDetection}
            />

            {/* Pipeline Modal */}
            <CreatePipelineModal
                isOpen={showPipelineModal}
                onClose={() => {
                    setShowPipelineModal(false);
                    setEditingPipeline(null);
                }}
                onSave={handleSavePipeline}
                trainingRuns={trainingRuns}
                editingPipeline={editingPipeline}
            />

            {/* Tag Modal */}
            <CreateTagModal
                isOpen={showTagModal}
                onClose={() => setShowTagModal(false)}
                onSave={handleCreateTag}
            />

            {/* Delete Confirmation Modal */}
            <ConfirmModal
                isOpen={showDeleteModal}
                onClose={() => setShowDeleteModal(false)}
                onConfirm={handleDeleteSelected}
                title="Delete Images"
                message={`Are you sure you want to delete ${selectedIds.size} image${selectedIds.size > 1 ? 's' : ''}? This will also delete all associated labels and detections. This action cannot be undone.`}
                confirmText="Delete"
                cancelText="Cancel"
                loading={isDeleting}
                variant="danger"
            />
        </div>
    );
};
