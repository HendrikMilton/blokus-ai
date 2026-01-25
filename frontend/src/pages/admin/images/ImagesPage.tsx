import { useRef, useState, useEffect, useCallback } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { http, API_URL } from '../../../api/config';
import { ML_API } from '../../../api/ml';
import { Button } from '../../../components/ui/Button';
import { Select } from '../../../components/ui/Select';
import type { SampleStatusFlags, TrainingRun, DetectionRun } from '@blokus-ai/shared';
import { getStatusLabels } from '@blokus-ai/shared';
import { statusColors, statusLabels } from '../../../lib/status';
import { TrainedModelsTable, TrainingProgress } from './TrainedModelsTable';
import { DetectionRunsTable, DetectionProgress } from './DetectionRunsTable';
import { RunDetectionModal } from './RunDetectionModal';
import { TestModelTab } from './TestModelTab';
import { ConfirmModal } from '../../../components/ui/ConfirmModal';
import { useSocketEvent } from '../../../hooks/useSocketEvent';

type Tab = 'images' | 'models' | 'detections' | 'test';
type FilterPreset = 'all' | 'unlabeled' | 'false-positives' | 'false-negatives' | 'low-iou' | 'disagreements';
type SortOption = 'default' | 'lowest-iou' | 'lowest-confidence';

const filterPresetLabels: Record<FilterPreset, string> = {
    'all': 'All',
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

interface Sample {
    id: string;
    filename: string;
    status: SampleStatusFlags;
    ai: {
        hasBoardDetected: boolean;
        isEntireBoardVisible: boolean;
        confidence: number;
        classifierConfidence: number | null;
    } | null;
    human: {
        hasBoardDetected: boolean;
        isEntireBoardVisible: boolean;
    } | null;
    iou: number | null;
    created_at: string;
    imageUrl?: string;
    thumbnailUrl?: string;
}

interface SamplesResponse {
    samples: Sample[];
}

interface PresignResponse {
    id: string;
    uploadUrl: string;
    key: string;
    imageUrl: string;
}

const PlusIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
);

const BackIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
    </svg>
);

const TagIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
    </svg>
);

const TrainIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
);

const TrashIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
);

const CheckIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
);

const DetectIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
);

const matchesPreset = (sample: Sample, preset: FilterPreset): boolean => {
    switch (preset) {
        case 'all':
            return true;
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
                if (a.iou === null && b.iou === null) return 0;
                if (a.iou === null) return 1;
                if (b.iou === null) return -1;
                return a.iou - b.iou;
            });
        case 'lowest-confidence':
            return sorted.sort((a, b) => {
                const aConf = a.ai?.confidence ?? null;
                const bConf = b.ai?.confidence ?? null;
                // Nulls and 0% go to the end
                const aIsEmpty = aConf === null || aConf === 0;
                const bIsEmpty = bConf === null || bConf === 0;
                if (aIsEmpty && bIsEmpty) return 0;
                if (aIsEmpty) return 1;
                if (bIsEmpty) return -1;
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
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState({ completed: 0, total: 0 });
    const [isLoading, setIsLoading] = useState(true);
    const [samples, setSamples] = useState<Sample[]>([]);
    const [isTraining, setIsTraining] = useState(false);
    const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
    const [isLoadingRuns, setIsLoadingRuns] = useState(false);
    const [trainingProgress, setTrainingProgress] = useState<Record<string, TrainingProgress>>({});

    // Detection runs state
    const [detectionRuns, setDetectionRuns] = useState<DetectionRun[]>([]);
    const [isLoadingDetections, setIsLoadingDetections] = useState(false);
    const [detectionProgress, setDetectionProgress] = useState<Record<string, DetectionProgress>>({});
    const [isRunningDetection, setIsRunningDetection] = useState(false);
    const [showDetectionModal, setShowDetectionModal] = useState(false);

    // Selection state
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [isDeleting, setIsDeleting] = useState(false);
    const [showDeleteModal, setShowDeleteModal] = useState(false);

    // Get tab from URL or default to 'images'
    const urlTab = searchParams.get('tab') as Tab | null;
    const tab: Tab = urlTab === 'models' ? 'models' : urlTab === 'detections' ? 'detections' : urlTab === 'test' ? 'test' : 'images';

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
    const validPresets: FilterPreset[] = ['all', 'unlabeled', 'false-positives', 'false-negatives', 'low-iou', 'disagreements'];
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

    const loadSamples = useCallback(async () => {
        try {
            const data = await http.get<SamplesResponse>('/api/ml/samples');
            setSamples(data.samples || []);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoading(false);
        }
    }, []);

    const loadTrainingRuns = useCallback(async () => {
        setIsLoadingRuns(true);
        try {
            const runs = await ML_API.getTrainingRuns();
            setTrainingRuns(runs);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoadingRuns(false);
        }
    }, []);

    const loadDetectionRuns = useCallback(async () => {
        setIsLoadingDetections(true);
        try {
            const runs = await ML_API.getDetectionRuns();
            setDetectionRuns(runs);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoadingDetections(false);
        }
    }, []);

    useEffect(() => {
        loadSamples();
    }, [loadSamples]);

    useEffect(() => {
        if (tab === 'models') {
            loadTrainingRuns();
        } else if (tab === 'detections') {
            loadDetectionRuns();
            // Also load training runs for the modal dropdown
            if (trainingRuns.length === 0) {
                loadTrainingRuns();
            }
        } else if (tab === 'test') {
            // Load training runs for model selection
            if (trainingRuns.length === 0) {
                loadTrainingRuns();
            }
        }
    }, [tab, loadTrainingRuns, loadDetectionRuns, trainingRuns.length]);

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
                const presign = await http.post<PresignResponse>('/api/ml/upload/presign', {});

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
                await http.post('/api/ml/upload/complete', { id: presign.id });

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
            loadSamples();
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
        setIsTraining(true);
        try {
            const run = await ML_API.startTraining();
            // Add the new run to the list immediately
            setTrainingRuns(prev => [run, ...prev]);
            toast.success('Training started');
        } catch {
            toast.error('Failed to start training');
            setIsTraining(false);
        }
    };

    const handleStartDetection = async (modelId: string, includeExistingAiLabeled: boolean) => {
        setIsRunningDetection(true);
        try {
            const run = await ML_API.startBulkDetection(modelId, includeExistingAiLabeled);
            // Add the new run to the list immediately
            setDetectionRuns(prev => [run, ...prev]);
            setShowDetectionModal(false);
            toast.success('Detection started');
        } catch {
            toast.error('Failed to start detection');
        } finally {
            setIsRunningDetection(false);
        }
    };

    const toggleSelect = (id: string) => {
        setSelectedIds(prev => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            } else {
                next.add(id);
            }
            return next;
        });
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
            await http.delete('/api/ml/samples', { ids: Array.from(selectedIds) });
            toast.success(`Deleted ${selectedIds.size} image${selectedIds.size > 1 ? 's' : ''}`);
            setSamples(prev => prev.filter(s => !selectedIds.has(s.id)));
            setSelectedIds(new Set());
            setShowDeleteModal(false);
        } catch {
            toast.error('Failed to delete images');
        } finally {
            setIsDeleting(false);
        }
    };

    // Listen for training events via WebSocket
    useSocketEvent('training-run-progress', (data) => {
        console.log('[DEBUG] Received training-run-progress event:', data);
        setTrainingProgress(prev => {
            const existing = prev[data.runId];
            return {
                ...prev,
                [data.runId]: {
                    // Preserve detection metrics when classifier phase starts
                    detectionAccuracy: data.phase === 'detection' ? data.accuracy : existing?.detectionAccuracy,
                    detectionLoss: data.phase === 'detection' ? data.loss : existing?.detectionLoss,
                    // Current phase info
                    phase: data.phase,
                    epoch: data.epoch,
                    totalEpochs: data.totalEpochs,
                    loss: data.loss,
                    accuracy: data.accuracy,
                },
            };
        });
    }, []);

    useSocketEvent('training-run-completed', (data) => {
        setTrainingRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setTrainingProgress(prev => {
            const { [data.run.id]: _removed, ...rest } = prev;
            void _removed;
            return rest;
        });
        setIsTraining(false);
        toast.success('Training completed successfully');
    }, []);

    useSocketEvent('training-run-failed', (data) => {
        setTrainingRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setTrainingProgress(prev => {
            const { [data.run.id]: _removed, ...rest } = prev;
            void _removed;
            return rest;
        });
        setIsTraining(false);
        toast.error(data.run.errorMessage || 'Training failed');
    }, []);

    useSocketEvent('training-run-deleted', (data) => {
        setTrainingRuns(prev => prev.filter(r => r.id !== data.runId));
    }, []);

    // Listen for detection events via WebSocket
    useSocketEvent('detection-run-progress', (data) => {
        setDetectionProgress(prev => ({
            ...prev,
            [data.runId]: {
                processedCount: data.processedCount,
                totalCount: data.totalCount,
                successCount: data.successCount,
                failedCount: data.failedCount,
            },
        }));
    }, []);

    useSocketEvent('detection-run-completed', (data) => {
        setDetectionRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setDetectionProgress(prev => {
            const { [data.run.id]: _removed, ...rest } = prev;
            void _removed;
            return rest;
        });
        toast.success('Detection completed successfully');
    }, []);

    useSocketEvent('detection-run-failed', (data) => {
        setDetectionRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setDetectionProgress(prev => {
            const { [data.run.id]: _removed, ...rest } = prev;
            void _removed;
            return rest;
        });
        toast.error(data.run.errorMessage || 'Detection failed');
    }, []);

    useSocketEvent('detection-run-deleted', (data) => {
        setDetectionRuns(prev => prev.filter(r => r.id !== data.runId));
        // Reload samples to reflect cleared AI labels
        loadSamples();
    }, [loadSamples]);

    const filteredSamples = sortSamples(
        samples.filter(s => matchesPreset(s, preset)),
        sort
    );

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleDateString(undefined, {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-3">
                <Link
                    to="/admin"
                    className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                >
                    <BackIcon />
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
                    leftIcon={<TagIcon />}
                    size="sm"
                    className="justify-center"
                >
                    <span className="hidden sm:inline">Data labeling</span>
                    <span className="sm:hidden">Label</span>
                </Button>
                <Button
                    onClick={() => setShowDetectionModal(true)}
                    variant="secondary"
                    leftIcon={<DetectIcon />}
                    size="sm"
                    className="justify-center"
                >
                    <span className="hidden sm:inline">Run AI Detection</span>
                    <span className="sm:hidden">Detect</span>
                </Button>
                <Button
                    onClick={handleTrain}
                    loading={isTraining}
                    leftIcon={<TrainIcon />}
                    size="sm"
                    className="justify-center"
                >
                    Train
                </Button>
                {tab === 'images' && (
                    <Button
                        onClick={handleUploadClick}
                        loading={isUploading}
                        leftIcon={<PlusIcon />}
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
            <div className="flex gap-1 border-b border-border">
                <button
                    onClick={() => setTab('images')}
                    className={`
                        px-4 py-2 text-sm font-medium transition-colors -mb-px
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
                        px-4 py-2 text-sm font-medium transition-colors -mb-px
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
                        px-4 py-2 text-sm font-medium transition-colors -mb-px
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
                        px-4 py-2 text-sm font-medium transition-colors -mb-px
                        ${tab === 'test'
            ? 'border-b-2 border-primary text-foreground'
            : 'text-secondary hover:text-foreground'
        }
                    `}
                >
                    Test Model
                </button>
            </div>

            {tab === 'images' ? (
                <>
                    {/* Selection bar */}
                    {selectedIds.size > 0 && (
                        <div className="flex items-center justify-between gap-2 p-2 bg-primary/10 border border-primary/20 rounded-lg">
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
                                leftIcon={<TrashIcon />}
                            >
                                Delete
                            </Button>
                        </div>
                    )}

                    {/* Filter and Sort Dropdowns */}
                    <div className="flex flex-wrap items-center gap-3">
                        <Select
                            label="Filter"
                            value={preset}
                            onChange={(v) => setPreset(v as FilterPreset)}
                            options={Object.entries(filterPresetLabels).map(([value, label]) => ({ value, label }))}
                            className="w-40"
                        />
                        <Select
                            label="Sort"
                            value={sort}
                            onChange={(v) => setSort(v as SortOption)}
                            options={Object.entries(sortLabels).map(([value, label]) => ({ value, label }))}
                            className="w-40"
                        />
                        <span className="text-xs text-muted self-end pb-2">
                            {filteredSamples.length} of {samples.length} images
                        </span>
                    </div>

                    {/* Grid */}
                    {isLoading ? (
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
                    ) : filteredSamples.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-16 text-center">
                            <div className="w-16 h-16 bg-surface-dim rounded-full flex items-center justify-center mb-4">
                                <svg className="w-8 h-8 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                            </div>
                            <p className="text-secondary font-medium">No images found</p>
                            <p className="text-sm text-muted mt-1">Upload board images to get started</p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                            {filteredSamples.map((sample) => {
                                const allLabels = getStatusLabels(sample.status);
                                const isSelected = selectedIds.has(sample.id);

                                return (
                                    <div
                                        key={sample.id}
                                        className={`relative bg-surface rounded-lg border overflow-hidden text-left transition-all ${
                                            isSelected
                                                ? 'border-primary ring-2 ring-primary/30'
                                                : 'border-border hover:shadow-gcp-hover'
                                        }`}
                                    >
                                        {/* Selection checkbox */}
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                toggleSelect(sample.id);
                                            }}
                                            className={`absolute top-2 left-2 z-10 w-6 h-6 rounded-md border-2 flex items-center justify-center transition-all ${
                                                isSelected
                                                    ? 'bg-primary border-primary text-white'
                                                    : 'bg-black/40 border-white/60 text-transparent hover:border-white hover:bg-black/60'
                                            }`}
                                        >
                                            <CheckIcon />
                                        </button>

                                        {/* Clickable card content */}
                                        <button
                                            onClick={() => navigate(`/admin/images/${sample.id}`)}
                                            className="w-full text-left focus:outline-none"
                                        >
                                            <img
                                                    src={sample.thumbnailUrl || sample.imageUrl || `${API_URL}/api/ml/samples/${sample.id}/image`}
                                                    alt={`Sample ${sample.id}`}
                                                    className="w-full h-auto"
                                                    loading="lazy"
                                                />
                                            <div className="px-2 py-1.5 text-xs">
                                                {/* Header: ID + badges */}
                                                <div className="flex items-center justify-between gap-1 mb-1">
                                                    <span className="text-muted font-mono">{sample.id.slice(0, 8)}</span>
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
                                            </div>
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </>
            ) : tab === 'models' ? (
                <TrainedModelsTable
                    runs={trainingRuns}
                    loading={isLoadingRuns}
                    progress={trainingProgress}
                    onDelete={(id) => setTrainingRuns(prev => prev.filter(r => r.id !== id))}
                />
            ) : tab === 'detections' ? (
                <DetectionRunsTable
                    runs={detectionRuns}
                    loading={isLoadingDetections}
                    progress={detectionProgress}
                    onDelete={(id) => {
                        setDetectionRuns(prev => prev.filter(r => r.id !== id));
                        loadSamples(); // Reload to reflect cleared AI labels
                    }}
                />
            ) : (
                <TestModelTab trainingRuns={trainingRuns} />
            )}

            {/* Detection Modal */}
            <RunDetectionModal
                isOpen={showDetectionModal}
                onClose={() => setShowDetectionModal(false)}
                onStart={handleStartDetection}
                trainingRuns={trainingRuns}
                isLoading={isRunningDetection}
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
