import { useRef, useState, useEffect, useCallback } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { http, API_URL } from '../../../api/config';
import { ML_API } from '../../../api/ml';
import { Button } from '../../../components/ui/Button';
import type { SampleStatusFlags, SampleStatusDisplay, TrainingRun } from '@blokus-ai/shared';
import { getStatusLabels } from '@blokus-ai/shared';
import { statusColors, statusLabels, isUnlabeled } from '../../../lib/status';
import { TrainedModelsTable, TrainingProgress } from './TrainedModelsTable';
import { useSocketEvent } from '../../../hooks/useSocketEvent';

type FilterStatus = SampleStatusDisplay | 'all';
type Tab = 'images' | 'models';

const validFilters: FilterStatus[] = ['all', 'unlabeled', 'ai-labeled', 'human-labeled', 'verified'];

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

const matchesFilter = (status: SampleStatusFlags, filter: FilterStatus): boolean => {
    if (filter === 'all') {
        return true;
    }
    if (filter === 'unlabeled') {
        return isUnlabeled(status);
    }
    if (filter === 'ai-labeled') {
        return status.aiLabeled;
    }
    if (filter === 'human-labeled') {
        return status.humanLabeled;
    }
    if (filter === 'verified') {
        return status.verified;
    }
    return true;
};

export const ImagesPage = () => {
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [samples, setSamples] = useState<Sample[]>([]);
    const [isTraining, setIsTraining] = useState(false);
    const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
    const [isLoadingRuns, setIsLoadingRuns] = useState(false);
    const [trainingProgress, setTrainingProgress] = useState<Record<string, TrainingProgress>>({});

    // Get tab from URL or default to 'images'
    const urlTab = searchParams.get('tab') as Tab | null;
    const tab: Tab = urlTab === 'models' ? 'models' : 'images';

    const setTab = (newTab: Tab) => {
        if (newTab === 'images') {
            searchParams.delete('tab');
        } else {
            searchParams.set('tab', newTab);
        }
        // Clear status filter when switching tabs
        searchParams.delete('status');
        setSearchParams(searchParams);
    };

    // Get filter from URL or default to 'all'
    const urlFilter = searchParams.get('status') as FilterStatus | null;
    const filter: FilterStatus = urlFilter && validFilters.includes(urlFilter) ? urlFilter : 'all';

    const setFilter = (newFilter: FilterStatus) => {
        if (newFilter === 'all') {
            searchParams.delete('status');
        } else {
            searchParams.set('status', newFilter);
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

    useEffect(() => {
        loadSamples();
    }, [loadSamples]);

    useEffect(() => {
        if (tab === 'models') {
            loadTrainingRuns();
        }
    }, [tab, loadTrainingRuns]);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) {
            return;
        }

        setIsUploading(true);

        try {
            // Get presigned URL from API
            const presign = await http.post<PresignResponse>('/api/ml/upload/presign', {});

            // Upload directly to S3
            const response = await fetch(presign.uploadUrl, {
                method: 'PUT',
                body: file,
                headers: {
                    'Content-Type': 'image/png',
                },
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            // Generate thumbnail
            await http.post('/api/ml/upload/complete', { id: presign.id });

            toast.success('Image uploaded');
            loadSamples();
        } catch {
            toast.error('Upload failed');
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
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

    // Listen for training events via WebSocket
    useSocketEvent('training-run-progress', (data) => {
        setTrainingProgress(prev => ({
            ...prev,
            [data.runId]: {
                epoch: data.epoch,
                totalEpochs: data.totalEpochs,
                loss: data.loss,
                accuracy: data.accuracy,
            },
        }));
    }, []);

    useSocketEvent('training-run-completed', (data) => {
        setTrainingRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setTrainingProgress(prev => {
            const { [data.run.id]: _, ...rest } = prev;
            return rest;
        });
        setIsTraining(false);
        toast.success('Training completed successfully');
    }, []);

    useSocketEvent('training-run-failed', (data) => {
        setTrainingRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setTrainingProgress(prev => {
            const { [data.run.id]: _, ...rest } = prev;
            return rest;
        });
        setIsTraining(false);
        toast.error(data.run.errorMessage || 'Training failed');
    }, []);

    const filteredSamples = samples.filter(s => matchesFilter(s.status, filter));

    const counts = {
        all: samples.length,
        unlabeled: samples.filter(s => isUnlabeled(s.status)).length,
        'ai-labeled': samples.filter(s => s.status.aiLabeled).length,
        'human-labeled': samples.filter(s => s.status.humanLabeled).length,
        verified: samples.filter(s => s.status.verified).length,
    };

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
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="flex items-center gap-3">
                    <Link
                        to="/admin"
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <BackIcon />
                    </Link>
                    <h1 className="text-xl font-medium text-foreground">Training</h1>
                </div>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                />
                <div className="flex gap-2">
                    <Button
                        onClick={() => navigate('/admin/images/label')}
                        variant="secondary"
                        leftIcon={<TagIcon />}
                    >
                        Data labeling mode
                    </Button>
                    <Button
                        onClick={handleTrain}
                        loading={isTraining}
                        leftIcon={<TrainIcon />}
                    >
                        Train Model
                    </Button>
                    {tab === 'images' && (
                        <Button
                            onClick={handleUploadClick}
                            loading={isUploading}
                            leftIcon={<PlusIcon />}
                        >
                            Upload
                        </Button>
                    )}
                </div>
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
            </div>

            {tab === 'images' ? (
                <>
                    {/* Filter Tabs */}
                    <div className="flex flex-wrap gap-1">
                        {validFilters.map((filterOption) => (
                            <button
                                key={filterOption}
                                onClick={() => setFilter(filterOption)}
                                className={`
                                    px-2.5 py-1 text-xs font-medium rounded-full transition-colors
                                    ${filter === filterOption
                                ? 'bg-primary text-white'
                                : 'bg-surface-dim text-secondary hover:text-foreground'
                            }
                                `}
                            >
                                {filterOption === 'all' ? 'All' : statusLabels[filterOption as SampleStatusDisplay]} {counts[filterOption] > 0 && `(${counts[filterOption]})`}
                            </button>
                        ))}
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

                                return (
                                    <button
                                        key={sample.id}
                                        onClick={() => navigate(`/admin/images/${sample.id}`)}
                                        className="bg-surface rounded-lg border border-border overflow-hidden text-left transition-shadow hover:shadow-gcp-hover focus:outline-none focus:ring-2 focus:ring-primary/30"
                                    >
                                        <div className="aspect-square bg-surface-dim">
                                            <img
                                                src={sample.thumbnailUrl || sample.imageUrl || `${API_URL}/api/ml/samples/${sample.id}/image`}
                                                alt={`Sample ${sample.id}`}
                                                className="w-full h-full object-cover"
                                                loading="lazy"
                                            />
                                        </div>
                                        <div className="p-2">
                                            <div className="flex items-center justify-between gap-2">
                                                <span className="text-xs text-muted truncate font-mono">
                                                    {sample.id.slice(0, 8)}
                                                </span>
                                                <div className="flex gap-0.5">
                                                    {allLabels.map(label => (
                                                        <span
                                                            key={label}
                                                            className={`px-1.5 py-0.5 text-[10px] font-medium rounded-full ${statusColors[label]}`}
                                                        >
                                                            {statusLabels[label]}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                            <p className="text-[11px] text-secondary mt-0.5">
                                                {formatDate(sample.created_at)}
                                            </p>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    )}
                </>
            ) : (
                <TrainedModelsTable runs={trainingRuns} loading={isLoadingRuns} progress={trainingProgress} />
            )}
        </div>
    );
};
