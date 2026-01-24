/**
 * MLAdminPage - Admin interface for training data labeling
 */

import { useState, useEffect, useCallback } from 'react';
import { toast } from 'react-hot-toast';
import {
    TrainingSample,
    TrainingSampleUpdate,
    AdminStats,
    SampleStatus,
    BoardCell,
} from '@blokus-ai/shared';
import { ADMIN_API } from '../../api/admin';
import {
    SampleGrid,
    Pagination,
    StatsPanel,
    TrainingPanel,
    UploadDropzone,
    BoardCreator,
    SampleLabelingSection,
} from '../../components/admin';
import { Button } from '../../components/ui/Button';

type FilterStatus = SampleStatus | 'all';

export const MLAdminPage = () => {
    // Data state
    const [samples, setSamples] = useState<TrainingSample[]>([]);
    const [stats, setStats] = useState<AdminStats | null>(null);
    const [selectedSample, setSelectedSample] = useState<TrainingSample | null>(null);

    // UI state
    const [isLoading, setIsLoading] = useState(true);
    const [isUploading, setIsUploading] = useState(false);
    const [filterStatus, setFilterStatus] = useState<FilterStatus>('all');
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [showBoardCreator, setShowBoardCreator] = useState(false);

    const pageSize = 20;

    // Load samples and stats
    const loadData = useCallback(async () => {
        try {
            const [samplesResponse, statsResponse] = await Promise.all([
                ADMIN_API.listSamples({
                    status: filterStatus === 'all' ? undefined : filterStatus,
                    page: currentPage,
                    page_size: pageSize,
                }),
                ADMIN_API.getStats(),
            ]);

            setSamples(samplesResponse.samples);
            setTotalPages(samplesResponse.total_pages);
            setStats(statsResponse);
        } catch (error) {
            toast.error('Failed to load samples');
            console.error('Load data error:', error);
        } finally {
            setIsLoading(false);
        }
    }, [filterStatus, currentPage]);

    useEffect(() => {
        loadData();
    }, [loadData]);

    // Handle file upload
    const handleUpload = async (file: File) => {
        setIsUploading(true);
        try {
            // Convert file to base64
            const base64 = await new Promise<string>((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const result = reader.result as string;
                    // Remove data URL prefix
                    const base64Data = result.split(',')[1];
                    resolve(base64Data);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });

            const newSample = await ADMIN_API.createSample({
                image: base64,
                source: 'upload',
            });

            // Refresh data
            await loadData();

            toast.success('Sample uploaded successfully');

            // Open the new sample for editing
            setSelectedSample(newSample);
        } catch (error) {
            toast.error('Failed to upload sample');
            console.error('Upload error:', error);
        } finally {
            setIsUploading(false);
        }
    };

    // Handle sample update
    const handleUpdateSample = async (id: string, update: TrainingSampleUpdate) => {
        try {
            const updated = await ADMIN_API.updateSample(id, update);

            // Update local state
            setSamples(prev => prev.map(s => s.id === id ? updated : s));
            if (selectedSample?.id === id) {
                setSelectedSample(updated);
            }

            // Refresh stats
            const newStats = await ADMIN_API.getStats();
            setStats(newStats);
        } catch (error) {
            toast.error('Failed to update sample');
            throw error;
        }
    };

    // Handle sample deletion
    const handleDeleteSample = async (id: string) => {
        try {
            await ADMIN_API.deleteSample(id);

            // Update local state
            setSamples(prev => prev.filter(s => s.id !== id));

            // Refresh stats
            const newStats = await ADMIN_API.getStats();
            setStats(newStats);

            toast.success('Sample deleted');
        } catch (error) {
            toast.error('Failed to delete sample');
            throw error;
        }
    };

    // Handle re-prediction
    const handleRepredict = async (id: string) => {
        try {
            const updated = await ADMIN_API.repredictSample(id);

            // Update local state
            setSamples(prev => prev.map(s => s.id === id ? updated : s));
            if (selectedSample?.id === id) {
                setSelectedSample(updated);
            }

            toast.success('Detection updated');
        } catch (error) {
            toast.error('Failed to re-detect');
            throw error;
        }
    };

    // Handle training
    const handleTrain = async (params: { validation_split: number; min_samples: number }) => {
        const result = await ADMIN_API.trainModel(params);
        if (result.success) {
            toast.success('Model trained successfully');
        }
        return result;
    };

    // Handle navigation between samples
    const handleNavigate = (direction: 'prev' | 'next') => {
        if (!selectedSample) {
            return;
        }

        const currentIndex = samples.findIndex(s => s.id === selectedSample.id);
        if (currentIndex === -1) {
            return;
        }

        const newIndex = direction === 'prev' ? currentIndex - 1 : currentIndex + 1;
        if (newIndex >= 0 && newIndex < samples.length) {
            setSelectedSample(samples[newIndex]);
        }
    };

    // Get current sample navigation info
    const currentIndex = selectedSample
        ? samples.findIndex(s => s.id === selectedSample.id)
        : -1;
    const hasNext = currentIndex >= 0 && currentIndex < samples.length - 1;
    const hasPrev = currentIndex > 0;

    // Handle board creator save
    const handleBoardCreatorSave = async (_boardState: BoardCell[][]) => {
        // Create a synthetic sample by generating a placeholder image
        // For now, we'll create a sample and immediately set its labeled state
        toast.error('Synthetic sample creation requires an image. Use the upload feature instead.');
        setShowBoardCreator(false);
    };

    return (
        <div className="min-h-screen bg-background">
            {/* Header */}
            <div className="bg-surface border-b border-border px-4 py-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                        <h1 className="text-xl font-bold text-foreground">ML Training Admin</h1>
                        <p className="text-sm text-muted-foreground mt-1">
                            Upload, label, and train board detection models
                        </p>
                    </div>

                    <div className="flex items-center gap-2">
                        <Button
                            variant="secondary"
                            size="sm"
                            onClick={() => setShowBoardCreator(true)}
                        >
                            Create Board
                        </Button>
                    </div>
                </div>
            </div>

            <div className="p-4 space-y-6">
                {/* Stats panel */}
                <StatsPanel stats={stats} loading={isLoading} />

                {/* Upload and Training row */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {/* Upload dropzone */}
                    <div className="space-y-2">
                        <h3 className="text-sm font-medium text-muted-foreground">Upload Images</h3>
                        <UploadDropzone
                            onUpload={handleUpload}
                            disabled={isUploading}
                        />
                    </div>

                    {/* Training panel */}
                    <TrainingPanel
                        labeledCount={(stats?.labeled_samples || 0) + (stats?.verified_samples || 0)}
                        onTrain={handleTrain}
                        minSamplesRequired={10}
                    />
                </div>

                {/* Filter tabs */}
                <div className="flex items-center gap-2 border-b border-border pb-2">
                    {(['all', 'pending', 'labeled', 'verified'] as const).map((status) => (
                        <button
                            key={status}
                            onClick={() => {
                                setFilterStatus(status);
                                setCurrentPage(1);
                            }}
                            className={`
                                px-3 py-1.5 text-sm rounded-md transition-colors capitalize
                                ${filterStatus === status
                            ? 'bg-primary text-white'
                            : 'bg-surface text-muted-foreground hover:text-foreground'
                        }
                            `}
                        >
                            {status}
                            {status !== 'all' && stats && (
                                <span className="ml-1.5 text-xs opacity-70">
                                    ({status === 'pending'
                                        ? stats.pending_samples
                                        : status === 'labeled'
                                            ? stats.labeled_samples
                                            : stats.verified_samples
                                    })
                                </span>
                            )}
                        </button>
                    ))}

                    <div className="flex-1" />

                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => loadData()}
                        disabled={isLoading}
                    >
                        <svg
                            className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                            />
                        </svg>
                    </Button>
                </div>

                {/* Sample grid */}
                <SampleGrid
                    samples={samples}
                    selectedId={selectedSample?.id || null}
                    onSelect={setSelectedSample}
                    loading={isLoading}
                />

                {/* Pagination */}
                <Pagination
                    currentPage={currentPage}
                    totalPages={totalPages}
                    onPageChange={setCurrentPage}
                />

                {/* Inline labeling section */}
                {selectedSample && (
                    <SampleLabelingSection
                        sample={selectedSample}
                        onUpdate={handleUpdateSample}
                        onDelete={handleDeleteSample}
                        onRepredict={handleRepredict}
                        onClose={() => setSelectedSample(null)}
                        onNavigate={handleNavigate}
                        hasNext={hasNext}
                        hasPrev={hasPrev}
                    />
                )}
            </div>

            {/* Board creator modal */}
            <BoardCreator
                isOpen={showBoardCreator}
                onClose={() => setShowBoardCreator(false)}
                onSave={handleBoardCreatorSave}
            />
        </div>
    );
};
