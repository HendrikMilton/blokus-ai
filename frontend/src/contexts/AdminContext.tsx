import { useState, useCallback, useEffect } from 'react';
import { makeContext } from './config';
import { http } from '../api/config';
import { ML_API } from '../api/ml';
import { useSocketEvent } from '../hooks/useSocketEvent';
import type { TrainingRun, DetectionRun, Sample, ModelPipeline, Tag, CornerDetectionRun } from '@blokus-ai/shared';

export type { Sample } from '@blokus-ai/shared';

interface SamplesResponse {
    samples: Sample[];
}

// Progress tracking types (must match TrainedModelsTable and DetectionRunsTable)
export interface TrainingProgress {
    phase: 'detection' | 'classifier' | 'pose';
    epoch: number;
    totalEpochs: number;
    loss: number;
    accuracy: number;
    detectionAccuracy?: number;
    detectionLoss?: number;
}

export interface DetectionProgress {
    processedCount: number;
    totalCount: number;
    successCount: number;
    failedCount: number;
}

export interface CornerDetectionProgress {
    processedCount: number;
    totalCount: number;
    successCount: number;
    failedCount: number;
}

export const [AdminProvider, useAdmin] = makeContext(() => {
    // Samples state
    const [samples, setSamples] = useState<Sample[]>([]);
    const [isLoadingSamples, setIsLoadingSamples] = useState(true);
    const [hasFetchedSamples, setHasFetchedSamples] = useState(false);

    // Selected models for viewing detections (empty = latest/all)
    const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);

    // Training runs state
    const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
    const [isLoadingTrainingRuns, setIsLoadingTrainingRuns] = useState(false);
    const [hasFetchedTrainingRuns, setHasFetchedTrainingRuns] = useState(false);
    const [trainingProgress, setTrainingProgress] = useState<Record<string, TrainingProgress>>({});

    // Detection runs state
    const [detectionRuns, setDetectionRuns] = useState<DetectionRun[]>([]);
    const [isLoadingDetectionRuns, setIsLoadingDetectionRuns] = useState(false);
    const [hasFetchedDetectionRuns, setHasFetchedDetectionRuns] = useState(false);
    const [detectionProgress, setDetectionProgress] = useState<Record<string, DetectionProgress>>({});

    // Pipelines state
    const [pipelines, setPipelines] = useState<ModelPipeline[]>([]);
    const [isLoadingPipelines, setIsLoadingPipelines] = useState(false);
    const [hasFetchedPipelines, setHasFetchedPipelines] = useState(false);

    // Tags state
    const [tags, setTags] = useState<Tag[]>([]);
    const [isLoadingTags, setIsLoadingTags] = useState(false);
    const [hasFetchedTags, setHasFetchedTags] = useState(false);

    // Corner detection runs state
    const [cornerRuns, setCornerRuns] = useState<CornerDetectionRun[]>([]);
    const [isLoadingCornerRuns, setIsLoadingCornerRuns] = useState(false);
    const [hasFetchedCornerRuns, setHasFetchedCornerRuns] = useState(false);
    const [cornerProgress, setCornerProgress] = useState<Record<string, CornerDetectionProgress>>({});

    // Fetch samples with optional model and tag filters
    const fetchSamples = useCallback(async (forceFetch = false, modelId?: string | null, tagId?: string | null) => {
        if (hasFetchedSamples && !forceFetch) {
            return;
        }

        setIsLoadingSamples(true);
        try {
            const params = new URLSearchParams();
            if (modelId) {
                params.set('modelId', modelId);
            }
            if (tagId) {
                params.set('tagId', tagId);
            }
            const queryString = params.toString();
            const url = queryString ? `/api/ml/samples?${queryString}` : '/api/ml/samples';
            const data = await http.get<SamplesResponse>(url);
            setSamples(data.samples || []);
            setHasFetchedSamples(true);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoadingSamples(false);
        }
    }, [hasFetchedSamples]);

    // Fetch training runs
    const fetchTrainingRuns = useCallback(async (forceFetch = false) => {
        if (hasFetchedTrainingRuns && !forceFetch) {
            return;
        }

        setIsLoadingTrainingRuns(true);
        try {
            const runs = await ML_API.getTrainingRuns();
            setTrainingRuns(runs);
            setHasFetchedTrainingRuns(true);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoadingTrainingRuns(false);
        }
    }, [hasFetchedTrainingRuns]);

    // Fetch detection runs
    const fetchDetectionRuns = useCallback(async (forceFetch = false) => {
        if (hasFetchedDetectionRuns && !forceFetch) {
            return;
        }

        setIsLoadingDetectionRuns(true);
        try {
            const runs = await ML_API.getDetectionRuns();
            setDetectionRuns(runs);
            setHasFetchedDetectionRuns(true);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoadingDetectionRuns(false);
        }
    }, [hasFetchedDetectionRuns]);

    // Fetch pipelines
    const fetchPipelines = useCallback(async (forceFetch = false) => {
        if (hasFetchedPipelines && !forceFetch) {
            return;
        }

        setIsLoadingPipelines(true);
        try {
            const pipelinesData = await ML_API.getPipelines();
            setPipelines(pipelinesData);
            setHasFetchedPipelines(true);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoadingPipelines(false);
        }
    }, [hasFetchedPipelines]);

    // Fetch tags
    const fetchTags = useCallback(async (forceFetch = false) => {
        if (hasFetchedTags && !forceFetch) {
            return;
        }

        setIsLoadingTags(true);
        try {
            const tagsData = await ML_API.getTags();
            setTags(tagsData);
            setHasFetchedTags(true);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoadingTags(false);
        }
    }, [hasFetchedTags]);

    // Fetch corner detection runs
    const fetchCornerRuns = useCallback(async (forceFetch = false) => {
        if (hasFetchedCornerRuns && !forceFetch) {
            return;
        }

        setIsLoadingCornerRuns(true);
        try {
            const runs = await ML_API.getCornerRuns();
            setCornerRuns(runs);
            setHasFetchedCornerRuns(true);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoadingCornerRuns(false);
        }
    }, [hasFetchedCornerRuns]);

    // Load all data on mount
    useEffect(() => {
        fetchSamples();
        fetchTrainingRuns();
        fetchDetectionRuns();
        fetchPipelines();
        fetchTags();
        fetchCornerRuns();
    }, [fetchSamples, fetchTrainingRuns, fetchDetectionRuns, fetchPipelines, fetchTags, fetchCornerRuns]);

    // Socket listeners for training runs
    useSocketEvent('training-run-started', (data) => {
        setTrainingRuns(prev => [data.run, ...prev.filter(r => r.id !== data.run.id)]);
    }, []);

    useSocketEvent('training-run-progress', (data) => {
        setTrainingProgress(prev => {
            const existing = prev[data.runId];
            return {
                ...prev,
                [data.runId]: {
                    // Preserve detection metrics when classifier/pose phase starts
                    detectionAccuracy: data.phase === 'detection' ? data.accuracy : existing?.detectionAccuracy,
                    detectionLoss: data.phase === 'detection' ? data.loss : existing?.detectionLoss,
                    // Preserve classifier metrics when pose phase starts
                    classifierAccuracy: data.phase === 'classifier' ? data.accuracy : existing?.classifierAccuracy,
                    classifierLoss: data.phase === 'classifier' ? data.loss : existing?.classifierLoss,
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
            const { [data.run.id]: _, ...rest } = prev;
            return rest;
        });
    }, []);

    useSocketEvent('training-run-failed', (data) => {
        setTrainingRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setTrainingProgress(prev => {
            const { [data.run.id]: _, ...rest } = prev;
            return rest;
        });
    }, []);

    useSocketEvent('training-run-deleted', (data) => {
        setTrainingRuns(prev => prev.filter(r => r.id !== data.runId));
    }, []);

    // Socket listeners for detection runs
    useSocketEvent('detection-run-started', (data) => {
        setDetectionRuns(prev => [data.run, ...prev.filter(r => r.id !== data.run.id)]);
    }, []);

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
            const { [data.run.id]: _, ...rest } = prev;
            return rest;
        });
        // Refresh samples to get updated detection results for the current model(s)
        const modelId = selectedModelIds.length === 1 ? selectedModelIds[0] : null;
        fetchSamples(true, modelId);
    }, [fetchSamples, selectedModelIds]);

    useSocketEvent('detection-run-failed', (data) => {
        setDetectionRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setDetectionProgress(prev => {
            const { [data.run.id]: _, ...rest } = prev;
            return rest;
        });
    }, []);

    // Socket listeners for corner detection runs
    useSocketEvent('corner-run-started', (data) => {
        setCornerRuns(prev => [data.run, ...prev.filter(r => r.id !== data.run.id)]);
    }, []);

    useSocketEvent('corner-run-progress', (data) => {
        setCornerProgress(prev => ({
            ...prev,
            [data.runId]: {
                processedCount: data.processedCount,
                totalCount: data.totalCount,
                successCount: data.successCount,
                failedCount: data.failedCount,
            },
        }));
    }, []);

    useSocketEvent('corner-run-completed', (data) => {
        setCornerRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setCornerProgress(prev => {
            const { [data.run.id]: _, ...rest } = prev;
            return rest;
        });
    }, []);

    useSocketEvent('corner-run-failed', (data) => {
        setCornerRuns(prev => prev.map(r => r.id === data.run.id ? data.run : r));
        setCornerProgress(prev => {
            const { [data.run.id]: _, ...rest } = prev;
            return rest;
        });
    }, []);

    // Helper functions for modifying state
    const addSample = useCallback((sample: Sample) => {
        setSamples(prev => [sample, ...prev]);
    }, []);

    const updateSample = useCallback((id: string, updates: Partial<Sample>) => {
        setSamples(prev => prev.map(s => s.id === id ? { ...s, ...updates } : s));
    }, []);

    const removeSample = useCallback((id: string) => {
        setSamples(prev => prev.filter(s => s.id !== id));
    }, []);

    const removeSamples = useCallback((ids: string[]) => {
        const idSet = new Set(ids);
        setSamples(prev => prev.filter(s => !idSet.has(s.id)));
    }, []);

    const addTrainingRun = useCallback((run: TrainingRun) => {
        setTrainingRuns(prev => [run, ...prev]);
    }, []);

    const removeTrainingRun = useCallback((id: string) => {
        setTrainingRuns(prev => prev.filter(r => r.id !== id));
    }, []);

    const addDetectionRun = useCallback((run: DetectionRun) => {
        setDetectionRuns(prev => [run, ...prev]);
    }, []);

    const removeDetectionRun = useCallback((id: string) => {
        setDetectionRuns(prev => prev.filter(r => r.id !== id));
    }, []);

    const addPipeline = useCallback((pipeline: ModelPipeline) => {
        setPipelines(prev => [pipeline, ...prev]);
    }, []);

    const updatePipelineState = useCallback((id: string, updates: Partial<ModelPipeline>) => {
        setPipelines(prev => prev.map(p => p.id === id ? { ...p, ...updates } : p));
    }, []);

    const removePipeline = useCallback((id: string) => {
        setPipelines(prev => prev.filter(p => p.id !== id));
    }, []);

    const addTag = useCallback((tag: Tag) => {
        setTags(prev => [...prev, tag].sort((a, b) => a.name.localeCompare(b.name)));
    }, []);

    const removeTag = useCallback((id: string) => {
        setTags(prev => prev.filter(t => t.id !== id));
    }, []);

    const addCornerRun = useCallback((run: CornerDetectionRun) => {
        setCornerRuns(prev => [run, ...prev]);
    }, []);

    const removeCornerRun = useCallback((id: string) => {
        setCornerRuns(prev => prev.filter(r => r.id !== id));
    }, []);

    return {
        // Samples
        samples,
        isLoadingSamples,
        fetchSamples,
        addSample,
        updateSample,
        removeSample,
        removeSamples,

        // Selected models for viewing detections
        selectedModelIds,
        setSelectedModelIds,

        // Training runs
        trainingRuns,
        isLoadingTrainingRuns,
        fetchTrainingRuns,
        trainingProgress,
        addTrainingRun,
        removeTrainingRun,

        // Detection runs
        detectionRuns,
        isLoadingDetectionRuns,
        fetchDetectionRuns,
        detectionProgress,
        addDetectionRun,
        removeDetectionRun,

        // Pipelines
        pipelines,
        isLoadingPipelines,
        fetchPipelines,
        addPipeline,
        updatePipelineState,
        removePipeline,

        // Tags
        tags,
        isLoadingTags,
        fetchTags,
        addTag,
        removeTag,

        // Corner detection runs
        cornerRuns,
        isLoadingCornerRuns,
        fetchCornerRuns,
        cornerProgress,
        addCornerRun,
        removeCornerRun,
    };
});
