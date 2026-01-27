import { useState, useCallback, useEffect } from 'react';
import { makeContext } from './config';
import { http } from '../api/config';
import { ML_API } from '../api/ml';
import { useSocketEvent } from '../hooks/useSocketEvent';
import type { TrainingRun, DetectionRun, Sample, ModelPipeline } from '@blokus-ai/shared';

export type { Sample } from '@blokus-ai/shared';

interface SamplesResponse {
    samples: Sample[];
}

// Progress tracking types (must match TrainedModelsTable and DetectionRunsTable)
export interface TrainingProgress {
    phase: 'detection' | 'classifier';
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

    // Fetch samples with optional model filter
    const fetchSamples = useCallback(async (forceFetch = false, modelId?: string | null) => {
        if (hasFetchedSamples && !forceFetch) {
            return;
        }

        setIsLoadingSamples(true);
        try {
            const url = modelId
                ? `/api/ml/samples?modelId=${modelId}`
                : '/api/ml/samples';
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

    // Load all data on mount
    useEffect(() => {
        fetchSamples();
        fetchTrainingRuns();
        fetchDetectionRuns();
        fetchPipelines();
    }, [fetchSamples, fetchTrainingRuns, fetchDetectionRuns, fetchPipelines]);

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
    };
});
