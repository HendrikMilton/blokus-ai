import { http } from './config';
import type {
    TrainingRun,
    DetectionRun,
    BoardDetection,
    ModelPipeline,
    Tag,
    CornerDetectionRun,
    CornerDetectionResult,
    BoardCorners,
    PerspectiveFromCornersRequest,
    PerspectiveFromCornersResponse,
    GridLabels,
    GridLabel,
    GridPrediction,
    GridPredictionRecord,
} from '@blokus-ai/shared';


export interface PerspectiveCorrectionMethod {
    method: string;
    image?: string;  // base64 encoded
    success: boolean;
    corners?: number[][];
}

export interface PerspectiveCorrections {
    hough_lines_merged: PerspectiveCorrectionMethod;
    hough_lines_extreme: PerspectiveCorrectionMethod;
    color_segmentation: PerspectiveCorrectionMethod;
    contour_detection: PerspectiveCorrectionMethod;
}

export interface TestDetectResponse {
    detection: BoardDetection;
    processing_time_ms: number;
    correctedBoardImage?: string;  // base64 encoded corrected board image (best method)
    perspectiveCorrections?: PerspectiveCorrections;  // all methods for comparison
}

export const ML_API = {
    startTraining: async (sampleIds?: string[]): Promise<TrainingRun> => {
        return http.post<TrainingRun>('/api/ml/train', { sampleIds });
    },

    getTrainingRuns: async (): Promise<TrainingRun[]> => {
        const response = await http.get<{ runs: TrainingRun[] }>('/api/ml/training-runs');
        return response.runs;
    },

    // Detection runs
    startBulkDetection: async (modelId: string, includeExistingAiLabeled = false): Promise<DetectionRun> => {
        return http.post<DetectionRun>('/api/ml/detection-runs', { modelId, includeExistingAiLabeled });
    },

    getDetectionRuns: async (): Promise<DetectionRun[]> => {
        const response = await http.get<{ runs: DetectionRun[] }>('/api/ml/detection-runs');
        return response.runs;
    },

    getDetectionRun: async (id: string): Promise<DetectionRun> => {
        return http.get<DetectionRun>(`/api/ml/detection-runs/${id}`);
    },

    // Test detection with camera image
    testDetect: async (image: string, modelId?: string): Promise<TestDetectResponse> => {
        return http.post<TestDetectResponse>('/api/ml/detect/test', { image, modelId });
    },

    // Pipeline CRUD
    getPipelines: async (): Promise<ModelPipeline[]> => {
        const response = await http.get<{ pipelines: ModelPipeline[] }>('/api/pipelines');
        return response.pipelines;
    },

    getPipeline: async (id: string): Promise<ModelPipeline> => {
        return http.get<ModelPipeline>(`/api/pipelines/${id}`);
    },

    createPipeline: async (name: string, componentIds: string[]): Promise<ModelPipeline> => {
        return http.post<ModelPipeline>('/api/pipelines', { name, componentIds });
    },

    updatePipeline: async (id: string, data: { name?: string; componentIds?: string[] }): Promise<ModelPipeline> => {
        return http.put<ModelPipeline>(`/api/pipelines/${id}`, data);
    },

    deletePipeline: async (id: string): Promise<void> => {
        await http.delete(`/api/pipelines/${id}`);
    },

    // Perspective correction for stored samples (old method using bbox)
    correctPerspective: async (sampleId: string, bbox: { x: number; y: number; width: number; height: number }): Promise<PerspectiveCorrections> => {
        return http.post<PerspectiveCorrections>(`/api/ml/detect/${sampleId}/perspective`, { bbox });
    },

    // Perspective correction using saved corner coordinates
    correctPerspectiveFromCorners: async (
        sampleId: string,
        corners: PerspectiveFromCornersRequest
    ): Promise<PerspectiveFromCornersResponse> => {
        return http.post<PerspectiveFromCornersResponse>(
            `/api/ml/detect/${sampleId}/perspective-from-corners`,
            corners
        );
    },

    // Corner detection (multiple methods)
    detectCorners: async (sampleId: string, bbox: { x: number; y: number; width: number; height: number }): Promise<CornerDetections> => {
        return http.post<CornerDetections>(`/api/ml/detect/${sampleId}/corners`, { bbox });
    },

    // Corner labeling
    getCorners: async (sampleId: string): Promise<{ corners: BoardCorners | null }> => {
        return http.get<{ corners: BoardCorners | null }>(`/api/samples/${sampleId}/corners`);
    },

    saveCorners: async (sampleId: string, corners: BoardCorners): Promise<void> => {
        await http.post(`/api/samples/${sampleId}/corners`, corners);
    },

    clearCorners: async (sampleId: string): Promise<void> => {
        await http.delete(`/api/samples/${sampleId}/corners`);
    },

    // Tags
    getTags: async (): Promise<Tag[]> => {
        const response = await http.get<{ tags: Tag[] }>('/api/tags');
        return response.tags;
    },

    createTag: async (name: string): Promise<Tag> => {
        return http.post<Tag>('/api/tags', { name });
    },

    deleteTag: async (id: string): Promise<void> => {
        await http.delete(`/api/tags/${id}`);
    },

    getTagsForSample: async (sampleId: string): Promise<Tag[]> => {
        const response = await http.get<{ tags: Tag[] }>(`/api/tags/samples/${sampleId}`);
        return response.tags;
    },

    addTagToSample: async (sampleId: string, tagId: string): Promise<void> => {
        await http.post(`/api/tags/samples/${sampleId}`, { tagId });
    },

    removeTagFromSample: async (sampleId: string, tagId: string): Promise<void> => {
        await http.delete(`/api/tags/samples/${sampleId}/${tagId}`);
    },

    // Corner detection runs
    startCornerDetection: async (sampleIds: string[]): Promise<CornerDetectionRun> => {
        return http.post<CornerDetectionRun>('/api/ml/corner-runs', { sampleIds });
    },

    getCornerRuns: async (): Promise<CornerDetectionRun[]> => {
        const response = await http.get<{ runs: CornerDetectionRun[] }>('/api/ml/corner-runs');
        return response.runs;
    },

    getCornerRun: async (id: string): Promise<CornerDetectionRun> => {
        return http.get<CornerDetectionRun>(`/api/ml/corner-runs/${id}`);
    },

    deleteCornerRun: async (id: string): Promise<void> => {
        await http.delete(`/api/ml/corner-runs/${id}`);
    },

    getCornerDetections: async (runId: string): Promise<CornerDetectionResult[]> => {
        const response = await http.get<{ results: CornerDetectionResult[] }>(`/api/ml/corner-detections?runId=${runId}`);
        return response.results;
    },

    getCornerDetectionsForSamples: async (sampleIds: string[]): Promise<CornerDetectionResult[]> => {
        const response = await http.get<{ results: CornerDetectionResult[] }>(`/api/ml/corner-detections?sampleIds=${sampleIds.join(',')}`);
        return response.results;
    },

    // Grid labels
    getGridLabel: async (sampleId: string): Promise<GridLabels | null> => {
        const response = await http.get<{ grid: GridLabels | null }>(`/api/samples/${sampleId}/grid`);
        return response.grid;
    },

    getGridLabels: async (sampleIds: string[]): Promise<GridLabel[]> => {
        if (sampleIds.length === 0) {
            return [];
        }
        const response = await http.get<{ labels: GridLabel[] }>(
            `/api/ml/grid-labels?sampleIds=${sampleIds.join(',')}`
        );
        return response.labels;
    },

    saveGridLabel: async (sampleId: string, grid: GridLabels): Promise<GridLabel> => {
        return http.post<GridLabel>(`/api/samples/${sampleId}/grid`, { grid });
    },

    clearGridLabel: async (sampleId: string): Promise<void> => {
        await http.delete(`/api/samples/${sampleId}/grid`);
    },

    predictGrid: async (sampleId: string): Promise<GridPrediction> => {
        return http.post<GridPrediction>(`/api/samples/${sampleId}/grid/predict`, {});
    },

    // Grid prediction runs (bulk AI prediction)
    startGridRun: async (sampleIds?: string[]): Promise<{ started: number; skipped: number }> => {
        return http.post<{ started: number; skipped: number }>('/api/ml/grid-runs', { sampleIds });
    },

    getGridPredictions: async (sampleIds: string[]): Promise<GridPredictionRecord[]> => {
        const response = await http.get<{ predictions: GridPredictionRecord[] }>(
            `/api/ml/grid-predictions?sampleIds=${sampleIds.join(',')}`
        );
        return response.predictions;
    },
};

export interface CornerDetectionMethod {
    method: string;
    corners: BoardCorners | null;
    success: boolean;
    debug_image?: string;  // base64 encoded image with corners drawn
}

export interface CornerDetections {
    contour: CornerDetectionMethod;
    hough_ransac: CornerDetectionMethod;
    edge_intersection: CornerDetectionMethod;
}
