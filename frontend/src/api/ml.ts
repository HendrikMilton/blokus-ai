import { http } from './config';
import {
    DetectBoardRequest,
    DetectBoardResponse,
    AnalyzeBoardRequest,
    AnalyzeBoardResponse,
    mlDetectBoardPath,
    mlAnalyzeBoardPath,
    BoardCell,
    PlayerColor,
    TrainingRun,
    DetectionRun,
    BoardDetection,
    ModelPipeline,
    Tag,
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
    detectBoard: async (image: string): Promise<DetectBoardResponse> => {
        const request: DetectBoardRequest = { image };
        return http.post<DetectBoardResponse>(mlDetectBoardPath, request);
    },

    analyze: async (boardState: BoardCell[][], selectedColor: PlayerColor): Promise<AnalyzeBoardResponse> => {
        const request: AnalyzeBoardRequest = { boardState, selectedColor };
        return http.post<AnalyzeBoardResponse>(mlAnalyzeBoardPath, request);
    },

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
        const response = await http.get<{ pipelines: ModelPipeline[] }>('/api/ml/pipelines');
        return response.pipelines;
    },

    getPipeline: async (id: string): Promise<ModelPipeline> => {
        return http.get<ModelPipeline>(`/api/ml/pipelines/${id}`);
    },

    createPipeline: async (name: string, componentIds: string[]): Promise<ModelPipeline> => {
        return http.post<ModelPipeline>('/api/ml/pipelines', { name, componentIds });
    },

    updatePipeline: async (id: string, data: { name?: string; componentIds?: string[] }): Promise<ModelPipeline> => {
        return http.put<ModelPipeline>(`/api/ml/pipelines/${id}`, data);
    },

    deletePipeline: async (id: string): Promise<void> => {
        await http.delete(`/api/ml/pipelines/${id}`);
    },

    // Perspective correction for stored samples
    correctPerspective: async (sampleId: string, bbox: { x: number; y: number; width: number; height: number }): Promise<PerspectiveCorrections> => {
        return http.post<PerspectiveCorrections>(`/api/ml/detect/${sampleId}/perspective`, { bbox });
    },

    // Corner labeling
    getCorners: async (sampleId: string): Promise<{ corners: BoardCorners | null }> => {
        return http.get<{ corners: BoardCorners | null }>(`/api/ml/samples/${sampleId}/corners`);
    },

    saveCorners: async (sampleId: string, corners: BoardCorners): Promise<void> => {
        await http.post(`/api/ml/samples/${sampleId}/corners`, corners);
    },

    clearCorners: async (sampleId: string): Promise<void> => {
        await http.delete(`/api/ml/samples/${sampleId}/corners`);
    },

    // Tags
    getTags: async (): Promise<Tag[]> => {
        const response = await http.get<{ tags: Tag[] }>('/api/ml/tags');
        return response.tags;
    },

    createTag: async (name: string): Promise<Tag> => {
        return http.post<Tag>('/api/ml/tags', { name });
    },

    deleteTag: async (id: string): Promise<void> => {
        await http.delete(`/api/ml/tags/${id}`);
    },

    getTagsForSample: async (sampleId: string): Promise<Tag[]> => {
        const response = await http.get<{ tags: Tag[] }>(`/api/ml/samples/${sampleId}/tags`);
        return response.tags;
    },

    addTagToSample: async (sampleId: string, tagId: string): Promise<void> => {
        await http.post(`/api/ml/samples/${sampleId}/tags`, { tagId });
    },

    removeTagFromSample: async (sampleId: string, tagId: string): Promise<void> => {
        await http.delete(`/api/ml/samples/${sampleId}/tags/${tagId}`);
    },
};

export interface BoardCorners {
    topLeft: { x: number; y: number };
    topRight: { x: number; y: number };
    bottomRight: { x: number; y: number };
    bottomLeft: { x: number; y: number };
}
