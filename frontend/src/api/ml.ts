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
} from '@blokus-ai/shared';

export interface TestDetectResponse {
    detection: BoardDetection;
    processing_time_ms: number;
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

    startTraining: async (): Promise<TrainingRun> => {
        return http.post<TrainingRun>('/api/ml/train', {});
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
};
