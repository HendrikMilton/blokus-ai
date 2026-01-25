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
} from '@blokus-ai/shared';

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
};
