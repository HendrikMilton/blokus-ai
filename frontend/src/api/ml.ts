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
};
