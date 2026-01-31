import { http } from './config';
import type { GridLabels, GridAnalysisResult, PieceDefinition } from '@blokus-ai/shared';

export const PIECES_API = {
    // Get all piece definitions for display
    getCatalog: async (): Promise<PieceDefinition[]> => {
        return http.get<PieceDefinition[]>('/api/pieces/catalog');
    },

    // Analyze a grid to find which pieces are on/off board for each color
    analyzeGrid: async (grid: GridLabels): Promise<GridAnalysisResult> => {
        return http.post<GridAnalysisResult>('/api/pieces/analyze', { grid });
    },
};
