import { http } from './config';
import type { SavedGame, SavedGameSummary, PlayerColor, Move } from '@blokus-ai/shared';

export interface SaveGameRequest {
    mode: string;
    humanColors: PlayerColor[];
    aiColors: PlayerColor[];
    aiModels?: Partial<Record<PlayerColor, string>> | null;
    aiTargetsHuman?: boolean | null;
    moves: Move[];
    scores: Record<PlayerColor, number>;
    winner: PlayerColor | null;
    totalMoves: number;
}

export const GAMES_API = {
    saveGame: async (data: SaveGameRequest): Promise<SavedGame> => {
        return http.post<SavedGame>('/api/games', data);
    },

    listGames: async (): Promise<{ games: SavedGameSummary[] }> => {
        return http.get('/api/games');
    },

    getGame: async (id: string): Promise<SavedGame> => {
        return http.get(`/api/games/${id}`);
    },

    deleteGame: async (id: string): Promise<void> => {
        await http.delete(`/api/games/${id}`);
    },
};
