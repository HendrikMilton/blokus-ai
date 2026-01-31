import { http } from './config';
import type {
    GameConfig,
    GameState,
    Move,
    GameSummary,
    PlayerColor,
    GamePieceDefinition,
} from '@blokus-ai/shared';

export interface NewGameResponse {
    gameId: string;
    config: GameConfig;
    state: GameState;
}

export interface MoveResponse {
    success: boolean;
    error?: string;
    state?: GameState;
    aiMoves?: Move[];
}

export interface LegalMovesResponse {
    moves: Move[];
    count: number;
}

export const GAME_API = {
    // Get all piece definitions with orientations
    getPieces: async (): Promise<GamePieceDefinition[]> => {
        return http.get<GamePieceDefinition[]>('/api/game/pieces');
    },

    // Create a new game
    newGame: async (config: GameConfig): Promise<NewGameResponse> => {
        return http.post<NewGameResponse>('/api/game/new', config);
    },

    // Get game state
    getGame: async (gameId: string): Promise<{ state: GameState; moveHistory: Move[] }> => {
        return http.get(`/api/game/${gameId}`);
    },

    // Get legal moves for current player
    getLegalMoves: async (gameId: string): Promise<LegalMovesResponse> => {
        return http.get(`/api/game/${gameId}/legal-moves`);
    },

    // Make a move
    makeMove: async (gameId: string, move: Move): Promise<MoveResponse> => {
        return http.post(`/api/game/${gameId}/move`, move);
    },

    // Make a move and let AI respond
    makeMoveWithAI: async (
        gameId: string,
        move: Move,
        aiColors: PlayerColor[]
    ): Promise<MoveResponse> => {
        return http.post(`/api/game/${gameId}/move-with-ai`, { move, aiColors });
    },

    // Pass turn
    pass: async (gameId: string): Promise<MoveResponse> => {
        return http.post(`/api/game/${gameId}/pass`, {});
    },

    // Pass and let AI continue
    passWithAI: async (gameId: string, aiColors: PlayerColor[]): Promise<MoveResponse> => {
        return http.post(`/api/game/${gameId}/pass-with-ai`, { aiColors });
    },

    // Get game summary
    getSummary: async (gameId: string): Promise<GameSummary> => {
        return http.get(`/api/game/${gameId}/summary`);
    },

    // Delete game
    deleteGame: async (gameId: string): Promise<void> => {
        await http.delete(`/api/game/${gameId}`);
    },
};
