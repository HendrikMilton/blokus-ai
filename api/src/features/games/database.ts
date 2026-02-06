import { Knex } from 'knex';
import { knex } from '../../knex';
import type { SavedGame, SavedGameSummary, PlayerColor, Move } from '@blokus-ai/shared';

interface GameRow {
    id: string;
    mode: string;
    human_colors: PlayerColor[];
    ai_colors: PlayerColor[];
    ai_models: Partial<Record<PlayerColor, string>> | null;
    ai_targets_human: boolean | null;
    moves: Move[];
    scores: Record<PlayerColor, number>;
    winner: string | null;
    total_moves: number;
    created_at: Date;
}

const rowToSavedGame = (row: GameRow): SavedGame => ({
    id: row.id,
    mode: row.mode,
    humanColors: row.human_colors,
    aiColors: row.ai_colors,
    aiModels: row.ai_models,
    aiTargetsHuman: row.ai_targets_human ?? null,
    moves: row.moves,
    scores: row.scores,
    winner: (row.winner as PlayerColor) || null,
    totalMoves: row.total_moves,
    createdAt: row.created_at.toISOString(),
});

const rowToSummary = (row: GameRow): SavedGameSummary => ({
    id: row.id,
    mode: row.mode,
    humanColors: row.human_colors,
    aiColors: row.ai_colors,
    aiModels: row.ai_models,
    aiTargetsHuman: row.ai_targets_human ?? null,
    scores: row.scores,
    winner: (row.winner as PlayerColor) || null,
    totalMoves: row.total_moves,
    createdAt: row.created_at.toISOString(),
});

export interface CreateGameData {
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

export const createGame = async (
    data: CreateGameData,
    trx: Knex.Transaction | Knex = knex
): Promise<SavedGame> => {
    const [row] = await trx<GameRow>('games')
        .insert({
            mode: data.mode,
            human_colors: JSON.stringify(data.humanColors) as unknown as PlayerColor[],
            ai_colors: JSON.stringify(data.aiColors) as unknown as PlayerColor[],
            ai_models: data.aiModels ? JSON.stringify(data.aiModels) as unknown as Partial<Record<PlayerColor, string>> : null,
            ai_targets_human: data.aiTargetsHuman ?? null,
            moves: JSON.stringify(data.moves) as unknown as Move[],
            scores: JSON.stringify(data.scores) as unknown as Record<PlayerColor, number>,
            winner: data.winner,
            total_moves: data.totalMoves,
        })
        .returning('*');
    return rowToSavedGame(row);
};

export const listGames = async (
    trx: Knex.Transaction | Knex = knex
): Promise<SavedGameSummary[]> => {
    const rows = await trx<GameRow>('games')
        .select('id', 'mode', 'human_colors', 'ai_colors', 'ai_models', 'ai_targets_human', 'scores', 'winner', 'total_moves', 'created_at')
        .orderBy('created_at', 'desc');
    return rows.map(rowToSummary);
};

export const getGame = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<SavedGame | null> => {
    const row = await trx<GameRow>('games')
        .where('id', id)
        .first();
    return row ? rowToSavedGame(row) : null;
};

export const deleteGame = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('games')
        .where('id', id)
        .delete();
    return deleted > 0;
};
