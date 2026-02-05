import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    const hasColumn = await knex.schema.hasColumn('alphazero_checkpoints', 'heuristic_win_rate');
    if (!hasColumn) {
        await knex.schema.alterTable('alphazero_checkpoints', (table) => {
            table.float('heuristic_win_rate').nullable();
        });
    }
};

export const down = async (knex: Knex): Promise<void> => {
    const hasColumn = await knex.schema.hasColumn('alphazero_checkpoints', 'heuristic_win_rate');
    if (hasColumn) {
        await knex.schema.alterTable('alphazero_checkpoints', (table) => {
            table.dropColumn('heuristic_win_rate');
        });
    }
};
