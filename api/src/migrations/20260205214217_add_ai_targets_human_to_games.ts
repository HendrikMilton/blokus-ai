import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('games', (table) => {
        table.boolean('ai_targets_human').nullable().defaultTo(null);
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('games', (table) => {
        table.dropColumn('ai_targets_human');
    });
};
