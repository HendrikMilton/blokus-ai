import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('games', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('mode', 20).notNullable(); // '1v3', '2v2', 'spectate', 'local'
        table.jsonb('human_colors').notNullable();
        table.jsonb('ai_colors').notNullable();
        table.jsonb('ai_models').nullable(); // Per-player AI model IDs
        table.jsonb('moves').notNullable(); // Move[] array
        table.jsonb('scores').notNullable(); // { blue: 45, yellow: 32, ... }
        table.string('winner', 10).nullable();
        table.integer('total_moves').notNullable();
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
    });

    await knex.schema.raw('CREATE INDEX idx_games_created_at ON games (created_at DESC)');
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('games');
};
