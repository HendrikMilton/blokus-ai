import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('grid_predictions', (table) => {
        table.uuid('sample_id').primary(); // One AI prediction per sample (latest)
        table.jsonb('grid').notNullable(); // 20x20 array of colors
        table.float('confidence').notNullable(); // Average confidence across cells
        table.integer('processing_time_ms').notNullable();
        table.timestamp('created_at').notNullable().defaultTo(knex.fn.now());
        table.timestamp('updated_at').notNullable().defaultTo(knex.fn.now());
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('grid_predictions');
};
