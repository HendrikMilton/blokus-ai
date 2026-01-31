import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('grid_labels', (table) => {
        table.uuid('sample_id').primary(); // No foreign key - samples are in S3
        table.jsonb('grid').notNullable(); // 20x20 array of cell colors
        table.timestamp('labeled_at').notNullable().defaultTo(knex.fn.now());
        table.timestamp('created_at').notNullable().defaultTo(knex.fn.now());
        table.timestamp('updated_at').notNullable().defaultTo(knex.fn.now());
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('grid_labels');
};
