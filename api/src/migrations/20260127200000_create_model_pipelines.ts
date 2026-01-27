import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('model_pipelines', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('name', 255).notNullable();
        table.jsonb('component_ids').notNullable(); // Array of training_run UUIDs in order
        table.timestamp('created_at').defaultTo(knex.fn.now());
        table.timestamp('updated_at').defaultTo(knex.fn.now());
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTable('model_pipelines');
};
