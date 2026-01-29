import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('tags', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('name', 100).notNullable().unique();
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('tags');
};
