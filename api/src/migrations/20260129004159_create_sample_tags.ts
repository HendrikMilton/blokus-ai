import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('sample_tags', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('sample_id', 36).notNullable();
        table.uuid('tag_id').notNullable().references('id').inTable('tags').onDelete('CASCADE');
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();

        table.unique(['sample_id', 'tag_id']);
        table.index('sample_id');
        table.index('tag_id');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('sample_tags');
};
