import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('training_samples', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('image_path', 500).notNullable();
        table.integer('image_width').notNullable();
        table.integer('image_height').notNullable();
        table.jsonb('detected_state').notNullable();
        table.jsonb('labeled_state').nullable();
        table.enum('source', ['upload', 'camera', 'synthetic']).notNullable().defaultTo('upload');
        table.text('notes').nullable();
        table.enum('status', ['pending', 'labeled', 'verified']).notNullable().defaultTo('pending');
        table.timestamp('labeled_at').nullable();
        table.boolean('features_extracted').notNullable().defaultTo(false);
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
        table.timestamp('updated_at').defaultTo(knex.fn.now()).notNullable();

        table.index('status');
        table.index('created_at');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('training_samples');
};
