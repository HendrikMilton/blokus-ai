import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('detection_runs', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('status', 20).notNullable().defaultTo('pending'); // pending, running, completed, failed
        table.uuid('model_id').notNullable().references('id').inTable('training_runs');
        table.integer('total_count').notNullable();
        table.integer('processed_count').notNullable().defaultTo(0);
        table.integer('success_count').nullable();
        table.integer('failed_count').nullable();
        table.timestamp('started_at').nullable();
        table.timestamp('completed_at').nullable();
        table.integer('duration_ms').nullable();
        table.text('error_message').nullable();
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
        table.timestamp('updated_at').defaultTo(knex.fn.now()).notNullable();

        table.index('status');
        table.index('created_at');
        table.index('model_id');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('detection_runs');
};
