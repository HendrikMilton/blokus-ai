import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('training_runs', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('status', 20).notNullable().defaultTo('pending'); // pending, running, completed, failed
        table.integer('image_count').notNullable();
        table.timestamp('started_at').nullable();
        table.timestamp('completed_at').nullable();
        table.integer('duration_ms').nullable();
        // Metrics (populated after training)
        table.float('accuracy').nullable();
        table.float('loss').nullable();
        table.float('val_accuracy').nullable();
        table.float('val_loss').nullable();
        // Error info for failed runs
        table.text('error_message').nullable();
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
        table.timestamp('updated_at').defaultTo(knex.fn.now()).notNullable();

        table.index('status');
        table.index('created_at');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('training_runs');
};
