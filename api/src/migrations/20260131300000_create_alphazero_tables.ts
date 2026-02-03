import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    // AlphaZero training runs
    await knex.schema.createTable('alphazero_runs', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('name', 100).notNullable();
        table.string('status', 20).notNullable().defaultTo('pending'); // pending, running, completed, failed, stopped
        table.jsonb('config').notNullable(); // Training configuration
        // Progress tracking
        table.integer('current_generation').defaultTo(0);
        table.integer('games_generated').defaultTo(0);
        table.integer('training_steps').defaultTo(0);
        // Metrics
        table.float('policy_loss').nullable();
        table.float('value_loss').nullable();
        table.float('best_win_rate').nullable();
        // Timing
        table.timestamp('started_at').nullable();
        table.timestamp('completed_at').nullable();
        // Error info
        table.text('error_message').nullable();
        // Timestamps
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
        table.timestamp('updated_at').defaultTo(knex.fn.now()).notNullable();

        table.index('status');
        table.index('created_at');
    });

    // AlphaZero model checkpoints
    await knex.schema.createTable('alphazero_checkpoints', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.uuid('run_id').notNullable().references('id').inTable('alphazero_runs').onDelete('CASCADE');
        table.integer('generation').notNullable();
        table.integer('step').notNullable();
        // Metrics at this checkpoint
        table.float('policy_loss').nullable();
        table.float('value_loss').nullable();
        table.float('eval_win_rate').nullable();
        table.float('eval_games').nullable();
        // Model info
        table.boolean('is_best').defaultTo(false);
        table.string('model_path', 255).notNullable();
        // Timestamps
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();

        table.index('run_id');
        table.index(['run_id', 'generation']);
        table.index('is_best');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('alphazero_checkpoints');
    await knex.schema.dropTableIfExists('alphazero_runs');
};
