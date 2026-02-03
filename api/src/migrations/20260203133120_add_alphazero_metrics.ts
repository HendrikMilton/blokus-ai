import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    // Create detailed training metrics table for per-step logging
    await knex.schema.createTable('alphazero_metrics', (table) => {
        table.uuid('id').primary().defaultTo(knex.raw('gen_random_uuid()'));
        table.uuid('run_id').notNullable().references('id').inTable('alphazero_runs').onDelete('CASCADE');
        table.integer('generation').notNullable();
        table.integer('step').notNullable();

        // Loss metrics
        table.float('policy_loss');
        table.float('value_loss');
        table.float('total_loss');

        // Learning rate (for tracking schedule)
        table.float('learning_rate');

        // Gradient stats (optional, for debugging)
        table.float('grad_norm');

        // Heuristic metrics from self-play (per generation, not per step)
        table.float('avg_corners');
        table.float('avg_mobility');
        table.float('center_coverage');

        // Self-play stats
        table.float('games_per_sec');
        table.float('moves_per_sec');
        table.integer('replay_buffer_size');

        // Timing
        table.float('step_time_ms');

        table.timestamp('created_at').notNullable().defaultTo(knex.fn.now());

        // Indexes for efficient querying
        table.index(['run_id', 'generation', 'step']);
        table.index('run_id');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('alphazero_metrics');
};
