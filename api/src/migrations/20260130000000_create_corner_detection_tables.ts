import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
    // Create corner_detection_runs table
    await knex.schema.createTable('corner_detection_runs', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('status', 20).notNullable().defaultTo('pending');
        table.integer('total_count').notNullable();
        table.integer('processed_count').notNullable().defaultTo(0);
        table.integer('success_count').nullable();
        table.integer('failed_count').nullable();
        table.timestamp('started_at').nullable();
        table.timestamp('completed_at').nullable();
        table.integer('duration_ms').nullable();
        table.text('error_message').nullable();
        table.timestamp('created_at').defaultTo(knex.fn.now());
        table.timestamp('updated_at').defaultTo(knex.fn.now());
    });

    // Create corner_detections table
    await knex.schema.createTable('corner_detections', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('sample_id', 36).notNullable();
        table.uuid('run_id').references('id').inTable('corner_detection_runs').onDelete('CASCADE');

        // Contour method results
        table.boolean('contour_success').nullable();
        table.float('contour_tl_x').nullable();
        table.float('contour_tl_y').nullable();
        table.float('contour_tr_x').nullable();
        table.float('contour_tr_y').nullable();
        table.float('contour_br_x').nullable();
        table.float('contour_br_y').nullable();
        table.float('contour_bl_x').nullable();
        table.float('contour_bl_y').nullable();
        table.float('contour_error').nullable();

        // Hough+RANSAC method results
        table.boolean('hough_success').nullable();
        table.float('hough_tl_x').nullable();
        table.float('hough_tl_y').nullable();
        table.float('hough_tr_x').nullable();
        table.float('hough_tr_y').nullable();
        table.float('hough_br_x').nullable();
        table.float('hough_br_y').nullable();
        table.float('hough_bl_x').nullable();
        table.float('hough_bl_y').nullable();
        table.float('hough_error').nullable();

        // Edge intersection method results
        table.boolean('edge_success').nullable();
        table.float('edge_tl_x').nullable();
        table.float('edge_tl_y').nullable();
        table.float('edge_tr_x').nullable();
        table.float('edge_tr_y').nullable();
        table.float('edge_br_x').nullable();
        table.float('edge_br_y').nullable();
        table.float('edge_bl_x').nullable();
        table.float('edge_bl_y').nullable();
        table.float('edge_error').nullable();

        table.timestamp('created_at').defaultTo(knex.fn.now());

        // Unique constraint on sample_id + run_id
        table.unique(['sample_id', 'run_id']);
    });

    // Create index on sample_id for efficient lookups
    await knex.schema.raw('CREATE INDEX idx_corner_detections_sample_id ON corner_detections(sample_id)');
}

export async function down(knex: Knex): Promise<void> {
    await knex.schema.dropTableIfExists('corner_detections');
    await knex.schema.dropTableIfExists('corner_detection_runs');
}
