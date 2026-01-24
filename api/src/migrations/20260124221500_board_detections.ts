import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.createTable('board_detections', (table) => {
        table.uuid('id').primary().defaultTo(knex.fn.uuid());
        table.string('sample_id', 36).notNullable().unique(); // References ML service sample
        table.boolean('has_board_detected').notNullable();
        table.boolean('is_entire_board_visible').notNullable();
        table.float('confidence').notNullable();
        table.float('bbox_x').nullable();
        table.float('bbox_y').nullable();
        table.float('bbox_width').nullable();
        table.float('bbox_height').nullable();
        table.float('processing_time_ms').notNullable();
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
        table.timestamp('updated_at').defaultTo(knex.fn.now()).notNullable();

        table.index('sample_id');
        table.index('created_at');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.dropTableIfExists('board_detections');
};
