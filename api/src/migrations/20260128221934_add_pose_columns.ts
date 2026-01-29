import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    // Add predicted corner columns to board_detections
    await knex.schema.alterTable('board_detections', (table) => {
        // Predicted corners from pose model (normalized 0-1)
        table.float('pred_corner_tl_x').nullable();
        table.float('pred_corner_tl_y').nullable();
        table.float('pred_corner_tr_x').nullable();
        table.float('pred_corner_tr_y').nullable();
        table.float('pred_corner_br_x').nullable();
        table.float('pred_corner_br_y').nullable();
        table.float('pred_corner_bl_x').nullable();
        table.float('pred_corner_bl_y').nullable();
        // Average corner error (normalized distance from ground truth)
        table.float('corner_error').nullable();
    });

    // Add pose metrics to training_runs
    await knex.schema.alterTable('training_runs', (table) => {
        table.float('pose_accuracy').nullable(); // PCK - % of corners within threshold
        table.float('pose_loss').nullable();
        table.integer('pose_image_count').nullable(); // Images with corner labels
    });

    // Add pose metrics to detection_runs
    await knex.schema.alterTable('detection_runs', (table) => {
        table.float('avg_corner_error').nullable(); // Mean normalized distance
        table.float('corner_accuracy').nullable(); // % within threshold
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.dropColumn('pred_corner_tl_x');
        table.dropColumn('pred_corner_tl_y');
        table.dropColumn('pred_corner_tr_x');
        table.dropColumn('pred_corner_tr_y');
        table.dropColumn('pred_corner_br_x');
        table.dropColumn('pred_corner_br_y');
        table.dropColumn('pred_corner_bl_x');
        table.dropColumn('pred_corner_bl_y');
        table.dropColumn('corner_error');
    });

    await knex.schema.alterTable('training_runs', (table) => {
        table.dropColumn('pose_accuracy');
        table.dropColumn('pose_loss');
        table.dropColumn('pose_image_count');
    });

    await knex.schema.alterTable('detection_runs', (table) => {
        table.dropColumn('avg_corner_error');
        table.dropColumn('corner_accuracy');
    });
};
