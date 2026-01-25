import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.renameColumn('corrected_has_board', 'human_has_board');
        table.renameColumn('corrected_is_entire_visible', 'human_is_entire_visible');
        table.renameColumn('corrected_bbox_x', 'human_bbox_x');
        table.renameColumn('corrected_bbox_y', 'human_bbox_y');
        table.renameColumn('corrected_bbox_width', 'human_bbox_width');
        table.renameColumn('corrected_bbox_height', 'human_bbox_height');
    });

    // Drop corrected_at since human_labeled_at already exists and serves the same purpose
    await knex.schema.alterTable('board_detections', (table) => {
        table.dropColumn('corrected_at');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    // Add back corrected_at
    await knex.schema.alterTable('board_detections', (table) => {
        table.timestamp('corrected_at', { useTz: true });
    });

    // Copy human_labeled_at to corrected_at
    await knex.raw('UPDATE board_detections SET corrected_at = human_labeled_at WHERE human_labeled_at IS NOT NULL');

    await knex.schema.alterTable('board_detections', (table) => {
        table.renameColumn('human_has_board', 'corrected_has_board');
        table.renameColumn('human_is_entire_visible', 'corrected_is_entire_visible');
        table.renameColumn('human_bbox_x', 'corrected_bbox_x');
        table.renameColumn('human_bbox_y', 'corrected_bbox_y');
        table.renameColumn('human_bbox_width', 'corrected_bbox_width');
        table.renameColumn('human_bbox_height', 'corrected_bbox_height');
    });
};
