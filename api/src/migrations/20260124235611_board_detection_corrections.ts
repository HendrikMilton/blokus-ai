import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.boolean('corrected_has_board').nullable();
        table.boolean('corrected_is_entire_visible').nullable();
        table.float('corrected_bbox_x').nullable();
        table.float('corrected_bbox_y').nullable();
        table.float('corrected_bbox_width').nullable();
        table.float('corrected_bbox_height').nullable();
        table.timestamp('corrected_at').nullable();
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.dropColumn('corrected_has_board');
        table.dropColumn('corrected_is_entire_visible');
        table.dropColumn('corrected_bbox_x');
        table.dropColumn('corrected_bbox_y');
        table.dropColumn('corrected_bbox_width');
        table.dropColumn('corrected_bbox_height');
        table.dropColumn('corrected_at');
    });
};
