import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        // Corner coordinates (normalized 0-1)
        // TL = top-left, TR = top-right, BR = bottom-right, BL = bottom-left
        table.float('corner_tl_x').nullable();
        table.float('corner_tl_y').nullable();
        table.float('corner_tr_x').nullable();
        table.float('corner_tr_y').nullable();
        table.float('corner_br_x').nullable();
        table.float('corner_br_y').nullable();
        table.float('corner_bl_x').nullable();
        table.float('corner_bl_y').nullable();
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.dropColumn('corner_tl_x');
        table.dropColumn('corner_tl_y');
        table.dropColumn('corner_tr_x');
        table.dropColumn('corner_tr_y');
        table.dropColumn('corner_br_x');
        table.dropColumn('corner_br_y');
        table.dropColumn('corner_bl_x');
        table.dropColumn('corner_bl_y');
    });
};
