import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
    await knex.schema.alterTable('corner_detections', (table) => {
        // Rename contour columns to best
        table.renameColumn('contour_success', 'best_success');
        table.renameColumn('contour_tl_x', 'best_tl_x');
        table.renameColumn('contour_tl_y', 'best_tl_y');
        table.renameColumn('contour_tr_x', 'best_tr_x');
        table.renameColumn('contour_tr_y', 'best_tr_y');
        table.renameColumn('contour_br_x', 'best_br_x');
        table.renameColumn('contour_br_y', 'best_br_y');
        table.renameColumn('contour_bl_x', 'best_bl_x');
        table.renameColumn('contour_bl_y', 'best_bl_y');
        table.renameColumn('contour_error', 'best_error');
        table.renameColumn('contour_aspect_ratio', 'best_aspect_ratio');
        table.renameColumn('contour_area_ratio', 'best_area_ratio');
        table.renameColumn('contour_avg_angle', 'best_avg_angle');
    });
}

export async function down(knex: Knex): Promise<void> {
    await knex.schema.alterTable('corner_detections', (table) => {
        // Rename best columns back to contour
        table.renameColumn('best_success', 'contour_success');
        table.renameColumn('best_tl_x', 'contour_tl_x');
        table.renameColumn('best_tl_y', 'contour_tl_y');
        table.renameColumn('best_tr_x', 'contour_tr_x');
        table.renameColumn('best_tr_y', 'contour_tr_y');
        table.renameColumn('best_br_x', 'contour_br_x');
        table.renameColumn('best_br_y', 'contour_br_y');
        table.renameColumn('best_bl_x', 'contour_bl_x');
        table.renameColumn('best_bl_y', 'contour_bl_y');
        table.renameColumn('best_error', 'contour_error');
        table.renameColumn('best_aspect_ratio', 'contour_aspect_ratio');
        table.renameColumn('best_area_ratio', 'contour_area_ratio');
        table.renameColumn('best_avg_angle', 'contour_avg_angle');
    });
}
