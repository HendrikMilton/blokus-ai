import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
    await knex.schema.alterTable('corner_detections', (table) => {
        // Contour quality metrics
        table.float('contour_aspect_ratio').nullable();
        table.float('contour_area_ratio').nullable();
        table.float('contour_avg_angle').nullable();

        // Hough quality metrics
        table.float('hough_aspect_ratio').nullable();
        table.float('hough_area_ratio').nullable();
        table.float('hough_avg_angle').nullable();

        // Edge quality metrics
        table.float('edge_aspect_ratio').nullable();
        table.float('edge_area_ratio').nullable();
        table.float('edge_avg_angle').nullable();
    });
}

export async function down(knex: Knex): Promise<void> {
    await knex.schema.alterTable('corner_detections', (table) => {
        table.dropColumn('contour_aspect_ratio');
        table.dropColumn('contour_area_ratio');
        table.dropColumn('contour_avg_angle');
        table.dropColumn('hough_aspect_ratio');
        table.dropColumn('hough_area_ratio');
        table.dropColumn('hough_avg_angle');
        table.dropColumn('edge_aspect_ratio');
        table.dropColumn('edge_area_ratio');
        table.dropColumn('edge_avg_angle');
    });
}
