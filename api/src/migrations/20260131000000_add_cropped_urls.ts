import type { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('corner_detections', (table) => {
        table.string('best_cropped_url', 500).nullable();
        table.string('hough_cropped_url', 500).nullable();
        table.string('edge_cropped_url', 500).nullable();
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('corner_detections', (table) => {
        table.dropColumn('best_cropped_url');
        table.dropColumn('hough_cropped_url');
        table.dropColumn('edge_cropped_url');
    });
};
