import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('detection_runs', (table) => {
        table.float('avg_iou').nullable();
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('detection_runs', (table) => {
        table.dropColumn('avg_iou');
    });
};
