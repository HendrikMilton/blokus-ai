import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.timestamp('human_labeled_at').nullable();
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.dropColumn('human_labeled_at');
    });
};
