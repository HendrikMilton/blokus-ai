import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.float('classifier_confidence').nullable();
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.dropColumn('classifier_confidence');
    });
};
