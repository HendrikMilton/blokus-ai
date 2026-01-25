import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('board_detections', (table) => {
        table.float('confidence').nullable().alter();
    });
};

export const down = async (knex: Knex): Promise<void> => {
    // Set any nulls to 0 before making NOT NULL again
    await knex('board_detections')
        .whereNull('confidence')
        .update({ confidence: 0 });

    await knex.schema.alterTable('board_detections', (table) => {
        table.float('confidence').notNullable().alter();
    });
};
