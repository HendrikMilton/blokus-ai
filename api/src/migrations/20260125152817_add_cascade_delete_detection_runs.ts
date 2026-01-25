import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('detection_runs', (table) => {
        table.dropForeign('model_id');
        table
            .foreign('model_id')
            .references('id')
            .inTable('training_runs')
            .onDelete('CASCADE');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('detection_runs', (table) => {
        table.dropForeign('model_id');
        table.foreign('model_id').references('id').inTable('training_runs');
    });
};
