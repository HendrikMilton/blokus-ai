import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('training_runs', (table) => {
        table.float('pose_error').nullable();
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('training_runs', (table) => {
        table.dropColumn('pose_error');
    });
};
