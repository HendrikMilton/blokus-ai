import { Knex } from 'knex';

export const up = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('detection_runs', (table) => {
        table.uuid('pipeline_id').nullable().references('id').inTable('model_pipelines').onDelete('SET NULL');
        table.index('pipeline_id');
    });
};

export const down = async (knex: Knex): Promise<void> => {
    await knex.schema.alterTable('detection_runs', (table) => {
        table.dropIndex('pipeline_id');
        table.dropColumn('pipeline_id');
    });
};
