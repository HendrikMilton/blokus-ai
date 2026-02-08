import { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
    await knex.schema.alterTable('training_runs', (table) => {
        table.text('model_s3_url').nullable();
    });
}

export async function down(knex: Knex): Promise<void> {
    await knex.schema.alterTable('training_runs', (table) => {
        table.dropColumn('model_s3_url');
    });
}
