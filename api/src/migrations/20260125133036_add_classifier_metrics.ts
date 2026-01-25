import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
    await knex.schema.alterTable('training_runs', (table) => {
        table.float('classifier_accuracy').nullable();
        table.float('classifier_loss').nullable();
    });
}

export async function down(knex: Knex): Promise<void> {
    await knex.schema.alterTable('training_runs', (table) => {
        table.dropColumn('classifier_accuracy');
        table.dropColumn('classifier_loss');
    });
}
