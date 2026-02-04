import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
    await knex.schema.alterTable('alphazero_metrics', (table) => {
        table.float('policy_entropy').nullable();
        table.float('policy_top1').nullable();
        table.float('policy_top5').nullable();
    });
}

export async function down(knex: Knex): Promise<void> {
    await knex.schema.alterTable('alphazero_metrics', (table) => {
        table.dropColumn('policy_entropy');
        table.dropColumn('policy_top1');
        table.dropColumn('policy_top5');
    });
}
