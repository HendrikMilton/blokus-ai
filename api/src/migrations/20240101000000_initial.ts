import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
    // Users table
    await knex.schema.createTable('users', (table) => {
        table.increments('id').primary();
        table.string('email').notNullable().unique();
        table.string('name').notNullable();
        table.string('password').nullable();
        table.enum('role', ['admin', 'user']).notNullable().defaultTo('user');
        table.enum('status', ['active', 'inactive']).notNullable().defaultTo('active');
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
        table.timestamp('updated_at').defaultTo(knex.fn.now()).notNullable();

        table.index('email');
        table.index('status');
    });

    // Application errors table (for error logging)
    await knex.schema.createTable('application_errors', (table) => {
        table.increments('id').primary();
        table.string('cid').notNullable().index();
        table.text('error_message').notNullable();
        table.text('error_stack');
        table.string('request_method', 10);
        table.string('request_path', 500);
        table.text('request_body');
        table.string('user_id').index();
        table.timestamp('created_at').defaultTo(knex.fn.now()).notNullable();
    });

    // Insert test user
    await knex('users').insert({
        email: 'admin@example.com',
        name: 'Admin User',
        password: '$2b$10$rQZ5kEHQ0Z8t5Xg0QZmzOeM5YX0QZmzOeM5YX0QZmzOeM5YX0Qzm', // "password123"
        role: 'admin',
        status: 'active',
    });

    // Enable CDC (Change Data Capture) using PostgreSQL logical replication
    await knex.raw(`
        CREATE PUBLICATION blokus_ai_cdc FOR TABLE users;
    `);

    await knex.raw(`
        SELECT CASE
            WHEN NOT EXISTS (
                SELECT 1 FROM pg_replication_slots WHERE slot_name = 'blokus_ai_cdc_slot'
            )
            THEN pg_create_logical_replication_slot('blokus_ai_cdc_slot', 'pgoutput')
            ELSE NULL
        END;
    `);
}

export async function down(knex: Knex): Promise<void> {
    // Drop replication slot
    await knex.raw(`
        SELECT pg_drop_replication_slot('blokus_ai_cdc_slot')
        WHERE EXISTS (
            SELECT 1 FROM pg_replication_slots WHERE slot_name = 'blokus_ai_cdc_slot'
        );
    `);

    // Drop publication
    await knex.raw('DROP PUBLICATION IF EXISTS blokus_ai_cdc;');

    await knex.schema.dropTableIfExists('application_errors');
    await knex.schema.dropTableIfExists('users');
}

// Slot creation is not transactional
export const config = { transaction: false };
