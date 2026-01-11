import Knex, { Knex as KnexType } from 'knex';
import { Client } from 'pg';
import { expect } from 'chai';
import request from 'supertest';
import { config } from '../config';

// Test database configuration
const testDbName = `${config.POSTGRES_DATABASE}_test`;

let testKnex: KnexType;
let truncateQuery: string | null = null;

// PostgreSQL admin client (connects to 'postgres' database)
const adminClient = new Client({
    host: config.POSTGRES_HOST,
    port: config.POSTGRES_PORT,
    user: config.POSTGRES_USER,
    password: config.POSTGRES_PASSWORD,
    database: 'postgres',
});

/**
 * Setup test database before all tests
 */
export const setupTestDatabase = async (): Promise<void> => {
    await adminClient.connect();

    // Terminate existing connections and recreate database
    await adminClient.query(`
        SELECT pg_terminate_backend(pg_stat_activity.pid)
        FROM pg_stat_activity
        WHERE pg_stat_activity.datname = '${testDbName}'
        AND pid <> pg_backend_pid();
    `);

    await adminClient.query(`DROP DATABASE IF EXISTS "${testDbName}"`);
    await adminClient.query(`CREATE DATABASE "${testDbName}"`);

    // Create test Knex instance
    testKnex = Knex({
        client: 'pg',
        connection: {
            host: config.POSTGRES_HOST,
            port: config.POSTGRES_PORT,
            user: config.POSTGRES_USER,
            password: config.POSTGRES_PASSWORD,
            database: testDbName,
        },
        migrations: {
            directory: './src/migrations',
        },
    });

    // Run migrations
    await testKnex.migrate.latest();

    // Replace the global knex instance
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const knexModule = require('../knex');
    knexModule.knex = testKnex;
};

/**
 * Truncate all tables before each test
 */
export const truncateTables = async (): Promise<void> => {
    if (!truncateQuery) {
        // Build and cache the truncate query
        const tables = await testKnex('pg_tables')
            .select('tablename')
            .where('schemaname', 'public')
            .whereNotIn('tablename', ['knex_migrations', 'knex_migrations_lock']);

        const tableNames = tables.map((t) => `"${t.tablename}"`).join(', ');
        if (tableNames) {
            truncateQuery = `TRUNCATE TABLE ${tableNames} RESTART IDENTITY CASCADE`;
        }
    }

    if (truncateQuery) {
        // Retry logic for deadlock handling
        const maxRetries = 3;
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                await testKnex.raw(truncateQuery);
                break;
            } catch (error) {
                if (attempt === maxRetries) {
                    throw error;
                }
                await new Promise((resolve) => setTimeout(resolve, 100 * attempt));
            }
        }
    }
};

/**
 * Cleanup test database after all tests
 */
export const teardownTestDatabase = async (): Promise<void> => {
    if (testKnex) {
        await testKnex.destroy();
    }

    if (adminClient) {
        await adminClient.query(`DROP DATABASE IF EXISTS "${testDbName}"`);
        await adminClient.end();
    }
};

/**
 * Get the test Knex instance
 */
export const getTestKnex = (): KnexType => testKnex;

// Re-export test utilities
export { expect, request };
