import type { Knex } from 'knex';

const config: Knex.Config = {
    client: 'pg',
    connection: process.env.DATABASE_URL || {
        host: 'localhost',
        port: 5432,
        user: 'dev',
        password: 'dev',
        database: 'blokus-ai',
    },
    migrations: {
        directory: './src/migrations',
        extension: 'ts',
    },
    seeds: {
        directory: './src/seeds',
        extension: 'ts',
    },
    asyncStackTraces: true,
};

export default config;
