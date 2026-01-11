import Knex from 'knex';
import { config as appConfig } from './config';

export const knex = Knex({
    client: 'pg',
    connection: process.env.DATABASE_URL || {
        host: appConfig.POSTGRES_HOST,
        port: appConfig.POSTGRES_PORT,
        user: appConfig.POSTGRES_USER,
        password: appConfig.POSTGRES_PASSWORD,
        database: appConfig.POSTGRES_DATABASE,
    },
    migrations: {
        directory: './src/migrations',
        extension: 'ts',
    },
    asyncStackTraces: true,
});
