import Knex from 'knex';
import { Client } from 'pg';
import { config as appConfig } from './config';

export const knexConfig = {
    client: 'pg',
    connection: {
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
};

export const knex = Knex(knexConfig);

export const initDatabase = async () => {
    const client = new Client({
        host: appConfig.POSTGRES_HOST,
        port: appConfig.POSTGRES_PORT,
        user: appConfig.POSTGRES_USER,
        password: appConfig.POSTGRES_PASSWORD,
        database: 'postgres',
    });
    await client.connect();
    try {
        await client.query(`CREATE DATABASE ${appConfig.POSTGRES_DATABASE}`);
        console.log(`Database ${appConfig.POSTGRES_DATABASE} created`);
    } catch (err) {
        if ((err as { code: string }).code !== '42P04') {
            throw err;
        }
        console.log(`Database ${appConfig.POSTGRES_DATABASE} already exists`);
    }
    await client.end();
};
