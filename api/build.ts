import { build } from 'bun';
import unpluginTypia from '@ryoppippi/unplugin-typia/bun';

await build({
    entrypoints: ['./src/index.ts'],
    outdir: './dist',
    target: 'node',
    plugins: [unpluginTypia()],
    external: [
        // pg-native is optional
        'pg-native',
        // knex tries to require all these database drivers
        'better-sqlite3',
        'mysql2',
        'sqlite3',
        'mysql',
        'pg-query-stream',
        'oracledb',
        'tedious',
    ],
});

console.log('Build complete!');
