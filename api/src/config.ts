import 'dotenv/config';

interface Config {
    NODE_ENV: 'dev' | 'prod' | 'test';
    PORT: number;

    // Database
    POSTGRES_HOST: string;
    POSTGRES_PORT: number;
    POSTGRES_USER: string;
    POSTGRES_PASSWORD: string;
    POSTGRES_DATABASE: string;

    // Frontend
    FRONTEND_URL: string;

    // ML Service
    ML_SERVICE_URL: string;

    // Telegram (optional - for error notifications)
    TELEGRAM_BOT_TOKEN: string;
    TELEGRAM_CHAT_ID: string;
}

const defaults: Config = {
    NODE_ENV: 'dev',
    PORT: 3000,

    POSTGRES_HOST: 'localhost',
    POSTGRES_PORT: 5432,
    POSTGRES_USER: 'c_connect_user',
    POSTGRES_PASSWORD: 'postgres',
    POSTGRES_DATABASE: 'blokus_ai',

    FRONTEND_URL: 'http://localhost:5174',

    ML_SERVICE_URL: 'http://localhost:8017',

    TELEGRAM_BOT_TOKEN: '',
    TELEGRAM_CHAT_ID: '',
};

const createConfig = (): Config => {
    const config = { ...defaults };

    for (const key of Object.keys(defaults) as (keyof Config)[]) {
        const envValue = process.env[key];
        if (envValue !== undefined) {
            const defaultValue = defaults[key];
            if (typeof defaultValue === 'number') {
                const parsed = parseInt(envValue, 10);
                if (!isNaN(parsed)) {
                    (config as any)[key] = parsed;
                }
            } else {
                (config as any)[key] = envValue;
            }
        }
    }

    return config;
};

export const config = createConfig();
