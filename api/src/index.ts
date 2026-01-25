import { createServer } from 'http';
import express from 'express';
import cors from 'cors';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { config } from './config';
import { knex, initDatabase } from './knex';
import { pingRouter } from './features/ping/router';
import { debugRouter } from './features/debug/router';
import { mlRouter } from './features/ml/router';
import { debugPath } from '@blokus-ai/shared';
import { cidMiddleware } from './middleware/cid';
import { errorHandler } from './error';
import { initializeSocket } from './lib/socket';
import { startCDCConsumer } from './lib/cdc-consumer';
import { log, expressLogger } from './lib/log';

const app = express();
const httpServer = createServer(app);

// Trust proxy for rate limiting behind nginx
// app.set('trust proxy', 1);

// Initialize database and run migrations on startup
const initDatabaseAndMigrate = async () => {
    if (!config.POSTGRES_HOST) {
        log.error('Database configuration is required. Set POSTGRES_HOST environment variable.');
        process.exit(1);
    }
    try {
        await initDatabase();
        log.info('Running database migrations...');
        await knex.migrate.latest();
        log.info('Migrations complete');
    } catch (err) {
        log.error('Database initialization failed:', err);
        process.exit(1);
    }
};

// Initialize WebSocket
initializeSocket(httpServer);

// Rate limiting
const limiter = rateLimit({
    windowMs: 60 * 1000,
    limit: 500,
    message: 'Too many requests, please try again after 1 minute',
});

// CORS configuration
const corsOrigin = (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
    // Allow requests with no origin (mobile apps, curl)
    if (!origin) {
        return callback(null, true);
    }
    // Allow configured frontend URL
    if (origin === config.FRONTEND_URL) {
        return callback(null, true);
    }
    // In dev mode, allow localhost, IP-based origins, and tunnels
    if (config.NODE_ENV === 'dev') {
        const isIPOrigin = /^https?:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$/.test(origin);
        const isLocalhost = origin.includes('localhost') || origin.includes('127.0.0.1');
        const isTunnel = origin.includes('trycloudflare.com') || origin.includes('ngrok-free.app') || origin.includes('ngrok.io') || origin.includes('hycapital.io');
        if (isIPOrigin || isLocalhost || isTunnel) {
            return callback(null, true);
        }
    }
    callback(new Error('Not allowed by CORS'));
};

// Middleware
// app.use(limiter);
app.use(compression());
app.use(expressLogger);
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(cors({ origin: corsOrigin, credentials: true }));
app.use(cidMiddleware);

// Routes
app.use('/ping', pingRouter);
app.use(debugPath, debugRouter);
app.use('/api/ml', mlRouter);

// Error handler (must be last)
app.use(errorHandler);

// Start server
const start = async () => {
    await initDatabaseAndMigrate();

    httpServer.listen(config.PORT, () => {
        log.info(`API running on http://localhost:${config.PORT}`);

        // Start CDC consumer after server is ready
        startCDCConsumer().catch((err) => {
            log.error('Failed to start CDC consumer:', err);
        });
    });
};

start();

// Graceful shutdown
process.on('SIGTERM', () => {
    log.info('SIGTERM received, shutting down...');
    httpServer.close(() => {
        log.info('Server closed');
        process.exit(0);
    });
});
