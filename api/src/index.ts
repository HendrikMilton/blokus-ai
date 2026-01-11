import { createServer } from 'http';
import express from 'express';
import cors from 'cors';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { config } from './config';
import { knex } from './knex';
import { pingRouter } from './features/ping/router';
import { usersRouter } from './features/users/router';
import { cidMiddleware } from './middleware/cid';
import { errorHandler } from './error';
import { initializeSocket } from './lib/socket';
import { startCDCConsumer } from './lib/cdc-consumer';

const app = express();
const httpServer = createServer(app);

// Run migrations on startup
const runMigrations = async () => {
    try {
        console.log('Running database migrations...');
        await knex.migrate.latest();
        console.log('Migrations complete');
    } catch (err) {
        console.error('Migration failed:', err);
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

// Middleware
app.use(limiter);
app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(
    cors({
        origin: config.FRONTEND_URL,
        credentials: true,
    })
);
app.use(cidMiddleware);

// Routes
app.use('/ping', pingRouter);
app.use('/users', usersRouter);

// Error handler (must be last)
app.use(errorHandler);

// Start server
const start = async () => {
    await runMigrations();

    httpServer.listen(config.PORT, () => {
        console.log(`API running on http://localhost:${config.PORT}`);

        // Start CDC consumer after server is ready
        startCDCConsumer().catch((err) => {
            console.error('Failed to start CDC consumer:', err);
        });
    });
};

start();

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down...');
    httpServer.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});
