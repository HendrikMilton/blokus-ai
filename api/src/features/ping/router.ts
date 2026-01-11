import { Router } from 'express';
import { knex } from '../../knex';
import { asyncHandler } from '../../lib/async-handler';

export const pingRouter = Router();

// Liveness probe - simple health check
pingRouter.get('/liveness', (_req, res) => {
    res.json({ message: 'pong' });
});

// Readiness probe - checks database connectivity
pingRouter.get('/readiness', asyncHandler(async (_req, res) => {
    try {
        await knex.raw('SELECT 1');
        res.json({ message: 'pong' });
    } catch (error) {
        res.status(503).json({ message: 'Database not ready' });
    }
}));
