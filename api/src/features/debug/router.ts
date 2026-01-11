import { Router } from 'express';
import { log } from '../../lib/log';
import { DebugPayload } from '@blokus-ai/shared';

export const debugRouter = Router();

debugRouter.post('/', (req, res) => {
    const { type, message, data, timestamp } = req.body as DebugPayload;
    const userAgent = req.headers['user-agent'] || 'unknown';

    const logData = {
        source: 'frontend',
        userAgent,
        timestamp: timestamp || new Date().toISOString(),
        data,
    };

    if (type === 'error') {
        log.error(`[FRONTEND] ${message}`, logData);
    } else {
        log.info(`[FRONTEND] ${message}`, logData);
    }

    res.status(204).send();
});
