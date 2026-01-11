import { createLogger, format, transports, Logger } from 'winston';
import { logger } from 'express-winston';

export const log = createLogger({
    level: 'debug',
    format: format.combine(
        format.timestamp(),
        format.json()
    ),
    transports: [
        new transports.Console()
    ],
});

/**
 * Create a child logger with CID attached to all log entries
 */
export const logWithCid = (cid: string): Logger => {
    return log.child({ cid });
};

export const expressLogger = logger({
    winstonInstance: log,
    expressFormat: true,
    skip: function (req, _res) {
        return req.path === '/ping/liveness' || req.path === '/ping/readiness';
    }
});
