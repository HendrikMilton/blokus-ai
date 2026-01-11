import { Request, Response, NextFunction } from 'express';
import { knex } from './knex';
import { config } from './config';

export class HttpError extends Error {
    constructor(public statusCode: number, message: string) {
        super(message);
    }
}

export class BadRequestError extends HttpError {
    constructor(message = 'Bad request') {
        super(400, message);
    }
}

export class UnauthorizedError extends HttpError {
    constructor(message = 'Unauthorized') {
        super(401, message);
    }
}

export class ForbiddenError extends HttpError {
    constructor(message = 'Forbidden') {
        super(403, message);
    }
}

export class NotFoundError extends HttpError {
    constructor(message = 'Not found') {
        super(404, message);
    }
}

/**
 * Log error to application_errors table
 */
const logErrorToDatabase = async (
    error: Error,
    cid: string,
    req: Request
) => {
    try {
        await knex('application_errors').insert({
            cid,
            error_message: error.message,
            error_stack: error.stack,
            request_method: req.method,
            request_path: req.path,
            request_body: JSON.stringify(req.body),
            user_id: (req as any).token?.sub || null,
            created_at: new Date(),
        });
    } catch (dbError) {
        console.error('Failed to log error to database:', dbError);
    }
};

/**
 * Send error notification to Telegram
 */
const notifyTelegram = async (error: Error, cid: string, req: Request) => {
    if (!config.TELEGRAM_BOT_TOKEN || !config.TELEGRAM_CHAT_ID) {
        return;
    }

    try {
        const message = `🚨 *Error in ${config.NODE_ENV}*

*CID:* \`${cid}\`
*Path:* ${req.method} ${req.path}
*Error:* ${error.message}

\`\`\`
${error.stack?.slice(0, 500)}
\`\`\``;

        await fetch(
            `https://api.telegram.org/bot${config.TELEGRAM_BOT_TOKEN}/sendMessage`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chat_id: config.TELEGRAM_CHAT_ID,
                    text: message,
                    parse_mode: 'Markdown',
                }),
            }
        );
    } catch (telegramError) {
        console.error('Failed to send Telegram notification:', telegramError);
    }
};

/**
 * Global error handler middleware
 */
export const errorHandler = async (
    err: Error,
    req: Request,
    res: Response,
    _next: NextFunction
) => {
    const cid = req.cid || 'unknown';

    console.error(`[${cid}] Error:`, err.message);
    console.error(err.stack);

    // Log to database and notify (don't await to not block response)
    if (config.NODE_ENV !== 'test') {
        logErrorToDatabase(err, cid, req).catch(() => {});

        // Only notify on 500 errors
        if (!(err instanceof HttpError)) {
            notifyTelegram(err, cid, req).catch(() => {});
        }
    }

    if (err instanceof HttpError) {
        return res.status(err.statusCode).json({
            message: err.message,
            cid,
        });
    }

    res.status(500).json({
        message: 'Internal server error',
        cid,
    });
};
