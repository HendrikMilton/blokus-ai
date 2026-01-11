import { Request, Response, NextFunction, RequestHandler } from 'express';

/**
 * Wrapper for async route handlers to properly catch errors and pass them to Express error middleware.
 * Without this, errors thrown in async handlers won't be caught by Express.
 */
export const asyncHandler = (
    fn: (req: Request, res: Response, next: NextFunction) => Promise<void>
): RequestHandler => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
