import { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'crypto';

declare global {
    // eslint-disable-next-line @typescript-eslint/no-namespace
    namespace Express {
        interface Request {
            cid: string;
        }
    }
}

export const cidMiddleware = (req: Request, res: Response, next: NextFunction) => {
    req.cid = req.headers['x-correlation-id'] as string || randomUUID();
    res.setHeader('X-Correlation-ID', req.cid);
    next();
};
