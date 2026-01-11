declare global {
    namespace Express {
        interface Request {
            cid: string;
        }
    }
}

export {};
