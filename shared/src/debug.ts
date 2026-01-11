export type DebugLogType = 'info' | 'error';

export interface DebugPayload {
    type: DebugLogType;
    message: string;
    data?: unknown;
    timestamp: string;
}

export const debugPath = '/api/debug';
