import { DebugLogType, DebugPayload, debugPath } from '@blokus-ai/shared';

const API_URL = import.meta.env.VITE_API_URL || '';

export const DEBUG_API = {
    log: (type: DebugLogType, message: string, data?: unknown): void => {
        const payload: DebugPayload = {
            type,
            message,
            data,
            timestamp: new Date().toISOString(),
        };
        // Fire and forget - use fetch directly to avoid circular dependency
        fetch(`${API_URL}${debugPath}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        }).catch(() => {});
    },
};

export const debug = {
    info: (message: string, data?: unknown) => {
        console.log(message, data);
        DEBUG_API.log('info', message, data);
    },
    error: (message: string, data?: unknown) => {
        console.error(message, data);
        DEBUG_API.log('error', message, data);
    },
};
