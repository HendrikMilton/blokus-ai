import { config } from '../config';

export class HttpError<T = unknown> {
    constructor(
        public status: number,
        public body: { message?: string; detail?: string; error?: string; details?: T }
    ) {}
}

export const createHttp = (baseUrl: string) => {
    const makeRequest = async <T>(
        path: string,
        method: string,
        body?: unknown,
        options?: { silent?: boolean; headers?: Record<string, string> }
    ): Promise<T> => {
        const fullUrl = `${baseUrl}${path}`;

        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
            ...options?.headers,
        };

        let response: Response;
        try {
            response = await fetch(fullUrl, {
                method,
                headers,
                ...(body !== undefined && { body: JSON.stringify(body) }),
            });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Network error';
            if (!options?.silent) {
                console.error(`[HTTP] ${method} ${fullUrl} failed:`, errorMessage);
            }
            throw error;
        }

        let responseBody: unknown;
        const contentType = response.headers.get('content-type');
        if (contentType?.includes('application/json')) {
            try {
                responseBody = await response.json();
            } catch { /* empty */ }
        } else {
            responseBody = await response.text();
        }

        if (!response.ok) {
            const error = new HttpError(response.status, responseBody as HttpError['body']);
            if (!options?.silent) {
                const errorMessage = error.body?.message || error.body?.detail || error.body?.error || `HTTP ${error.status}`;
                console.error(`[HTTP] ${method} ${fullUrl} error:`, errorMessage);
            }
            throw error;
        }

        return responseBody as T;
    };

    return {
        get: <T>(path: string, options?: { silent?: boolean; headers?: Record<string, string> }) =>
            makeRequest<T>(path, 'GET', undefined, options),
        post: <T>(path: string, body?: unknown, options?: { silent?: boolean; headers?: Record<string, string> }) =>
            makeRequest<T>(path, 'POST', body, options),
        patch: <T>(path: string, body?: unknown, options?: { silent?: boolean; headers?: Record<string, string> }) =>
            makeRequest<T>(path, 'PATCH', body, options),
        put: <T>(path: string, body?: unknown, options?: { silent?: boolean; headers?: Record<string, string> }) =>
            makeRequest<T>(path, 'PUT', body, options),
        delete: <T>(path: string, body?: unknown, options?: { silent?: boolean; headers?: Record<string, string> }) =>
            makeRequest<T>(path, 'DELETE', body, options),
    };
};

// Pre-configured ML service client
export const mlHttp = createHttp(config.ML_SERVICE_URL);
