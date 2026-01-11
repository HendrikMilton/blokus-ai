import { io, Socket } from 'socket.io-client';
import { toast } from 'react-hot-toast';

// API base URL
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

// Storage keys
export const storageKeys = {
    accessToken: 'accessToken',
    refreshToken: 'refreshToken',
};

// Generate correlation ID for request tracing
const generateCid = (): string => {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
};

// Request cache for deduplication
const pendingRequests = new Map<string, Promise<unknown>>();
const REQUEST_TIMEOUT_MS = 120_000;

// Socket.IO instance
export let apiSocket: Socket | null = null;

// Initialize socket connection
export const initializeSocket = () => {
    if (apiSocket) {
        return apiSocket;
    }

    apiSocket = io(API_URL, {
        withCredentials: true,
        autoConnect: false,
    });

    apiSocket.on('connect', () => {
        console.log('Socket connected');
        joinUserRoom();
    });

    apiSocket.on('disconnect', () => {
        console.log('Socket disconnected');
    });

    return apiSocket;
};

// Join user-specific room for targeted events
export const joinUserRoom = () => {
    const token = localStorage.getItem(storageKeys.accessToken);
    if (!token || !apiSocket) {
        return;
    }

    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        if (payload?.sub) {
            apiSocket.emit('joinRoom', `user-${payload.sub}`);
        }
    } catch {
        // Invalid token format
    }
};

// Connect socket (call after login)
export const connectSocket = () => {
    if (!apiSocket) {
        initializeSocket();
    }
    apiSocket?.connect();
};

// Disconnect socket (call on logout)
export const disconnectSocket = () => {
    apiSocket?.disconnect();
};

// HTTP client
const createHttp = () => {
    const makeRequest = async <T>(
        path: string,
        method: string,
        body?: unknown
    ): Promise<T> => {
        const cid = generateCid();
        const url = `${API_URL}${path}`;

        // Deduplicate identical GET requests
        if (method === 'GET') {
            const cacheKey = `${method}:${url}`;
            const pending = pendingRequests.get(cacheKey);
            if (pending) {
                return pending as Promise<T>;
            }
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
            'X-Correlation-ID': cid,
        };

        const token = localStorage.getItem(storageKeys.accessToken);
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const requestPromise = (async () => {
            try {
                const response = await fetch(url, {
                    method,
                    headers,
                    body: body ? JSON.stringify(body) : undefined,
                    signal: controller.signal,
                    credentials: 'include',
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    const message = errorData.message || `Request failed with status ${response.status}`;

                    // Show error toast with CID for support
                    toast.error(
                        `${message}\n\nCID: ${cid}`,
                        { duration: 5000 }
                    );

                    throw new Error(message);
                }

                // Parse JSON response (handles empty responses like 204 No Content)
                let responseBody: T | undefined;
                try {
                    responseBody = await response.json() as T;
                } catch { /* empty response */ }

                return responseBody as T;
            } finally {
                if (method === 'GET') {
                    pendingRequests.delete(`${method}:${url}`);
                }
            }
        })();

        if (method === 'GET') {
            pendingRequests.set(`${method}:${url}`, requestPromise);
        }

        return requestPromise;
    };

    return {
        get: <T>(path: string) => makeRequest<T>(path, 'GET'),
        post: <T>(path: string, body?: unknown) => makeRequest<T>(path, 'POST', body),
        put: <T>(path: string, body?: unknown) => makeRequest<T>(path, 'PUT', body),
        patch: <T>(path: string, body?: unknown) => makeRequest<T>(path, 'PATCH', body),
        delete: <T>(path: string) => makeRequest<T>(path, 'DELETE'),
    };
};

export const http = createHttp();
