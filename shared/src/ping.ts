// Ping/health check types

export interface PingResponse {
    message: 'pong';
}

export interface ReadinessResponse {
    message: 'pong' | 'Database not ready';
}

// API path
export const pingPath = '/ping';
