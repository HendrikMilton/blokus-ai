import { Server as HttpServer } from 'http';
import { Server as SocketServer } from 'socket.io';
import type { EventPayloads, SocketEvent } from '@blokus-ai/shared';

let io: SocketServer | null = null;

// CORS origin checker for socket.io (matches Express CORS config)
const checkOrigin = (origin: string | undefined): boolean => {
    if (!origin) return true;
    if (origin === process.env.FRONTEND_URL) return true;
    if (process.env.NODE_ENV === 'dev') {
        const isIPOrigin = /^https?:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$/.test(origin);
        const isLocalhost = origin.includes('localhost') || origin.includes('127.0.0.1');
        const isCloudflare = origin.includes('trycloudflare.com');
        if (isIPOrigin || isLocalhost || isCloudflare) return true;
    }
    return false;
};

export const initializeSocket = (httpServer: HttpServer): SocketServer => {
    io = new SocketServer(httpServer, {
        cors: {
            origin: (origin, callback) => {
                if (checkOrigin(origin)) {
                    callback(null, true);
                } else {
                    callback(new Error('Not allowed by CORS'));
                }
            },
            credentials: true,
        },
    });

    io.on('connection', (socket) => {
        console.log(`Socket connected: ${socket.id}`);

        // Handle room joining
        socket.on('joinRoom', (roomId: string) => {
            socket.join(roomId);
            console.log(`Socket ${socket.id} joined room: ${roomId}`);
        });

        // Handle room leaving
        socket.on('leaveRoom', (roomId: string) => {
            socket.leave(roomId);
            console.log(`Socket ${socket.id} left room: ${roomId}`);
        });

        socket.on('disconnect', () => {
            console.log(`Socket disconnected: ${socket.id}`);
        });
    });

    return io;
};

export const getSocketServer = (): SocketServer | null => io;

/**
 * Emit event to all connected clients
 */
export const emitEvent = <E extends SocketEvent>(
    eventName: E,
    data: EventPayloads[E]
): void => {
    if (!io) {
        console.warn('Socket.io not initialized');
        return;
    }
    io.emit(eventName, JSON.stringify(data));
};

/**
 * Emit event to specific users (by user IDs)
 */
export const emitEventToUsers = <E extends SocketEvent>(
    eventName: E,
    data: EventPayloads[E],
    userIds: number[]
): void => {
    if (!io) {
        console.warn('Socket.io not initialized');
        return;
    }

    userIds.forEach((userId) => {
        io!.to(`user-${userId}`).emit(eventName, JSON.stringify(data));
    });
};

/**
 * Emit event to a specific room
 */
export const emitEventToRoom = <E extends SocketEvent>(
    eventName: E,
    data: EventPayloads[E],
    roomId: string
): void => {
    if (!io) {
        console.warn('Socket.io not initialized');
        return;
    }
    io.to(roomId).emit(eventName, JSON.stringify(data));
};
