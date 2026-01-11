import { Server as HttpServer } from 'http';
import { Server as SocketServer } from 'socket.io';
import type { EventPayloads, SocketEvent } from '@blokus-ai/shared';

let io: SocketServer | null = null;

export const initializeSocket = (httpServer: HttpServer): SocketServer => {
    io = new SocketServer(httpServer, {
        cors: {
            origin: process.env.FRONTEND_URL || 'http://localhost:5174',
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
