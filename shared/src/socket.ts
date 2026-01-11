// WebSocket event types and payloads
import type { SafeUser } from './user';

// Socket event names
export type SocketEvent =
    | 'user-created'
    | 'user-updated'
    | 'user-deleted';

// Event payload types
export interface UserCreatedPayload {
    user: SafeUser;
}

export interface UserUpdatedPayload {
    user: SafeUser;
}

export interface UserDeletedPayload {
    userId: number;
}

// Discriminated union of all event payloads
export type EventPayloads = {
    'user-created': UserCreatedPayload;
    'user-updated': UserUpdatedPayload;
    'user-deleted': UserDeletedPayload;
};

// Room names
export const getUserRoom = (userId: number): string => `user-${userId}`;
