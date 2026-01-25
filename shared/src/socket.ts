// WebSocket event types and payloads
import type { SafeUser } from './user';
import type { TrainingRun } from './ml';

// Socket event names
export type SocketEvent =
    | 'user-created'
    | 'user-updated'
    | 'user-deleted'
    | 'training-run-started'
    | 'training-run-progress'
    | 'training-run-completed'
    | 'training-run-failed';

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

export interface TrainingRunStartedPayload {
    run: TrainingRun;
}

export interface TrainingRunProgressPayload {
    runId: string;
    epoch: number;
    totalEpochs: number;
    loss: number;
    accuracy: number;
}

export interface TrainingRunCompletedPayload {
    run: TrainingRun;
}

export interface TrainingRunFailedPayload {
    run: TrainingRun;
}

// Discriminated union of all event payloads
export type EventPayloads = {
    'user-created': UserCreatedPayload;
    'user-updated': UserUpdatedPayload;
    'user-deleted': UserDeletedPayload;
    'training-run-started': TrainingRunStartedPayload;
    'training-run-progress': TrainingRunProgressPayload;
    'training-run-completed': TrainingRunCompletedPayload;
    'training-run-failed': TrainingRunFailedPayload;
};

// Room names
export const getUserRoom = (userId: number): string => `user-${userId}`;
export const TRAINING_ROOM = 'training';
