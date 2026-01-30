// WebSocket event types and payloads
import type { SafeUser } from './user';
import type { TrainingRun, DetectionRun, CornerDetectionRun } from './ml';

// Socket event names
export type SocketEvent =
    | 'user-created'
    | 'user-updated'
    | 'user-deleted'
    | 'training-run-started'
    | 'training-run-progress'
    | 'training-run-completed'
    | 'training-run-failed'
    | 'training-run-deleted'
    | 'detection-run-started'
    | 'detection-run-progress'
    | 'detection-run-completed'
    | 'detection-run-failed'
    | 'detection-run-deleted'
    | 'corner-run-started'
    | 'corner-run-progress'
    | 'corner-run-completed'
    | 'corner-run-failed';

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
    phase: 'detection' | 'classifier';
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

export interface TrainingRunDeletedPayload {
    runId: string;
}

// Detection run event payloads
export interface DetectionRunStartedPayload {
    run: DetectionRun;
}

export interface DetectionRunProgressPayload {
    runId: string;
    processedCount: number;
    totalCount: number;
    successCount: number;
    failedCount: number;
}

export interface DetectionRunCompletedPayload {
    run: DetectionRun;
}

export interface DetectionRunFailedPayload {
    run: DetectionRun;
}

export interface DetectionRunDeletedPayload {
    runId: string;
}

// Corner detection run event payloads
export interface CornerRunStartedPayload {
    run: CornerDetectionRun;
}

export interface CornerRunProgressPayload {
    runId: string;
    processedCount: number;
    totalCount: number;
    successCount: number;
    failedCount: number;
}

export interface CornerRunCompletedPayload {
    run: CornerDetectionRun;
}

export interface CornerRunFailedPayload {
    run: CornerDetectionRun;
}

// Discriminated union of all event payloads
// Using Record<SocketEvent, ...> ensures all events have corresponding payloads
export type EventPayloads = Record<SocketEvent, unknown> & {
    'user-created': UserCreatedPayload;
    'user-updated': UserUpdatedPayload;
    'user-deleted': UserDeletedPayload;
    'training-run-started': TrainingRunStartedPayload;
    'training-run-progress': TrainingRunProgressPayload;
    'training-run-completed': TrainingRunCompletedPayload;
    'training-run-failed': TrainingRunFailedPayload;
    'training-run-deleted': TrainingRunDeletedPayload;
    'detection-run-started': DetectionRunStartedPayload;
    'detection-run-progress': DetectionRunProgressPayload;
    'detection-run-completed': DetectionRunCompletedPayload;
    'detection-run-failed': DetectionRunFailedPayload;
    'detection-run-deleted': DetectionRunDeletedPayload;
    'corner-run-started': CornerRunStartedPayload;
    'corner-run-progress': CornerRunProgressPayload;
    'corner-run-completed': CornerRunCompletedPayload;
    'corner-run-failed': CornerRunFailedPayload;
};

// Room names
export const getUserRoom = (userId: number): string => `user-${userId}`;
export const TRAINING_ROOM = 'training';
