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
    | 'corner-run-failed'
    | 'grid-run-started'
    | 'grid-run-progress'
    | 'grid-run-completed'
    | 'grid-run-failed'
    | 'alphazero-progress'
    | 'alphazero-run-started'
    | 'alphazero-run-stopped'
    | 'alphazero-run-completed'
    | 'alphazero-run-failed'
    | 'alphazero-run-deleted'
    | 'alphazero-checkpoint-created';

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

// Grid prediction run event payloads
export interface GridRunStartedPayload {
    totalCount: number;
}

export interface GridRunProgressPayload {
    processedCount: number;
    totalCount: number;
    successCount: number;
    failedCount: number;
}

export interface GridRunCompletedPayload {
    successCount: number;
    failedCount: number;
    durationMs: number;
}

export interface GridRunFailedPayload {
    errorMessage: string;
}

// AlphaZero training event payloads
export interface AlphaZeroProgressPayload {
    runId: string;
    // Current state
    generation: number;
    maxGenerations: number;
    gamesGenerated: number;
    trainingSteps: number;
    policyLoss: number | null;
    valueLoss: number | null;
    // Phase info
    phase: 'self_play' | 'training' | 'evaluation';
    phaseCurrent: number;
    phaseTotal: number;
    phaseProgress: number; // 0-100
    phaseEtaSec: number;
    // Generation info
    generationProgress: number; // 0-100
    generationEtaSec: number;
    // Overall info
    overallProgress: number; // 0-100
    overallEtaSec: number;
    // Human-readable message
    message: string;
    // Extra metrics
    gamesPerMin?: number;
    avgGameTimeSec?: number;
    lastGameMoves?: number;
    stepsPerSec?: number;
    learningRate?: number;
    totalLoss?: number;
    evalWins?: number;
    evalGames?: number;
    currentWinRate?: number;
    promotionThreshold?: number;
}

export interface AlphaZeroRunStartedPayload {
    runId: string;
    name: string;
}

export interface AlphaZeroRunStoppedPayload {
    run: {
        id: string;
        name: string;
        status: string;
    };
}

export interface AlphaZeroRunCompletedPayload {
    run: {
        id: string;
        name: string;
        status: string;
        policyLoss: number;
        valueLoss: number;
        bestWinRate: number;
    };
}

export interface AlphaZeroRunFailedPayload {
    runId: string;
    errorMessage: string;
}

export interface AlphaZeroRunDeletedPayload {
    runId: string;
}

export interface AlphaZeroCheckpointCreatedPayload {
    checkpoint: {
        id: string;
        runId: string;
        generation: number;
        step: number;
        policyLoss: number | null;
        valueLoss: number | null;
        evalWinRate: number | null;
        isBest: boolean;
        modelPath: string;
    };
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
    'grid-run-started': GridRunStartedPayload;
    'grid-run-progress': GridRunProgressPayload;
    'grid-run-completed': GridRunCompletedPayload;
    'grid-run-failed': GridRunFailedPayload;
    'alphazero-progress': AlphaZeroProgressPayload;
    'alphazero-run-started': AlphaZeroRunStartedPayload;
    'alphazero-run-stopped': AlphaZeroRunStoppedPayload;
    'alphazero-run-completed': AlphaZeroRunCompletedPayload;
    'alphazero-run-failed': AlphaZeroRunFailedPayload;
    'alphazero-run-deleted': AlphaZeroRunDeletedPayload;
    'alphazero-checkpoint-created': AlphaZeroCheckpointCreatedPayload;
};

// Room names
export const getUserRoom = (userId: number): string => `user-${userId}`;
export const TRAINING_ROOM = 'training';
