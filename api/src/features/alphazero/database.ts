import { Knex } from 'knex';
import { knex } from '../../knex';

// ============================================================================
// Types
// ============================================================================

export type AlphaZeroRunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'stopped';

export interface AlphaZeroConfig {
    // Self-play
    gamesPerGeneration: number;
    mctsSimulations: number;
    mctsCPuct: number;
    temperatureThreshold: number;
    dirichletAlpha: number;
    dirichletEpsilon: number;
    numParallelGames: number;
    // Training
    batchSize: number;
    learningRate: number;
    lrDecay: string;
    trainingStepsPerGen: number;
    replayBufferSize: number;
    mixedPrecision: boolean;
    // Network
    numResBlocks: number;
    channels: number;
    // Evaluation
    evalGames: number;
    promotionThreshold: number;
    // Limits
    maxGenerations: number;
}

export interface AlphaZeroRun {
    id: string;
    name: string;
    status: AlphaZeroRunStatus;
    config: AlphaZeroConfig;
    currentGeneration: number;
    gamesGenerated: number;
    trainingSteps: number;
    policyLoss: number | null;
    valueLoss: number | null;
    bestWinRate: number | null;
    startedAt: string | null;
    completedAt: string | null;
    errorMessage: string | null;
    createdAt: string;
    updatedAt: string;
}

export interface AlphaZeroCheckpoint {
    id: string;
    runId: string;
    generation: number;
    step: number;
    policyLoss: number | null;
    valueLoss: number | null;
    evalWinRate: number | null;
    evalGames: number | null;
    isBest: boolean;
    modelPath: string;
    createdAt: string;
}

// ============================================================================
// Database Row Types
// ============================================================================

interface AlphaZeroRunRow {
    id: string;
    name: string;
    status: AlphaZeroRunStatus;
    config: AlphaZeroConfig;
    current_generation: number;
    games_generated: number;
    training_steps: number;
    policy_loss: number | null;
    value_loss: number | null;
    best_win_rate: number | null;
    started_at: Date | null;
    completed_at: Date | null;
    error_message: string | null;
    created_at: Date;
    updated_at: Date;
}

interface AlphaZeroCheckpointRow {
    id: string;
    run_id: string;
    generation: number;
    step: number;
    policy_loss: number | null;
    value_loss: number | null;
    eval_win_rate: number | null;
    eval_games: number | null;
    is_best: boolean;
    model_path: string;
    created_at: Date;
}

// ============================================================================
// Converters
// ============================================================================

const rowToRun = (row: AlphaZeroRunRow): AlphaZeroRun => ({
    id: row.id,
    name: row.name,
    status: row.status,
    config: row.config,
    currentGeneration: row.current_generation,
    gamesGenerated: row.games_generated,
    trainingSteps: row.training_steps,
    policyLoss: row.policy_loss,
    valueLoss: row.value_loss,
    bestWinRate: row.best_win_rate,
    startedAt: row.started_at?.toISOString() ?? null,
    completedAt: row.completed_at?.toISOString() ?? null,
    errorMessage: row.error_message,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
});

const rowToCheckpoint = (row: AlphaZeroCheckpointRow): AlphaZeroCheckpoint => ({
    id: row.id,
    runId: row.run_id,
    generation: row.generation,
    step: row.step,
    policyLoss: row.policy_loss,
    valueLoss: row.value_loss,
    evalWinRate: row.eval_win_rate,
    evalGames: row.eval_games,
    isBest: row.is_best,
    modelPath: row.model_path,
    createdAt: row.created_at.toISOString(),
});

// ============================================================================
// AlphaZero Runs CRUD
// ============================================================================

export const createAlphaZeroRun = async (
    name: string,
    config: AlphaZeroConfig,
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroRun> => {
    const [row] = await trx<AlphaZeroRunRow>('alphazero_runs')
        .insert({
            name,
            config: JSON.stringify(config) as unknown as AlphaZeroConfig,
        })
        .returning('*');
    return rowToRun(row);
};

export interface AlphaZeroRunUpdate {
    status?: AlphaZeroRunStatus;
    currentGeneration?: number;
    gamesGenerated?: number;
    trainingSteps?: number;
    policyLoss?: number;
    valueLoss?: number;
    bestWinRate?: number;
    startedAt?: Date;
    completedAt?: Date;
    errorMessage?: string;
}

export const updateAlphaZeroRun = async (
    id: string,
    updates: AlphaZeroRunUpdate,
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroRun | null> => {
    const dbUpdates: Partial<AlphaZeroRunRow> = {
        updated_at: new Date(),
    };

    if (updates.status !== undefined) {
        dbUpdates.status = updates.status;
    }
    if (updates.currentGeneration !== undefined) {
        dbUpdates.current_generation = updates.currentGeneration;
    }
    if (updates.gamesGenerated !== undefined) {
        dbUpdates.games_generated = updates.gamesGenerated;
    }
    if (updates.trainingSteps !== undefined) {
        dbUpdates.training_steps = updates.trainingSteps;
    }
    if (updates.policyLoss !== undefined) {
        dbUpdates.policy_loss = updates.policyLoss;
    }
    if (updates.valueLoss !== undefined) {
        dbUpdates.value_loss = updates.valueLoss;
    }
    if (updates.bestWinRate !== undefined) {
        dbUpdates.best_win_rate = updates.bestWinRate;
    }
    if (updates.startedAt !== undefined) {
        dbUpdates.started_at = updates.startedAt;
    }
    if (updates.completedAt !== undefined) {
        dbUpdates.completed_at = updates.completedAt;
    }
    if (updates.errorMessage !== undefined) {
        dbUpdates.error_message = updates.errorMessage;
    }

    const [row] = await trx<AlphaZeroRunRow>('alphazero_runs')
        .where('id', id)
        .update(dbUpdates)
        .returning('*');

    return row ? rowToRun(row) : null;
};

export const getAlphaZeroRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroRun | null> => {
    const row = await trx<AlphaZeroRunRow>('alphazero_runs')
        .where('id', id)
        .first();
    return row ? rowToRun(row) : null;
};

export const listAlphaZeroRuns = async (
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroRun[]> => {
    const rows = await trx<AlphaZeroRunRow>('alphazero_runs')
        .orderBy('created_at', 'desc');
    return rows.map(rowToRun);
};

export const deleteAlphaZeroRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('alphazero_runs')
        .where('id', id)
        .delete();
    return deleted > 0;
};

// ============================================================================
// AlphaZero Checkpoints CRUD
// ============================================================================

export interface CreateCheckpointData {
    runId: string;
    generation: number;
    step: number;
    policyLoss?: number | null;
    valueLoss?: number | null;
    evalWinRate?: number | null;
    evalGames?: number | null;
    isBest?: boolean;
    modelPath: string;
}

export const createAlphaZeroCheckpoint = async (
    data: CreateCheckpointData,
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroCheckpoint> => {
    const [row] = await trx<AlphaZeroCheckpointRow>('alphazero_checkpoints')
        .insert({
            run_id: data.runId,
            generation: data.generation,
            step: data.step,
            policy_loss: data.policyLoss ?? null,
            value_loss: data.valueLoss ?? null,
            eval_win_rate: data.evalWinRate ?? null,
            eval_games: data.evalGames ?? null,
            is_best: data.isBest ?? false,
            model_path: data.modelPath,
        })
        .returning('*');
    return rowToCheckpoint(row);
};

export const getAlphaZeroCheckpoint = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroCheckpoint | null> => {
    const row = await trx<AlphaZeroCheckpointRow>('alphazero_checkpoints')
        .where('id', id)
        .first();
    return row ? rowToCheckpoint(row) : null;
};

export const listAlphaZeroCheckpoints = async (
    runId?: string,
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroCheckpoint[]> => {
    let query = trx<AlphaZeroCheckpointRow>('alphazero_checkpoints');
    if (runId) {
        query = query.where('run_id', runId);
    }
    const rows = await query.orderBy('created_at', 'desc');
    return rows.map(rowToCheckpoint);
};

export const getBestCheckpoint = async (
    runId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroCheckpoint | null> => {
    const row = await trx<AlphaZeroCheckpointRow>('alphazero_checkpoints')
        .where('run_id', runId)
        .where('is_best', true)
        .orderBy('generation', 'desc')
        .first();
    return row ? rowToCheckpoint(row) : null;
};

export const getLatestCheckpoint = async (
    runId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroCheckpoint | null> => {
    const row = await trx<AlphaZeroCheckpointRow>('alphazero_checkpoints')
        .where('run_id', runId)
        .orderBy('generation', 'desc')
        .orderBy('step', 'desc')
        .first();
    return row ? rowToCheckpoint(row) : null;
};

export const setCheckpointAsBest = async (
    checkpointId: string,
    runId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    // Clear previous best
    await trx('alphazero_checkpoints')
        .where('run_id', runId)
        .where('is_best', true)
        .update({ is_best: false });

    // Set new best
    await trx('alphazero_checkpoints')
        .where('id', checkpointId)
        .update({ is_best: true });
};

export const deleteAlphaZeroCheckpoint = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('alphazero_checkpoints')
        .where('id', id)
        .delete();
    return deleted > 0;
};

// ============================================================================
// Utility Functions
// ============================================================================

export const getActiveAlphaZeroRun = async (
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroRun | null> => {
    const row = await trx<AlphaZeroRunRow>('alphazero_runs')
        .where('status', 'running')
        .first();
    return row ? rowToRun(row) : null;
};

export const getAllBestCheckpoints = async (
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroCheckpoint[]> => {
    const rows = await trx<AlphaZeroCheckpointRow>('alphazero_checkpoints')
        .where('is_best', true)
        .orderBy('created_at', 'desc');
    return rows.map(rowToCheckpoint);
};

// ============================================================================
// AlphaZero Metrics
// ============================================================================

export interface AlphaZeroMetric {
    id: string;
    runId: string;
    generation: number;
    step: number;
    policyLoss: number | null;
    valueLoss: number | null;
    totalLoss: number | null;
    learningRate: number | null;
    gradNorm: number | null;
    avgCorners: number | null;
    avgMobility: number | null;
    centerCoverage: number | null;
    gamesPerSec: number | null;
    movesPerSec: number | null;
    replayBufferSize: number | null;
    stepTimeMs: number | null;
    policyEntropy: number | null;
    policyTop1: number | null;
    policyTop5: number | null;
    createdAt: string;
}

interface AlphaZeroMetricRow {
    id: string;
    run_id: string;
    generation: number;
    step: number;
    policy_loss: number | null;
    value_loss: number | null;
    total_loss: number | null;
    learning_rate: number | null;
    grad_norm: number | null;
    avg_corners: number | null;
    avg_mobility: number | null;
    center_coverage: number | null;
    games_per_sec: number | null;
    moves_per_sec: number | null;
    replay_buffer_size: number | null;
    step_time_ms: number | null;
    policy_entropy: number | null;
    policy_top1: number | null;
    policy_top5: number | null;
    created_at: Date;
}

export interface CreateMetricData {
    runId: string;
    generation: number;
    step: number;
    policyLoss?: number | null;
    valueLoss?: number | null;
    totalLoss?: number | null;
    learningRate?: number | null;
    gradNorm?: number | null;
    avgCorners?: number | null;
    avgMobility?: number | null;
    centerCoverage?: number | null;
    gamesPerSec?: number | null;
    movesPerSec?: number | null;
    replayBufferSize?: number | null;
    stepTimeMs?: number | null;
    policyEntropy?: number | null;
    policyTop1?: number | null;
    policyTop5?: number | null;
}

export const createAlphaZeroMetric = async (
    data: CreateMetricData,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    await trx('alphazero_metrics').insert({
        run_id: data.runId,
        generation: data.generation,
        step: data.step,
        policy_loss: data.policyLoss ?? null,
        value_loss: data.valueLoss ?? null,
        total_loss: data.totalLoss ?? null,
        learning_rate: data.learningRate ?? null,
        grad_norm: data.gradNorm ?? null,
        avg_corners: data.avgCorners ?? null,
        avg_mobility: data.avgMobility ?? null,
        center_coverage: data.centerCoverage ?? null,
        games_per_sec: data.gamesPerSec ?? null,
        moves_per_sec: data.movesPerSec ?? null,
        replay_buffer_size: data.replayBufferSize ?? null,
        step_time_ms: data.stepTimeMs ?? null,
    });
};

export const createAlphaZeroMetricsBatch = async (
    metrics: CreateMetricData[],
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    if (metrics.length === 0) {
        return;
    }

    const rows = metrics.map(data => ({
        run_id: data.runId,
        generation: data.generation,
        step: data.step,
        policy_loss: data.policyLoss ?? null,
        value_loss: data.valueLoss ?? null,
        total_loss: data.totalLoss ?? null,
        learning_rate: data.learningRate ?? null,
        grad_norm: data.gradNorm ?? null,
        avg_corners: data.avgCorners ?? null,
        avg_mobility: data.avgMobility ?? null,
        center_coverage: data.centerCoverage ?? null,
        games_per_sec: data.gamesPerSec ?? null,
        moves_per_sec: data.movesPerSec ?? null,
        replay_buffer_size: data.replayBufferSize ?? null,
        step_time_ms: data.stepTimeMs ?? null,
        policy_entropy: data.policyEntropy ?? null,
        policy_top1: data.policyTop1 ?? null,
        policy_top5: data.policyTop5 ?? null,
    }));

    await trx('alphazero_metrics').insert(rows);
};

const rowToMetric = (row: AlphaZeroMetricRow): AlphaZeroMetric => ({
    id: row.id,
    runId: row.run_id,
    generation: row.generation,
    step: row.step,
    policyLoss: row.policy_loss,
    valueLoss: row.value_loss,
    totalLoss: row.total_loss,
    learningRate: row.learning_rate,
    gradNorm: row.grad_norm,
    avgCorners: row.avg_corners,
    avgMobility: row.avg_mobility,
    centerCoverage: row.center_coverage,
    gamesPerSec: row.games_per_sec,
    movesPerSec: row.moves_per_sec,
    replayBufferSize: row.replay_buffer_size,
    stepTimeMs: row.step_time_ms,
    policyEntropy: row.policy_entropy,
    policyTop1: row.policy_top1,
    policyTop5: row.policy_top5,
    createdAt: row.created_at.toISOString(),
});

export const listAlphaZeroMetrics = async (
    runId: string,
    options?: { generation?: number; limit?: number },
    trx: Knex.Transaction | Knex = knex
): Promise<AlphaZeroMetric[]> => {
    let query = trx<AlphaZeroMetricRow>('alphazero_metrics')
        .where('run_id', runId)
        .orderBy('generation', 'asc')
        .orderBy('step', 'asc');

    if (options?.generation !== undefined) {
        query = query.where('generation', options.generation);
    }
    if (options?.limit !== undefined) {
        query = query.limit(options.limit);
    }

    const rows = await query;
    return rows.map(rowToMetric);
};
