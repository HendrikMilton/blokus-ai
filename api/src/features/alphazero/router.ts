import { Router } from 'express';
import { asyncHandler } from '../../lib/async-handler';
import { mlHttp } from '../../lib/http';
import { BadRequestError, NotFoundError } from '../../error';
import { emitEvent } from '../../lib/socket';
import { log } from '../../lib/log';
import {
    createAlphaZeroRun,
    updateAlphaZeroRun,
    getAlphaZeroRun,
    listAlphaZeroRuns,
    deleteAlphaZeroRun,
    createAlphaZeroCheckpoint,
    getAlphaZeroCheckpoint,
    listAlphaZeroCheckpoints,
    getBestCheckpoint,
    setCheckpointAsBest,
    getActiveAlphaZeroRun,
    getAllBestCheckpoints,
    createAlphaZeroMetricsBatch,
    listAlphaZeroMetrics,
    AlphaZeroConfig,
    CreateMetricData,
} from './database';

export const alphaZeroRouter = Router();

// Default training configuration
const DEFAULT_CONFIG: AlphaZeroConfig = {
    // Self-play
    gamesPerGeneration: 100,
    mctsSimulations: 400,
    mctsCPuct: 1.5,
    temperatureThreshold: 20,
    dirichletAlpha: 0.3,
    dirichletEpsilon: 0.25,
    numParallelGames: 8,
    // Training - optimized for 32GB VRAM
    batchSize: 4096,
    learningRate: 0.002,
    lrDecay: 'cosine',
    trainingStepsPerGen: 2000,
    replayBufferSize: 1_000_000,
    mixedPrecision: true,
    // Network
    numResBlocks: 15,
    channels: 256,
    // Evaluation
    evalGames: 100,
    promotionThreshold: 0.55,
    // Limits
    maxGenerations: 100,
};

// ============================================================================
// Training Run Routes
// ============================================================================

// POST /api/ml/alphazero/runs - Start a new AlphaZero training run
alphaZeroRouter.post('/runs', asyncHandler(async (req, res) => {
    const { name, config: userConfig } = req.body as {
        name: string;
        config?: Partial<AlphaZeroConfig>;
    };

    if (!name || typeof name !== 'string' || name.trim().length === 0) {
        throw new BadRequestError('name is required');
    }

    // Check for existing active run
    const activeRun = await getActiveAlphaZeroRun();
    if (activeRun) {
        throw new BadRequestError(`Training run "${activeRun.name}" is already running. Stop it first.`);
    }

    // Merge user config with defaults
    const config: AlphaZeroConfig = {
        ...DEFAULT_CONFIG,
        ...userConfig,
    };

    // Create the run
    const run = await createAlphaZeroRun(name.trim(), config);

    // Update to running status
    const runningRun = await updateAlphaZeroRun(run.id, {
        status: 'running',
        startedAt: new Date(),
    });

    // Emit start event
    emitEvent('alphazero-run-started', { runId: run.id, name });
    log.info(`AlphaZero training run started: ${run.id} (${name})`);

    // Return immediately - training runs in background
    res.json(runningRun);

    // Fire-and-forget: ML service handles the training loop
    mlHttp.post('/alphazero/train', {
        run_id: run.id,
        config,
    }).catch(err => {
        log.warn(`Initial AlphaZero training request failed: ${err}`);
    });
}));

// GET /api/ml/alphazero/runs - List all training runs
alphaZeroRouter.get('/runs', asyncHandler(async (_req, res) => {
    const runs = await listAlphaZeroRuns();
    res.json({ runs });
}));

// GET /api/ml/alphazero/runs/:id - Get a single training run
alphaZeroRouter.get('/runs/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getAlphaZeroRun(id);
    if (!run) {
        throw new NotFoundError('Training run not found');
    }

    res.json(run);
}));

// POST /api/ml/alphazero/runs/:id/stop - Stop a running training
alphaZeroRouter.post('/runs/:id/stop', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getAlphaZeroRun(id);
    if (!run) {
        throw new NotFoundError('Training run not found');
    }

    if (run.status !== 'running') {
        throw new BadRequestError('Can only stop running training');
    }

    // Signal ML service to stop
    try {
        await mlHttp.post(`/alphazero/stop/${id}`, {});
    } catch (err) {
        log.warn(`Failed to signal ML service to stop: ${err}`);
    }

    // Update status
    const stoppedRun = await updateAlphaZeroRun(id, {
        status: 'stopped',
        completedAt: new Date(),
    });

    emitEvent('alphazero-run-stopped', { run: stoppedRun! });
    log.info(`AlphaZero training run stopped: ${id}`);

    res.json(stoppedRun);
}));

// DELETE /api/ml/alphazero/runs/:id - Delete a training run
alphaZeroRouter.delete('/runs/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getAlphaZeroRun(id);
    if (!run) {
        throw new NotFoundError('Training run not found');
    }

    // Stop if running
    if (run.status === 'running') {
        try {
            await mlHttp.post(`/alphazero/stop/${id}`, {});
        } catch (err) {
            log.warn(`Failed to stop running training: ${err}`);
        }
    }

    // Delete model files via ML service
    try {
        await mlHttp.delete(`/alphazero/models/${id}`);
    } catch (err) {
        log.warn(`Failed to delete model files: ${err}`);
    }

    // Delete from database (cascade deletes checkpoints)
    await deleteAlphaZeroRun(id);

    emitEvent('alphazero-run-deleted', { runId: id });
    log.info(`AlphaZero training run deleted: ${id}`);

    res.json({ success: true });
}));

// ============================================================================
// ML Service Callbacks
// ============================================================================

// POST /api/ml/alphazero/runs/:id/progress - Progress update from ML service
alphaZeroRouter.post('/runs/:id/progress', asyncHandler(async (req, res) => {
    const { id } = req.params;
    // Accept all fields from ML service (already camelCase from Python)
    const progress = req.body;

    // Update the run record with key metrics
    await updateAlphaZeroRun(id, {
        currentGeneration: progress.generation,
        gamesGenerated: progress.gamesGenerated,
        trainingSteps: progress.trainingSteps,
        policyLoss: progress.policyLoss,
        valueLoss: progress.valueLoss,
    });

    // Emit full progress event with all details
    emitEvent('alphazero-progress', {
        runId: id,
        // Current state
        generation: progress.generation,
        maxGenerations: progress.maxGenerations,
        gamesGenerated: progress.gamesGenerated,
        trainingSteps: progress.trainingSteps,
        policyLoss: progress.policyLoss,
        valueLoss: progress.valueLoss,
        // Phase info
        phase: progress.phase,
        phaseCurrent: progress.phaseCurrent,
        phaseTotal: progress.phaseTotal,
        phaseProgress: progress.phaseProgress,
        phaseEtaSec: progress.phaseEtaSec,
        // Generation info
        generationProgress: progress.generationProgress,
        generationEtaSec: progress.generationEtaSec,
        // Overall info
        overallProgress: progress.overallProgress,
        overallEtaSec: progress.overallEtaSec,
        // Human-readable message
        message: progress.message,
        // Extra metrics (gamesPerMin, stepsPerSec, etc.)
        ...progress,
    });

    res.json({ success: true });
}));

// POST /api/ml/alphazero/runs/:id/checkpoint - New checkpoint created
alphaZeroRouter.post('/runs/:id/checkpoint', asyncHandler(async (req, res) => {
    const { id: runId } = req.params;
    const {
        generation,
        step,
        policy_loss,
        value_loss,
        eval_win_rate,
        eval_games,
        is_best,
        model_path,
    } = req.body;

    const checkpoint = await createAlphaZeroCheckpoint({
        runId,
        generation,
        step,
        policyLoss: policy_loss,
        valueLoss: value_loss,
        evalWinRate: eval_win_rate,
        evalGames: eval_games,
        isBest: is_best,
        modelPath: model_path,
    });

    // If this is the new best, update the run's best win rate
    if (is_best && eval_win_rate !== undefined) {
        await updateAlphaZeroRun(runId, {
            bestWinRate: eval_win_rate,
        });
    }

    emitEvent('alphazero-checkpoint-created', { checkpoint });
    log.info(`AlphaZero checkpoint created: gen=${generation}, step=${step}, best=${is_best}`);

    res.json({ success: true, checkpoint });
}));

// POST /api/ml/alphazero/runs/:id/complete - Training completed
alphaZeroRouter.post('/runs/:id/complete', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { policy_loss, value_loss, best_win_rate } = req.body;

    const run = await updateAlphaZeroRun(id, {
        status: 'completed',
        completedAt: new Date(),
        policyLoss: policy_loss,
        valueLoss: value_loss,
        bestWinRate: best_win_rate,
    });

    emitEvent('alphazero-run-completed', {
        run: {
            id: run!.id,
            name: run!.name,
            status: run!.status,
            policyLoss: run!.policyLoss ?? 0,
            valueLoss: run!.valueLoss ?? 0,
            bestWinRate: run!.bestWinRate ?? 0,
        },
    });
    log.info(`AlphaZero training run completed: ${id}`);

    res.json({ success: true });
}));

// POST /api/ml/alphazero/runs/:id/failed - Training failed
alphaZeroRouter.post('/runs/:id/failed', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { error_message } = req.body;

    await updateAlphaZeroRun(id, {
        status: 'failed',
        completedAt: new Date(),
        errorMessage: error_message,
    });

    emitEvent('alphazero-run-failed', { runId: id, errorMessage: error_message ?? 'Unknown error' });
    log.error(`AlphaZero training run failed: ${id}`, { error: error_message });

    res.json({ success: true });
}));

// POST /api/ml/alphazero/runs/:id/metrics - Batch insert training metrics
alphaZeroRouter.post('/runs/:id/metrics', asyncHandler(async (req, res) => {
    const { id: runId } = req.params;
    const { metrics } = req.body as { metrics: Array<{
        generation: number;
        step: number;
        policy_loss?: number;
        value_loss?: number;
        total_loss?: number;
        learning_rate?: number;
        grad_norm?: number;
        avg_corners?: number;
        avg_mobility?: number;
        center_coverage?: number;
        games_per_sec?: number;
        moves_per_sec?: number;
        replay_buffer_size?: number;
        step_time_ms?: number;
        policy_entropy?: number;
        policy_top1?: number;
        policy_top5?: number;
    }> };

    if (!metrics || !Array.isArray(metrics)) {
        throw new BadRequestError('metrics array is required');
    }

    // Convert snake_case from Python to camelCase for database function
    const metricData: CreateMetricData[] = metrics.map(m => ({
        runId,
        generation: m.generation,
        step: m.step,
        policyLoss: m.policy_loss,
        valueLoss: m.value_loss,
        totalLoss: m.total_loss,
        learningRate: m.learning_rate,
        gradNorm: m.grad_norm,
        avgCorners: m.avg_corners,
        avgMobility: m.avg_mobility,
        centerCoverage: m.center_coverage,
        gamesPerSec: m.games_per_sec,
        movesPerSec: m.moves_per_sec,
        replayBufferSize: m.replay_buffer_size,
        stepTimeMs: m.step_time_ms,
        policyEntropy: m.policy_entropy,
        policyTop1: m.policy_top1,
        policyTop5: m.policy_top5,
    }));

    await createAlphaZeroMetricsBatch(metricData);

    res.json({ success: true, count: metrics.length });
}));

// GET /api/ml/alphazero/runs/:id/metrics - Get training metrics for a run
alphaZeroRouter.get('/runs/:id/metrics', asyncHandler(async (req, res) => {
    const { id: runId } = req.params;
    const { generation, limit } = req.query;

    const metrics = await listAlphaZeroMetrics(runId, {
        generation: generation ? parseInt(generation as string, 10) : undefined,
        limit: limit ? parseInt(limit as string, 10) : undefined,
    });

    res.json({ metrics });
}));

// ============================================================================
// Checkpoint Routes
// ============================================================================

// GET /api/ml/alphazero/checkpoints - List checkpoints (optionally by run)
alphaZeroRouter.get('/checkpoints', asyncHandler(async (req, res) => {
    const { runId, bestOnly } = req.query;

    if (bestOnly === 'true') {
        const checkpoints = await getAllBestCheckpoints();
        res.json({ checkpoints });
        return;
    }

    const checkpoints = await listAlphaZeroCheckpoints(
        typeof runId === 'string' ? runId : undefined
    );
    res.json({ checkpoints });
}));

// GET /api/ml/alphazero/checkpoints/:id - Get a single checkpoint
alphaZeroRouter.get('/checkpoints/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const checkpoint = await getAlphaZeroCheckpoint(id);
    if (!checkpoint) {
        throw new NotFoundError('Checkpoint not found');
    }

    res.json(checkpoint);
}));

// POST /api/ml/alphazero/checkpoints/:id/activate - Set checkpoint as best/active
alphaZeroRouter.post('/checkpoints/:id/activate', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const checkpoint = await getAlphaZeroCheckpoint(id);
    if (!checkpoint) {
        throw new NotFoundError('Checkpoint not found');
    }

    await setCheckpointAsBest(id, checkpoint.runId);

    // Note: No socket event for checkpoint activation (not critical for UI)
    log.info(`AlphaZero checkpoint activated: ${id}`);

    res.json({ success: true });
}));

// GET /api/ml/alphazero/checkpoints/best/:runId - Get best checkpoint for a run
alphaZeroRouter.get('/checkpoints/best/:runId', asyncHandler(async (req, res) => {
    const { runId } = req.params;

    const checkpoint = await getBestCheckpoint(runId);
    if (!checkpoint) {
        throw new NotFoundError('No best checkpoint found for this run');
    }

    res.json(checkpoint);
}));

// ============================================================================
// Game/Move Routes (for playing against trained models)
// ============================================================================

// POST /api/ml/alphazero/move - Get AI move for a game state
alphaZeroRouter.post('/move', asyncHandler(async (req, res) => {
    const { gameState, checkpointId, mctsSimulations } = req.body as {
        gameState: unknown;
        checkpointId?: string;
        mctsSimulations?: number;
    };

    if (!gameState) {
        throw new BadRequestError('gameState is required');
    }

    // Determine which model to use
    let modelPath: string;
    if (checkpointId) {
        const checkpoint = await getAlphaZeroCheckpoint(checkpointId);
        if (!checkpoint) {
            throw new NotFoundError('Checkpoint not found');
        }
        modelPath = checkpoint.modelPath;
    } else {
        // Use the latest best checkpoint from any run
        const bestCheckpoints = await getAllBestCheckpoints();
        if (bestCheckpoints.length === 0) {
            throw new BadRequestError('No trained models available. Train a model first.');
        }
        modelPath = bestCheckpoints[0].modelPath;
    }

    // Call ML service to get move
    const response = await mlHttp.post('/alphazero/move', {
        game_state: gameState,
        model_path: modelPath,
        mcts_simulations: mctsSimulations ?? 400,
    });

    res.json(response);
}));

// GET /api/ml/alphazero/runs/:id/export/onnx - Export model to ONNX format
alphaZeroRouter.get('/runs/:id/export/onnx', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { checkpointId } = req.query;

    const run = await getAlphaZeroRun(id);
    if (!run) {
        throw new NotFoundError('Training run not found');
    }

    log.info(`Exporting ONNX model for run: ${id}`);

    // Call ML service to export
    const result = await mlHttp.post(`/alphazero/export/onnx/${id}`, {
        checkpoint_id: typeof checkpointId === 'string' ? checkpointId : undefined,
    });

    res.json(result);
}));

// GET /api/ml/alphazero/config/default - Get default training configuration
alphaZeroRouter.get('/config/default', asyncHandler(async (_req, res) => {
    res.json(DEFAULT_CONFIG);
}));
