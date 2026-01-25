import { Router } from 'express';
import { randomUUID } from 'crypto';
import { asyncHandler } from '../../lib/async-handler';
import { mlHttp } from '../../lib/http';
import type { DetectResponse } from '@blokus-ai/shared';
import { BadRequestError, NotFoundError } from '../../error';
import { getBoardDetectionWithCorrectionBySampleId, upsertBoardDetection, deleteBoardDetection, saveBoardDetectionCorrection, CorrectionData, setVerificationStatus, saveManualLabel, ManualLabelData, getSampleStatusFlags, getSampleStatusFlagsById, createTrainingRun, updateTrainingRun, listTrainingRuns, getTrainingRun, getHumanLabeledImageCount, getTrainingData } from './database';
import type { SampleStatusFlags } from '@blokus-ai/shared';
import { listSamples, getSample, getUploadPresignedUrl, getCloudFrontUrl, getThumbnailUrl, generateThumbnail } from './s3';
import { emitEvent } from '../../lib/socket';
import { log } from '../../lib/log';

export const mlRouter = Router();

// POST /api/ml/upload/presign - Get presigned URL for direct S3 upload
mlRouter.post('/upload/presign', asyncHandler(async (_req, res) => {
    const id = randomUUID();
    const { url, key } = await getUploadPresignedUrl(id);

    res.json({
        id,
        uploadUrl: url,
        key,
        imageUrl: getCloudFrontUrl(id),
        thumbnailUrl: getThumbnailUrl(id),
    });
}));

// POST /api/ml/upload/complete - Called after S3 upload to generate thumbnail
mlRouter.post('/upload/complete', asyncHandler(async (req, res) => {
    const { id } = req.body;

    if (!id) {
        throw new BadRequestError('id is required');
    }

    await generateThumbnail(id);
    res.json({ success: true, thumbnailUrl: getThumbnailUrl(id) });
}));

// POST /api/ml/upload - Upload board image (legacy, through API)
mlRouter.post('/upload', asyncHandler(async (req, res) => {
    const { image } = req.body;

    if (!image) {
        throw new BadRequestError('Image is required');
    }

    const data = await mlHttp.post('/upload/', { image });
    res.json(data);
}));

const defaultStatusFlags: SampleStatusFlags = {
    aiLabeled: false,
    humanLabeled: false,
    verified: false,
};

// GET /api/ml/samples - List all samples from S3
mlRouter.get('/samples', asyncHandler(async (_req, res) => {
    const s3Samples = await listSamples();
    const statusMap = await getSampleStatusFlags();

    const samples = s3Samples.map(s => {
        const status = statusMap.get(s.id) || defaultStatusFlags;

        return {
            id: s.id,
            filename: `${s.id}.png`,
            status,
            created_at: s.lastModified.toISOString(),
            imageUrl: getCloudFrontUrl(s.id),
            thumbnailUrl: getThumbnailUrl(s.id),
        };
    });

    res.json({ samples });
}));

// GET /api/ml/samples/:id - Get single sample
mlRouter.get('/samples/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const s3Sample = await getSample(id);
    if (!s3Sample) {
        throw new NotFoundError('Sample not found');
    }

    const status = await getSampleStatusFlagsById(id) || defaultStatusFlags;

    res.json({
        id: s3Sample.id,
        filename: `${s3Sample.id}.png`,
        status,
        created_at: s3Sample.lastModified.toISOString(),
        imageUrl: getCloudFrontUrl(s3Sample.id),
        thumbnailUrl: getThumbnailUrl(s3Sample.id),
    });
}));

// GET /api/ml/samples/:id/image - Redirect to CloudFront
mlRouter.get('/samples/:id/image', asyncHandler(async (req, res) => {
    const { id } = req.params;
    res.redirect(301, getCloudFrontUrl(id));
}));

// POST /api/ml/detect/:id - Run board detection and store result
mlRouter.post('/detect/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { model = 'yolo' } = req.body;

    const data = await mlHttp.post<DetectResponse>(`/detect/${id}?model=${model}`, {});
    await upsertBoardDetection(data);
    res.json(data);
}));

// GET /api/ml/detect/:id - Get cached detection result with correction
mlRouter.get('/detect/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const cached = await getBoardDetectionWithCorrectionBySampleId(id);
    res.json(cached);
}));

// DELETE /api/ml/detect/:id - Delete detection result
mlRouter.delete('/detect/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const deleted = await deleteBoardDetection(id);
    res.json({ deleted });
}));

// PATCH /api/ml/detect/:id/correction - Save correction
mlRouter.patch('/detect/:id/correction', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const correction = req.body as CorrectionData;

    const updated = await saveBoardDetectionCorrection(id, correction);
    if (!updated) {
        throw new NotFoundError('Detection not found');
    }
    res.json({ success: true });
}));

// POST /api/ml/detect/:id/verify - Verify detection
mlRouter.post('/detect/:id/verify', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const updated = await setVerificationStatus(id, true);
    if (!updated) {
        throw new NotFoundError('Detection not found');
    }
    res.json({ success: true });
}));

// DELETE /api/ml/detect/:id/verify - Unverify detection
mlRouter.delete('/detect/:id/verify', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const updated = await setVerificationStatus(id, false);
    if (!updated) {
        throw new NotFoundError('Detection not found');
    }
    res.json({ success: true });
}));

// POST /api/ml/samples/:id/label - Save manual label (for data labeling without ML detection)
mlRouter.post('/samples/:id/label', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const label = req.body as ManualLabelData;

    await saveManualLabel(id, label);
    res.json({ success: true });
}));

// ============================================================================
// Training Routes
// ============================================================================

// GET /api/ml/training-data - Get human-labeled samples for training
mlRouter.get('/training-data', asyncHandler(async (_req, res) => {
    const samples = await getTrainingData();
    res.json({ samples });
}));

// POST /api/ml/train - Start a new training run
mlRouter.post('/train', asyncHandler(async (_req, res) => {
    // Get count of human-labeled images (used for training)
    const imageCount = await getHumanLabeledImageCount();

    if (imageCount === 0) {
        throw new BadRequestError('No human-labeled images available for training');
    }

    // Create training run record
    const run = await createTrainingRun(imageCount);

    // Update to running status
    const runningRun = await updateTrainingRun(run.id, {
        status: 'running',
        startedAt: new Date(),
    });

    // Emit start event
    emitEvent('training-run-started', { run: runningRun! });

    // Return immediately - training runs in background
    res.json(runningRun);

    // Run training in background (don't await)
    const startTime = Date.now();
    (async () => {
        try {
            interface TrainResult {
                accuracy?: number;
                loss?: number;
                val_accuracy?: number;
                val_loss?: number;
            }
            const result = await mlHttp.post<TrainResult>('/train/', { run_id: run.id });

            const durationMs = Date.now() - startTime;
            const completedRun = await updateTrainingRun(run.id, {
                status: 'completed',
                completedAt: new Date(),
                durationMs,
                accuracy: result.accuracy,
                loss: result.loss,
                valAccuracy: result.val_accuracy,
                valLoss: result.val_loss,
            });

            // Emit completion event
            emitEvent('training-run-completed', { run: completedRun! });
            log.info(`Training run ${run.id} completed`, { durationMs, accuracy: result.accuracy });
        } catch (error) {
            const durationMs = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : 'Training failed';

            const failedRun = await updateTrainingRun(run.id, {
                status: 'failed',
                completedAt: new Date(),
                durationMs,
                errorMessage,
            });

            // Emit failure event
            emitEvent('training-run-failed', { run: failedRun! });
            log.error(`Training run ${run.id} failed`, { error: errorMessage, durationMs });
        }
    })();
}));

// GET /api/ml/training-runs - List all training runs
mlRouter.get('/training-runs', asyncHandler(async (_req, res) => {
    const runs = await listTrainingRuns();
    res.json({ runs });
}));

// GET /api/ml/training-runs/:id - Get single training run
mlRouter.get('/training-runs/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getTrainingRun(id);
    if (!run) {
        throw new NotFoundError('Training run not found');
    }

    res.json(run);
}));

// POST /api/ml/training-runs/:id/progress - Receive progress update from ML service
mlRouter.post('/training-runs/:id/progress', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { epoch, totalEpochs, loss, accuracy } = req.body;

    // Emit progress event via WebSocket
    emitEvent('training-run-progress', {
        runId: id,
        epoch,
        totalEpochs,
        loss,
        accuracy,
    });

    res.json({ success: true });
}));
