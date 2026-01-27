import { Router } from 'express';
import { randomUUID } from 'crypto';
import { asyncHandler } from '../../lib/async-handler';
import { mlHttp } from '../../lib/http';
import type { DetectResponse } from '@blokus-ai/shared';
import { BadRequestError, NotFoundError } from '../../error';
import { getBoardDetectionWithCorrectionBySampleId, upsertBoardDetection, deleteBoardDetection, deleteBoardDetections, saveBoardDetectionCorrection, CorrectionData, setVerificationStatus, saveManualLabel, ManualLabelData, getSampleStatusFlags, getSampleStatusFlagsById, getSampleDetectionInfoMap, createTrainingRun, updateTrainingRun, listTrainingRuns, getTrainingRun, deleteTrainingRun, getHumanLabeledImageCount, getTrainingData, getClassifierTrainingData, createDetectionRun, updateDetectionRun, listDetectionRuns, getDetectionRun, deleteDetectionRun, clearAllAiLabels, getSampleIdsForBulkDetection, calculateAverageIoU } from './database';
import type { SampleStatusFlags } from '@blokus-ai/shared';
import { listSamples, getSample, getUploadPresignedUrl, getCloudFrontUrl, getThumbnailUrl, generateThumbnail, deleteSamples } from './s3';
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

// POST /api/ml/thumbnails/regenerate - Regenerate all thumbnails
mlRouter.post('/thumbnails/regenerate', asyncHandler(async (_req, res) => {
    const samples = await listSamples();
    log.info(`Regenerating ${samples.length} thumbnails...`);

    let success = 0;
    let failed = 0;

    for (const sample of samples) {
        try {
            await generateThumbnail(sample.id);
            success++;
            log.info(`Regenerated thumbnail ${success}/${samples.length}: ${sample.id}`);
        } catch (err) {
            failed++;
            log.error(`Failed to regenerate thumbnail for ${sample.id}`, { error: err });
        }
    }

    log.info(`Thumbnail regeneration complete: ${success} success, ${failed} failed`);
    res.json({ success, failed, total: samples.length });
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
    const detectionInfoMap = await getSampleDetectionInfoMap();

    const samples = s3Samples.map(s => {
        const info = detectionInfoMap.get(s.id);

        return {
            id: s.id,
            filename: `${s.id}.png`,
            status: info?.status || defaultStatusFlags,
            ai: info?.ai || null,
            human: info?.human || null,
            iou: info?.iou ?? null,
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

// DELETE /api/ml/samples - Delete multiple samples (images + detections)
mlRouter.delete('/samples', asyncHandler(async (req, res) => {
    const { ids } = req.body;

    if (!ids || !Array.isArray(ids) || ids.length === 0) {
        throw new BadRequestError('ids array is required');
    }

    // Delete detections from database first
    const detectionsDeleted = await deleteBoardDetections(ids);

    // Delete images from S3
    const samplesDeleted = await deleteSamples(ids);

    log.info(`Deleted ${samplesDeleted} samples and ${detectionsDeleted} detections`);

    res.json({
        deleted: samplesDeleted,
        detectionsDeleted,
    });
}));

// POST /api/ml/detect/test - Test detection with base64 image (proxy to ML service)
// MUST be defined before /detect/:id to avoid route conflicts
mlRouter.post('/detect/test', asyncHandler(async (req, res) => {
    const { image, modelId } = req.body as { image: string; modelId?: string };

    if (!image) {
        throw new BadRequestError('Image is required');
    }

    const response = await mlHttp.post('/detect/test', {
        image,
        model_id: modelId,
    });

    res.json(response);
}));

// POST /api/ml/detect/:id - Run board detection and store result
mlRouter.post('/detect/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { modelId } = req.body;

    // Build query params - if modelId is provided, use it for trained model detection
    const queryParams = modelId ? `?model_id=${modelId}` : '';
    const data = await mlHttp.post<DetectResponse>(`/detect/${id}${queryParams}`, {});
    await upsertBoardDetection(data);
    res.json(data);
}));

// GET /api/ml/detect/:id/preview - Run detection without saving (for testing models)
mlRouter.get('/detect/:id/preview', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { modelId } = req.query;

    const queryParams = modelId ? `?model_id=${modelId}` : '';
    const data = await mlHttp.post<DetectResponse>(`/detect/${id}${queryParams}`, {});
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

// GET /api/ml/classifier-training-data - Get human-labeled samples for classifier training
mlRouter.get('/classifier-training-data', asyncHandler(async (_req, res) => {
    const samples = await getClassifierTrainingData();
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
                classifier_accuracy?: number;
                classifier_loss?: number;
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
                classifierAccuracy: result.classifier_accuracy,
                classifierLoss: result.classifier_loss,
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
    const { phase, epoch, totalEpochs, loss, accuracy } = req.body;

    log.info(`[DEBUG] Received progress for run ${id}: phase=${phase}, epoch=${epoch}/${totalEpochs}, loss=${loss}, accuracy=${accuracy}`);

    // Emit progress event via WebSocket
    const payload = {
        runId: id,
        phase: phase || 'detection',
        epoch,
        totalEpochs,
        loss,
        accuracy,
    };
    log.info(`[DEBUG] Emitting training-run-progress event: ${JSON.stringify(payload)}`);
    emitEvent('training-run-progress', payload);

    res.json({ success: true });
}));

// DELETE /api/ml/training-runs/:id - Delete a training run and its model files
mlRouter.delete('/training-runs/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getTrainingRun(id);
    if (!run) {
        throw new NotFoundError('Training run not found');
    }

    // Delete model files via ML service
    try {
        await mlHttp.delete(`/models/${id}`);
    } catch (err) {
        // Log but don't fail if model deletion fails (might not exist)
        log.warn(`Failed to delete model files for run ${id}: ${err}`);
    }

    // Clear all AI labels (detection runs will be cascade-deleted, but we need to clear labels too)
    const clearedCount = await clearAllAiLabels();

    // Delete from database (cascade deletes detection_runs)
    await deleteTrainingRun(id);

    // Notify all clients
    emitEvent('training-run-deleted', { runId: id });

    log.info(`Deleted training run ${id}, cleared ${clearedCount} AI labels`);
    res.json({ success: true, clearedAiLabels: clearedCount });
}));

// ============================================================================
// Detection Run Routes
// ============================================================================

// POST /api/ml/detection-runs - Start a new bulk detection run
mlRouter.post('/detection-runs', asyncHandler(async (req, res) => {
    const { modelId, includeExistingAiLabeled = false } = req.body;

    log.info(`[DEBUG] Starting detection run with modelId: ${modelId}`);

    if (!modelId) {
        throw new BadRequestError('modelId is required');
    }

    // Validate that the model exists and is completed
    const trainingRun = await getTrainingRun(modelId);
    log.info(`[DEBUG] Found training run: ${JSON.stringify(trainingRun)}`);
    if (!trainingRun) {
        throw new NotFoundError('Training run not found');
    }
    if (trainingRun.status !== 'completed') {
        throw new BadRequestError('Training run must be completed to use for detection');
    }

    // Get all sample IDs from S3
    const s3Samples = await listSamples();
    const allSampleIds = s3Samples.map(s => s.id);
    log.info(`[DEBUG] Found ${allSampleIds.length} samples in S3, includeExistingAiLabeled=${includeExistingAiLabeled}`);

    // Get eligible sample IDs (optionally excluding samples with existing AI detections)
    const { sampleIds, existingDetectionCount } = await getSampleIdsForBulkDetection(allSampleIds, includeExistingAiLabeled);
    log.info(`[DEBUG] Found ${sampleIds.length} eligible samples (${existingDetectionCount} already have AI detections, ${includeExistingAiLabeled ? 'included' : 'excluded'})`);

    if (sampleIds.length === 0) {
        if (!includeExistingAiLabeled && existingDetectionCount > 0) {
            throw new BadRequestError(`All ${existingDetectionCount} samples already have AI detections. Check "Include samples with existing AI detections" to re-process them.`);
        } else {
            throw new BadRequestError('No samples found in S3.');
        }
    }

    // Create detection run record
    const run = await createDetectionRun(modelId, sampleIds.length);

    // Update to running status
    const runningRun = await updateDetectionRun(run.id, {
        status: 'running',
        startedAt: new Date(),
    });

    // Emit start event
    emitEvent('detection-run-started', { run: runningRun! });

    // Return immediately - detection runs in background
    res.json(runningRun);

    // Run detection in background (don't await)
    const startTime = Date.now();
    (async () => {
        try {
            await mlHttp.post('/detect/bulk', {
                run_id: run.id,
                model_id: modelId,
                sample_ids: sampleIds,
            });
            // ML service will call back to /detection-runs/:id/complete when done
        } catch (error) {
            const durationMs = Date.now() - startTime;
            const errorMessage = error instanceof Error ? error.message : 'Detection failed';

            const failedRun = await updateDetectionRun(run.id, {
                status: 'failed',
                completedAt: new Date(),
                durationMs,
                errorMessage,
            });

            // Emit failure event
            emitEvent('detection-run-failed', { run: failedRun! });
            log.error(`Detection run ${run.id} failed`, { error: errorMessage, durationMs });
        }
    })();
}));

// GET /api/ml/detection-runs - List all detection runs
mlRouter.get('/detection-runs', asyncHandler(async (_req, res) => {
    const runs = await listDetectionRuns();
    res.json({ runs });
}));

// GET /api/ml/detection-runs/:id - Get single detection run
mlRouter.get('/detection-runs/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getDetectionRun(id);
    if (!run) {
        throw new NotFoundError('Detection run not found');
    }

    res.json(run);
}));

// POST /api/ml/detection-runs/:id/progress - Receive progress update from ML service
mlRouter.post('/detection-runs/:id/progress', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { processedCount, successCount, failedCount, totalCount } = req.body;

    // Update the detection run record
    await updateDetectionRun(id, {
        processedCount,
        successCount,
        failedCount,
    });

    // Emit progress event via WebSocket
    emitEvent('detection-run-progress', {
        runId: id,
        processedCount,
        totalCount,
        successCount,
        failedCount,
    });

    res.json({ success: true });
}));

// POST /api/ml/detection-runs/:id/complete - ML service calls when detection is complete
mlRouter.post('/detection-runs/:id/complete', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { successCount, failedCount, durationMs } = req.body;

    // Calculate average IoU after detection completes
    const avgIou = await calculateAverageIoU();

    const completedRun = await updateDetectionRun(id, {
        status: 'completed',
        completedAt: new Date(),
        successCount,
        failedCount,
        durationMs,
        avgIou: avgIou ?? undefined,
    });

    if (!completedRun) {
        throw new NotFoundError('Detection run not found');
    }

    // Emit completion event
    emitEvent('detection-run-completed', { run: completedRun });
    log.info(`Detection run ${id} completed`, { successCount, failedCount, durationMs, avgIou });

    res.json({ success: true });
}));

// DELETE /api/ml/detection-runs/:id - Delete a detection run and clear associated AI labels
mlRouter.delete('/detection-runs/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getDetectionRun(id);
    if (!run) {
        throw new NotFoundError('Detection run not found');
    }

    // Clear all AI labels from board_detections (resets to placeholder values)
    const clearedCount = await clearAllAiLabels();

    // Delete the detection run record
    await deleteDetectionRun(id);

    // Notify all clients
    emitEvent('detection-run-deleted', { runId: id });

    log.info(`Deleted detection run ${id}, cleared ${clearedCount} AI labels`);
    res.json({ success: true, clearedAiLabels: clearedCount });
}));

// POST /api/ml/detect/:id/result - Store single detection result (internal, used by ML service)
mlRouter.post('/detect/:id/result', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const detection = req.body as DetectResponse;

    // Ensure the sample_id matches
    if (detection.sample_id !== id) {
        detection.sample_id = id;
    }

    await upsertBoardDetection(detection);
    res.json({ success: true });
}));
