import { Router } from 'express';
import { randomUUID } from 'crypto';
import { asyncHandler } from '../../lib/async-handler';
import { mlHttp } from '../../lib/http';
import type { DetectResponse, BoardCorners, PerspectiveFromCornersRequest } from '@blokus-ai/shared';
import { BadRequestError, NotFoundError } from '../../error';
import { getBoardDetectionWithCorrectionBySampleId, upsertModelDetection, deleteBoardDetection, deleteBoardDetections, saveBoardDetectionCorrection, CorrectionData, setVerificationStatus, saveManualLabel, ManualLabelData, getSampleStatusFlagsById, getSampleDetectionInfoMap, createTrainingRun, updateTrainingRun, listTrainingRuns, getTrainingRun, deleteTrainingRun, getHumanLabeledImageCount, getTrainingData, getClassifierTrainingData, getPoseTrainingData, getPoseTrainingImageCount, createDetectionRun, updateDetectionRun, listDetectionRuns, getDetectionRun, deleteDetectionRun, getSampleIdsForBulkDetection, calculateAverageIoU, createPipeline, listPipelines, getPipeline, getPipelineWithComponents, updatePipeline, deletePipeline, saveCorners, getCorners, clearCorners, getTagsForSamples, createCornerDetectionRun, updateCornerDetectionRun, listCornerDetectionRuns, getCornerDetectionRun, deleteCornerDetectionRun, upsertCornerDetection, getCornerDetectionsByRunId, getCornerDetectionsForSamples, getGridLabel, saveGridLabel, deleteGridLabel, getGridLabelsForSamples, getGridPredictionsForSamples, saveGridPrediction } from './database';
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
// Optional query params:
//   ?modelId= to filter AI detections by model (can be pipeline:uuid for pipelines)
//   ?filter= to apply preset filter (unlabeled, false-positives, false-negatives, low-iou, disagreements)
//   ?tagId= to filter by tag
mlRouter.get('/samples', asyncHandler(async (req, res) => {
    const modelIdParam = req.query.modelId as string | undefined;
    const filter = req.query.filter as string | undefined;
    const tagIdParam = req.query.tagId as string | undefined;

    // Handle pipeline: prefix - expand to component model IDs
    let modelIds: string[] | null = null;
    if (modelIdParam) {
        if (modelIdParam.startsWith('pipeline:')) {
            const pipelineId = modelIdParam.replace('pipeline:', '');
            const pipeline = await getPipeline(pipelineId);
            if (pipeline) {
                modelIds = pipeline.componentIds;
            }
        } else {
            modelIds = [modelIdParam];
        }
    }

    const s3Samples = await listSamples();
    // Pass first model ID for now - multi-model support would need refactoring
    const detectionInfoMap = await getSampleDetectionInfoMap(modelIds?.[0] || null);

    // Get tags for all samples
    const sampleIds = s3Samples.map(s => s.id);
    const tagsMap = await getTagsForSamples(sampleIds);

    let samples = s3Samples.map(s => {
        const info = detectionInfoMap.get(s.id);
        const sampleTags = tagsMap.get(s.id) || [];

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
            tags: sampleTags,
        };
    });

    // Apply tag filter if specified
    if (tagIdParam) {
        samples = samples.filter(s => s.tags.some(t => t.id === tagIdParam));
    }

    // Apply filter if specified
    if (filter) {
        samples = samples.filter(s => {
            switch (filter) {
                case 'unlabeled':
                    return !s.status.humanLabeled;
                case 'false-positives':
                    return s.ai?.hasBoardDetected === true && s.human?.hasBoardDetected === false;
                case 'false-negatives':
                    return s.ai?.hasBoardDetected === false && s.human?.hasBoardDetected === true;
                case 'low-iou':
                    return s.iou !== null && s.iou < 0.8;
                case 'disagreements':
                    return s.ai !== null && s.human !== null &&
                    s.ai.hasBoardDetected !== s.human.hasBoardDetected;
                default:
                    return true;
            }
        });
    }

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

    // Check if this is a pipeline request
    if (modelId && modelId.startsWith('pipeline:')) {
        const pipelineId = modelId.replace('pipeline:', '');
        const pipeline = await getPipeline(pipelineId);

        if (!pipeline) {
            throw new NotFoundError('Pipeline not found');
        }

        // Try each model in the pipeline until one detects a board
        interface TestDetectResponse {
            detection: {
                hasBoardDetected: boolean;
                isEntireBoardVisible: boolean;
                confidence: number;
                classifierConfidence?: number | null;
                boundingBox: { x: number; y: number; width: number; height: number } | null;
            };
            processing_time_ms: number;
        }

        let lastResult: TestDetectResponse | null = null;
        let totalProcessingTime = 0;

        for (const componentModelId of pipeline.componentIds) {
            const result = await mlHttp.post<TestDetectResponse>('/detect/test', {
                image,
                model_id: componentModelId,
            });

            totalProcessingTime += result.processing_time_ms;
            lastResult = result;

            // If board detected, stop and return this result
            if (result.detection.hasBoardDetected) {
                res.json({
                    ...result,
                    processing_time_ms: totalProcessingTime,
                });
                return;
            }
        }

        // No model detected a board, return the last result
        if (lastResult) {
            res.json({
                ...lastResult,
                processing_time_ms: totalProcessingTime,
            });
        } else {
            throw new BadRequestError('Pipeline has no components');
        }
        return;
    }

    // Regular model detection
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

    if (!modelId) {
        throw new BadRequestError('modelId is required');
    }

    // Run detection with the specified model
    const queryParams = `?model_id=${modelId}`;
    const data = await mlHttp.post<DetectResponse>(`/detect/${id}${queryParams}`, {});

    // Store in model_detections table
    await upsertModelDetection(data, modelId);
    res.json(data);
}));

// GET /api/ml/detect/:id/preview - Run detection without saving (for testing models)
mlRouter.get('/detect/:id/preview', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { modelId } = req.query;

    // Check if this is a pipeline request
    if (modelId && typeof modelId === 'string' && modelId.startsWith('pipeline:')) {
        const pipelineId = modelId.replace('pipeline:', '');
        const pipeline = await getPipeline(pipelineId);

        if (!pipeline) {
            throw new NotFoundError('Pipeline not found');
        }

        // Try each model in the pipeline until one detects a board
        let lastResult: DetectResponse | null = null;

        for (const componentModelId of pipeline.componentIds) {
            const result = await mlHttp.post<DetectResponse>(`/detect/${id}?model_id=${componentModelId}`, {});
            lastResult = result;

            // If board detected, stop and return this result
            if (result.detection.hasBoardDetected) {
                res.json(result);
                return;
            }
        }

        // No model detected a board, return the last result
        if (lastResult) {
            res.json(lastResult);
        } else {
            throw new BadRequestError('Pipeline has no components');
        }
        return;
    }

    // Regular model detection
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

// GET /api/ml/samples/:id/corners - Get corners for a sample
mlRouter.get('/samples/:id/corners', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const corners = await getCorners(id);
    res.json({ corners });
}));

// POST /api/ml/samples/:id/corners - Save corners for a sample
mlRouter.post('/samples/:id/corners', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const corners = req.body as BoardCorners;
    await saveCorners(id, corners);
    res.json({ success: true });
}));

// DELETE /api/ml/samples/:id/corners - Clear corners for a sample
mlRouter.delete('/samples/:id/corners', asyncHandler(async (req, res) => {
    const { id } = req.params;
    await clearCorners(id);
    res.json({ success: true });
}));

// ============================================================================
// Training Routes
// ============================================================================

// GET /api/ml/training-data - Get human-labeled samples for training
// Optional query param: ?sampleIds=id1,id2,id3 to filter by specific samples
mlRouter.get('/training-data', asyncHandler(async (req, res) => {
    const sampleIdsParam = req.query.sampleIds as string | undefined;
    const sampleIds = sampleIdsParam ? sampleIdsParam.split(',') : undefined;
    const samples = await getTrainingData(sampleIds);
    res.json({ samples });
}));

// GET /api/ml/classifier-training-data - Get human-labeled samples for classifier training
mlRouter.get('/classifier-training-data', asyncHandler(async (_req, res) => {
    const samples = await getClassifierTrainingData();
    res.json({ samples });
}));

// GET /api/ml/pose-training-data - Get human-labeled samples with corners for pose training
mlRouter.get('/pose-training-data', asyncHandler(async (_req, res) => {
    const samples = await getPoseTrainingData();
    res.json({ samples, count: samples.length });
}));

// GET /api/ml/pose-training-count - Get count of samples with corner labels
mlRouter.get('/pose-training-count', asyncHandler(async (_req, res) => {
    const count = await getPoseTrainingImageCount();
    res.json({ count });
}));

// POST /api/ml/train - Start a new training run
mlRouter.post('/train', asyncHandler(async (req, res) => {
    const { sampleIds } = req.body as { sampleIds?: string[] };

    // Get count of human-labeled images (used for training)
    // If sampleIds provided, only count those that are human-labeled
    const imageCount = await getHumanLabeledImageCount(sampleIds);

    if (imageCount === 0) {
        if (sampleIds && sampleIds.length > 0) {
            throw new BadRequestError('None of the selected images have human labels');
        }
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

    // Fire-and-forget: ML service will call back on completion/failure
    mlHttp.post('/train/', { run_id: run.id, sample_ids: sampleIds }).catch(err => {
        // Log but don't fail - ML service might still succeed and call back
        log.warn(`Initial training request failed (ML service will callback on completion): ${err}`);
    });
}));

// POST /api/ml/training-runs/:id/complete - ML service callback on training completion
mlRouter.post('/training-runs/:id/complete', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const {
        accuracy,
        loss,
        val_accuracy,
        val_loss,
        classifier_accuracy,
        classifier_loss,
        pose_accuracy,
        pose_loss,
        pose_error,
        pose_image_count,
        duration_ms,
    } = req.body;

    log.info(`[train] Training run ${id} completed callback received`);

    const run = await getTrainingRun(id);
    if (!run) {
        throw new NotFoundError('Training run not found');
    }

    const completedRun = await updateTrainingRun(id, {
        status: 'completed',
        completedAt: new Date(),
        durationMs: duration_ms,
        accuracy,
        loss,
        valAccuracy: val_accuracy,
        valLoss: val_loss,
        classifierAccuracy: classifier_accuracy,
        classifierLoss: classifier_loss,
        poseAccuracy: pose_accuracy,
        poseLoss: pose_loss,
        poseError: pose_error,
        poseImageCount: pose_image_count,
    });

    emitEvent('training-run-completed', { run: completedRun! });
    log.info(`Training run ${id} completed`, { durationMs: duration_ms, accuracy });

    res.json({ success: true });
}));

// POST /api/ml/training-runs/:id/failed - ML service callback on training failure
mlRouter.post('/training-runs/:id/failed', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { error_message, duration_ms } = req.body;

    log.info(`[train] Training run ${id} failed callback received: ${error_message}`);

    const run = await getTrainingRun(id);
    if (!run) {
        throw new NotFoundError('Training run not found');
    }

    const failedRun = await updateTrainingRun(id, {
        status: 'failed',
        completedAt: new Date(),
        durationMs: duration_ms,
        errorMessage: error_message,
    });

    emitEvent('training-run-failed', { run: failedRun! });
    log.error(`Training run ${id} failed`, { error: error_message, durationMs: duration_ms });

    res.json({ success: true });
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

    // model_detections are cascade-deleted when training_run is deleted
    // detection_runs are also cascade-deleted

    // Delete from database (cascade deletes model_detections and detection_runs)
    await deleteTrainingRun(id);

    // Notify all clients
    emitEvent('training-run-deleted', { runId: id });

    log.info(`Deleted training run ${id}`);
    res.json({ success: true });
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

    // Check if this is a pipeline request
    let pipelineComponentIds: string[] | null = null;
    let effectiveModelId = modelId;

    if (modelId.startsWith('pipeline:')) {
        const pipelineId = modelId.replace('pipeline:', '');
        const pipeline = await getPipeline(pipelineId);

        if (!pipeline) {
            throw new NotFoundError('Pipeline not found');
        }

        if (pipeline.componentIds.length === 0) {
            throw new BadRequestError('Pipeline has no components');
        }

        // Use the first model in the pipeline for detection run storage
        // The detection run will use this as the "primary" model
        effectiveModelId = pipeline.componentIds[0];
        pipelineComponentIds = pipeline.componentIds;
        log.info(`[DEBUG] Pipeline ${pipelineId} has ${pipeline.componentIds.length} components`);
    }

    // Validate that the model exists and is completed
    const trainingRun = await getTrainingRun(effectiveModelId);
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

    // Get eligible sample IDs (optionally excluding samples with existing AI detections for this model)
    const { sampleIds, existingDetectionCount } = await getSampleIdsForBulkDetection(allSampleIds, effectiveModelId, includeExistingAiLabeled);
    log.info(`[DEBUG] Found ${sampleIds.length} eligible samples (${existingDetectionCount} already have AI detections for this model, ${includeExistingAiLabeled ? 'included' : 'excluded'})`);

    if (sampleIds.length === 0) {
        if (!includeExistingAiLabeled && existingDetectionCount > 0) {
            throw new BadRequestError(`All ${existingDetectionCount} samples already have AI detections. Check "Include samples with existing AI detections" to re-process them.`);
        } else {
            throw new BadRequestError('No samples found in S3.');
        }
    }

    // Create detection run record
    // For pipelines, store the pipeline ID separately and use effectiveModelId for model reference
    const pipelineId = pipelineComponentIds ? modelId.replace('pipeline:', '') : null;
    const run = await createDetectionRun(effectiveModelId, sampleIds.length, pipelineId);

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
                model_id: effectiveModelId,
                sample_ids: sampleIds,
                // Include pipeline components if this is a pipeline detection
                pipeline_components: pipelineComponentIds,
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

    // Get the detection run to find the model ID
    const detectionRun = await getDetectionRun(id);
    if (!detectionRun) {
        throw new NotFoundError('Detection run not found');
    }

    // Calculate average IoU after detection completes (for this model)
    const avgIou = await calculateAverageIoU(detectionRun.modelId);

    const completedRun = await updateDetectionRun(id, {
        status: 'completed',
        completedAt: new Date(),
        successCount,
        failedCount,
        durationMs,
        avgIou: avgIou ?? undefined,
    });

    // Emit completion event
    emitEvent('detection-run-completed', { run: completedRun! });
    log.info(`Detection run ${id} completed`, { successCount, failedCount, durationMs, avgIou });

    res.json({ success: true });
}));

// DELETE /api/ml/detection-runs/:id - Delete a detection run
// Note: Model detections remain in model_detections table (associated with the training_run/model)
mlRouter.delete('/detection-runs/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getDetectionRun(id);
    if (!run) {
        throw new NotFoundError('Detection run not found');
    }

    // Delete the detection run record
    // Note: This doesn't delete model_detections - they're associated with the model, not the run
    await deleteDetectionRun(id);

    // Notify all clients
    emitEvent('detection-run-deleted', { runId: id });

    log.info(`Deleted detection run ${id}`);
    res.json({ success: true });
}));

// POST /api/ml/detect/:id/perspective - Run perspective correction on a sample (old method)
mlRouter.post('/detect/:id/perspective', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { bbox } = req.body as { bbox: { x: number; y: number; width: number; height: number } };

    if (!bbox) {
        throw new BadRequestError('bbox is required');
    }

    const response = await mlHttp.post(`/detect/${id}/perspective`, {
        bbox_x: bbox.x,
        bbox_y: bbox.y,
        bbox_width: bbox.width,
        bbox_height: bbox.height,
    });

    res.json(response);
}));

// POST /api/ml/detect/:id/perspective-from-corners - Generate perspective correction using saved corners
mlRouter.post('/detect/:id/perspective-from-corners', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { best, hough, edge }: PerspectiveFromCornersRequest = req.body;

    const response = await mlHttp.post(`/detect/${id}/perspective-from-corners`, {
        best,
        hough,
        edge,
    });

    res.json(response);
}));

// POST /api/ml/detect/:id/corners - Run corner detection on a sample
mlRouter.post('/detect/:id/corners', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { bbox } = req.body as { bbox: { x: number; y: number; width: number; height: number } };

    if (!bbox) {
        throw new BadRequestError('bbox is required');
    }

    const response = await mlHttp.post(`/detect/${id}/corners`, {
        bbox_x: bbox.x,
        bbox_y: bbox.y,
        bbox_width: bbox.width,
        bbox_height: bbox.height,
    });

    res.json(response);
}));

// POST /api/ml/detect/:id/result - Store single detection result (internal, used by ML service)
mlRouter.post('/detect/:id/result', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { modelId } = req.query;
    const detection = req.body as DetectResponse;

    if (!modelId || typeof modelId !== 'string') {
        throw new BadRequestError('modelId query param is required');
    }

    // Ensure the sample_id matches
    if (detection.sample_id !== id) {
        detection.sample_id = id;
    }

    await upsertModelDetection(detection, modelId);
    res.json({ success: true });
}));

// ============================================================================
// Pipeline Routes
// ============================================================================

// POST /api/ml/pipelines - Create a new pipeline
mlRouter.post('/pipelines', asyncHandler(async (req, res) => {
    const { name, componentIds } = req.body as { name: string; componentIds: string[] };

    if (!name || typeof name !== 'string' || name.trim().length === 0) {
        throw new BadRequestError('name is required');
    }
    if (!componentIds || !Array.isArray(componentIds) || componentIds.length === 0) {
        throw new BadRequestError('componentIds must be a non-empty array');
    }

    // Validate that all component IDs are valid completed training runs
    for (const id of componentIds) {
        const run = await getTrainingRun(id);
        if (!run) {
            throw new BadRequestError(`Training run not found: ${id}`);
        }
        if (run.status !== 'completed') {
            throw new BadRequestError(`Training run is not completed: ${id}`);
        }
    }

    const pipeline = await createPipeline(name.trim(), componentIds);
    res.json(pipeline);
}));

// GET /api/ml/pipelines - List all pipelines
mlRouter.get('/pipelines', asyncHandler(async (_req, res) => {
    const pipelines = await listPipelines();
    res.json({ pipelines });
}));

// GET /api/ml/pipelines/:id - Get single pipeline with component details
mlRouter.get('/pipelines/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const pipeline = await getPipelineWithComponents(id);
    if (!pipeline) {
        throw new NotFoundError('Pipeline not found');
    }

    res.json(pipeline);
}));

// PUT /api/ml/pipelines/:id - Update a pipeline
mlRouter.put('/pipelines/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { name, componentIds } = req.body as { name?: string; componentIds?: string[] };

    const existing = await getPipeline(id);
    if (!existing) {
        throw new NotFoundError('Pipeline not found');
    }

    // Validate component IDs if provided
    if (componentIds) {
        if (!Array.isArray(componentIds) || componentIds.length === 0) {
            throw new BadRequestError('componentIds must be a non-empty array');
        }
        for (const compId of componentIds) {
            const run = await getTrainingRun(compId);
            if (!run) {
                throw new BadRequestError(`Training run not found: ${compId}`);
            }
            if (run.status !== 'completed') {
                throw new BadRequestError(`Training run is not completed: ${compId}`);
            }
        }
    }

    const updated = await updatePipeline(id, {
        name: name?.trim(),
        componentIds,
    });

    res.json(updated);
}));

// DELETE /api/ml/pipelines/:id - Delete a pipeline
mlRouter.delete('/pipelines/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const existing = await getPipeline(id);
    if (!existing) {
        throw new NotFoundError('Pipeline not found');
    }

    await deletePipeline(id);
    log.info(`Deleted pipeline ${id}`);
    res.json({ success: true });
}));

// ============================================================================
// Tag Routes
// ============================================================================

import { getAllTags, createTag, deleteTag, getTagsForSample, addTagToSample, removeTagFromSample } from './database';

// GET /api/ml/tags - List all tags
mlRouter.get('/tags', asyncHandler(async (_req, res) => {
    const tags = await getAllTags();
    res.json({ tags });
}));

// POST /api/ml/tags - Create a new tag
mlRouter.post('/tags', asyncHandler(async (req, res) => {
    const { name } = req.body as { name: string };

    if (!name || typeof name !== 'string' || name.trim().length === 0) {
        throw new BadRequestError('name is required');
    }

    const tag = await createTag(name.trim());
    res.json(tag);
}));

// DELETE /api/ml/tags/:id - Delete a tag
mlRouter.delete('/tags/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const deleted = await deleteTag(id);
    if (!deleted) {
        throw new NotFoundError('Tag not found');
    }

    res.json({ success: true });
}));

// GET /api/ml/samples/:id/tags - Get tags for a sample
mlRouter.get('/samples/:id/tags', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const tags = await getTagsForSample(id);
    res.json({ tags });
}));

// POST /api/ml/samples/:id/tags - Add tag to sample
mlRouter.post('/samples/:id/tags', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { tagId } = req.body as { tagId: string };

    if (!tagId) {
        throw new BadRequestError('tagId is required');
    }

    await addTagToSample(id, tagId);
    res.json({ success: true });
}));

// DELETE /api/ml/samples/:id/tags/:tagId - Remove tag from sample
mlRouter.delete('/samples/:id/tags/:tagId', asyncHandler(async (req, res) => {
    const { id, tagId } = req.params;

    const removed = await removeTagFromSample(id, tagId);
    if (!removed) {
        throw new NotFoundError('Tag assignment not found');
    }

    res.json({ success: true });
}));

// ============================================================================
// Corner Detection Run Routes
// ============================================================================

// POST /api/ml/corner-runs - Start bulk corner detection
mlRouter.post('/corner-runs', asyncHandler(async (req, res) => {
    const { sampleIds } = req.body as { sampleIds: string[] };

    if (!sampleIds || !Array.isArray(sampleIds) || sampleIds.length === 0) {
        throw new BadRequestError('sampleIds array is required');
    }

    // Create corner detection run record
    const run = await createCornerDetectionRun(sampleIds.length);

    // Update to running status
    const runningRun = await updateCornerDetectionRun(run.id, {
        status: 'running',
        startedAt: new Date(),
    });

    // Emit start event
    emitEvent('corner-run-started', { run: runningRun! });

    // Return immediately - corner detection runs in background
    res.json(runningRun);

    // Fire-and-forget: ML service will call back on progress/completion
    mlHttp.post('/detect/bulk-corners', {
        run_id: run.id,
        sample_ids: sampleIds,
    }).catch(async (err) => {
        log.warn(`Initial corner detection request failed: ${err}`);
        // Update run to failed status and emit event
        const failedRun = await updateCornerDetectionRun(run.id, {
            status: 'failed',
            completedAt: new Date(),
            errorMessage: `ML service request failed: ${err}`,
        });
        if (failedRun) {
            emitEvent('corner-run-failed', { run: failedRun });
        }
    });
}));

// GET /api/ml/corner-runs - List all corner detection runs
mlRouter.get('/corner-runs', asyncHandler(async (_req, res) => {
    const runs = await listCornerDetectionRuns();
    res.json({ runs });
}));

// GET /api/ml/corner-runs/:id - Get single corner detection run
mlRouter.get('/corner-runs/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getCornerDetectionRun(id);
    if (!run) {
        throw new NotFoundError('Corner detection run not found');
    }

    res.json(run);
}));

// POST /api/ml/corner-runs/:id/progress - ML service callback for progress
mlRouter.post('/corner-runs/:id/progress', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { processedCount, totalCount, successCount, failedCount } = req.body;

    // Update the corner detection run record
    await updateCornerDetectionRun(id, {
        processedCount,
        successCount,
        failedCount,
    });

    // Emit progress event via WebSocket
    emitEvent('corner-run-progress', {
        runId: id,
        processedCount,
        totalCount,
        successCount,
        failedCount,
    });

    res.json({ success: true });
}));

// POST /api/ml/corner-runs/:id/complete - ML service callback on completion
mlRouter.post('/corner-runs/:id/complete', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { successCount, failedCount, durationMs } = req.body;

    const run = await getCornerDetectionRun(id);
    if (!run) {
        throw new NotFoundError('Corner detection run not found');
    }

    const completedRun = await updateCornerDetectionRun(id, {
        status: 'completed',
        completedAt: new Date(),
        successCount,
        failedCount,
        durationMs,
    });

    // Emit completion event
    emitEvent('corner-run-completed', { run: completedRun! });
    log.info(`Corner detection run ${id} completed`, { successCount, failedCount, durationMs });

    res.json({ success: true });
}));

// POST /api/ml/corner-runs/:id/failed - ML service callback on failure
mlRouter.post('/corner-runs/:id/failed', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { errorMessage, durationMs } = req.body;

    const run = await getCornerDetectionRun(id);
    if (!run) {
        throw new NotFoundError('Corner detection run not found');
    }

    const failedRun = await updateCornerDetectionRun(id, {
        status: 'failed',
        completedAt: new Date(),
        durationMs,
        errorMessage,
    });

    emitEvent('corner-run-failed', { run: failedRun! });
    log.error(`Corner detection run ${id} failed`, { error: errorMessage, durationMs });

    res.json({ success: true });
}));

// DELETE /api/ml/corner-runs/:id - Delete a corner detection run
mlRouter.delete('/corner-runs/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const run = await getCornerDetectionRun(id);
    if (!run) {
        throw new NotFoundError('Corner detection run not found');
    }

    // Delete the corner detection run (cascade deletes corner_detections)
    await deleteCornerDetectionRun(id);

    log.info(`Deleted corner detection run ${id}`);
    res.json({ success: true });
}));

// POST /api/ml/corner-runs/:id/result - Store single corner detection result (internal, used by ML service)
mlRouter.post('/corner-runs/:id/result', asyncHandler(async (req, res) => {
    const { id: runId } = req.params;
    const result = req.body;

    await upsertCornerDetection({
        sampleId: result.sample_id,
        runId,
        best: result.best,
        hough: result.hough,
        edge: result.edge,
    });

    res.json({ success: true });
}));

// GET /api/ml/corner-detections - Get corner detections for a run or samples
mlRouter.get('/corner-detections', asyncHandler(async (req, res) => {
    const { runId, sampleIds } = req.query;

    if (runId && typeof runId === 'string') {
        const results = await getCornerDetectionsByRunId(runId);
        res.json({ results });
    } else if (sampleIds && typeof sampleIds === 'string') {
        const ids = sampleIds.split(',');
        const resultsMap = await getCornerDetectionsForSamples(ids);
        const results = Array.from(resultsMap.values());
        res.json({ results });
    } else {
        throw new BadRequestError('runId or sampleIds query param required');
    }
}));

// ============================================================================
// Grid Labels
// ============================================================================

// GET /api/ml/samples/:id/grid - Get grid label for a sample
mlRouter.get('/samples/:id/grid', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const label = await getGridLabel(id);
    res.json({ grid: label?.grid ?? null });
}));

// POST /api/ml/samples/:id/grid - Save grid label for a sample
mlRouter.post('/samples/:id/grid', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { grid } = req.body;

    if (!grid || !Array.isArray(grid)) {
        throw new BadRequestError('grid must be a 20x20 array');
    }

    // Validate grid dimensions
    if (grid.length !== 20) {
        throw new BadRequestError('grid must have 20 rows');
    }
    for (const row of grid) {
        if (!Array.isArray(row) || row.length !== 20) {
            throw new BadRequestError('each row must have 20 cells');
        }
    }

    const label = await saveGridLabel(id, grid);
    res.json(label);
}));

// DELETE /api/ml/samples/:id/grid - Delete grid label for a sample
mlRouter.delete('/samples/:id/grid', asyncHandler(async (req, res) => {
    const { id } = req.params;
    await deleteGridLabel(id);
    res.json({ success: true });
}));

// GET /api/ml/grid-labels - Get grid labels for multiple samples (batch)
mlRouter.get('/grid-labels', asyncHandler(async (req, res) => {
    const sampleIdsParam = req.query.sampleIds as string | undefined;

    if (!sampleIdsParam) {
        throw new BadRequestError('sampleIds query parameter is required');
    }

    const sampleIds = sampleIdsParam.split(',');
    const labels = await getGridLabelsForSamples(sampleIds);

    // Convert Map to array
    const labelsArray = Array.from(labels.values());
    res.json({ labels: labelsArray });
}));

// POST /api/ml/samples/:id/grid/predict - Predict grid colors using AI
mlRouter.post('/samples/:id/grid/predict', asyncHandler(async (req, res) => {
    const { id } = req.params;

    // Get corner detection result to find cropped image URL
    const cornerResults = await getCornerDetectionsForSamples([id]);
    const cornerResult = cornerResults.get(id);

    if (!cornerResult) {
        throw new NotFoundError('No corner detection found for this sample. Run corner detection first.');
    }

    // Find the best available cropped image URL
    const croppedUrl = cornerResult.best?.croppedUrl ||
                       cornerResult.hough?.croppedUrl ||
                       cornerResult.edge?.croppedUrl;

    if (!croppedUrl) {
        throw new NotFoundError('No cropped image available. Run corner detection first.');
    }

    // Call ML service to predict grid colors
    const prediction = await mlHttp.post<{
        grid: (string | null)[][];
        confidence: number;
        cellConfidences: number[][];
        processingTimeMs: number;
            }>('/detect/grid-predict', {
                image_url: croppedUrl,
            });

    res.json(prediction);
}));

// ============================================================================
// Grid Predictions (Bulk AI Prediction)
// ============================================================================

// POST /api/ml/grid-runs - Start bulk grid prediction
mlRouter.post('/grid-runs', asyncHandler(async (req, res) => {
    const { sampleIds } = req.body as { sampleIds?: string[] };

    // Get samples with corner detections (cropped images available)
    const s3Samples = await listSamples();
    const allSampleIds = s3Samples.map(s => s.id);

    // Get corner detections to find which samples have cropped images
    const cornerDetections = await getCornerDetectionsForSamples(allSampleIds);

    // Get detection info to check isEntireBoardVisible
    const detectionInfoMap = await getSampleDetectionInfoMap(null);

    // Filter to samples that have a cropped image URL AND entire board is visible
    let eligibleSampleIds = allSampleIds.filter(id => {
        const cd = cornerDetections.get(id);
        const hasCroppedImage = cd?.best?.croppedUrl || cd?.hough?.croppedUrl || cd?.edge?.croppedUrl;

        // Check if entire board is visible (prefer human label, fall back to AI)
        const info = detectionInfoMap.get(id);
        const isEntireBoardVisible = info?.human?.isEntireBoardVisible ?? info?.ai?.isEntireBoardVisible ?? false;

        return hasCroppedImage && isEntireBoardVisible;
    });

    // If specific sampleIds provided, filter to those
    if (sampleIds && sampleIds.length > 0) {
        const requestedSet = new Set(sampleIds);
        eligibleSampleIds = eligibleSampleIds.filter(id => requestedSet.has(id));
    }

    if (eligibleSampleIds.length === 0) {
        throw new BadRequestError('No eligible samples. Samples need cropped images and "Entire board visible" = Yes.');
    }

    // Build sample data for ML service
    const sampleData = eligibleSampleIds.map(id => {
        const cd = cornerDetections.get(id)!;
        const croppedUrl = cd.best?.croppedUrl || cd.hough?.croppedUrl || cd.edge?.croppedUrl;
        return { sample_id: id, cropped_url: croppedUrl };
    });

    const started = sampleData.length;

    // Emit start event
    emitEvent('grid-run-started', { totalCount: started });

    // Return immediately
    res.json({ started, skipped: allSampleIds.length - started });

    // Fire-and-forget: ML service will call back with results
    mlHttp.post('/detect/bulk-grid-predict', {
        samples: sampleData,
        api_base_url: process.env.API_URL || 'http://localhost:3009',
    }).catch(err => {
        log.warn(`Initial grid prediction request failed: ${err}`);
        emitEvent('grid-run-failed', { errorMessage: `ML service request failed: ${err}` });
    });
}));

// POST /api/ml/grid-runs/progress - ML service callback for progress
mlRouter.post('/grid-runs/progress', asyncHandler(async (req, res) => {
    const { processedCount, totalCount, successCount, failedCount } = req.body;

    // Emit progress event via WebSocket
    emitEvent('grid-run-progress', {
        processedCount,
        totalCount,
        successCount,
        failedCount,
    });

    res.json({ success: true });
}));

// POST /api/ml/grid-runs/complete - ML service callback on completion
mlRouter.post('/grid-runs/complete', asyncHandler(async (req, res) => {
    const { successCount, failedCount, durationMs } = req.body;

    // Emit completion event
    emitEvent('grid-run-completed', {
        successCount,
        failedCount,
        durationMs,
    });
    log.info('Grid prediction completed', { successCount, failedCount, durationMs });

    res.json({ success: true });
}));

// POST /api/ml/grid-runs/failed - ML service callback on failure
mlRouter.post('/grid-runs/failed', asyncHandler(async (req, res) => {
    const { errorMessage, durationMs } = req.body;

    // Emit failure event
    emitEvent('grid-run-failed', { errorMessage });
    log.error('Grid prediction failed', { error: errorMessage, durationMs });

    res.json({ success: true });
}));

// GET /api/ml/grid-predictions - Get AI predictions for samples (batch)
mlRouter.get('/grid-predictions', asyncHandler(async (req, res) => {
    const sampleIdsParam = req.query.sampleIds as string | undefined;

    if (!sampleIdsParam) {
        throw new BadRequestError('sampleIds query parameter is required');
    }

    const sampleIds = sampleIdsParam.split(',');
    const predictions = await getGridPredictionsForSamples(sampleIds);

    // Convert Map to array
    const predictionsArray = Array.from(predictions.values());
    res.json({ predictions: predictionsArray });
}));

// POST /api/ml/grid-predictions/:sampleId - Store grid prediction result (callback from ML service)
mlRouter.post('/grid-predictions/:sampleId', asyncHandler(async (req, res) => {
    const { sampleId } = req.params;
    const { grid, confidence, processingTimeMs } = req.body;

    if (!grid || typeof confidence !== 'number' || typeof processingTimeMs !== 'number') {
        throw new BadRequestError('grid, confidence, and processingTimeMs are required');
    }

    await saveGridPrediction(sampleId, grid, confidence, processingTimeMs);
    res.json({ success: true });
}));
