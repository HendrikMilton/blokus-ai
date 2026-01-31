import { Router } from 'express';
import { randomUUID } from 'crypto';
import { asyncHandler } from '../../lib/async-handler';
import { mlHttp } from '../../lib/http';
import { BadRequestError, NotFoundError } from '../../error';
import type { BoardCorners, SampleStatusFlags } from '@blokus-ai/shared';
import { listSamples, getSample, getUploadPresignedUrl, getCloudFrontUrl, getThumbnailUrl, generateThumbnail, deleteSamples } from './s3';
import { getTagsForSamples } from '../tags/database';
import { getSampleStatusFlagsById, getSampleDetectionInfoMap, deleteBoardDetections, saveManualLabel, ManualLabelData, saveCorners, getCorners, clearCorners } from '../ml/database';
import { getPipeline } from '../pipelines/database';
import { log } from '../../lib/log';

export const samplesRouter = Router();

const defaultStatusFlags: SampleStatusFlags = {
    aiLabeled: false,
    humanLabeled: false,
    verified: false,
};

// POST /api/samples/upload/presign - Get presigned URL for direct S3 upload
samplesRouter.post('/upload/presign', asyncHandler(async (_req, res) => {
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

// POST /api/samples/upload/complete - Called after S3 upload to generate thumbnail
samplesRouter.post('/upload/complete', asyncHandler(async (req, res) => {
    const { id } = req.body;

    if (!id) {
        throw new BadRequestError('id is required');
    }

    await generateThumbnail(id);
    res.json({ success: true, thumbnailUrl: getThumbnailUrl(id) });
}));

// POST /api/samples/thumbnails/regenerate - Regenerate all thumbnails
samplesRouter.post('/thumbnails/regenerate', asyncHandler(async (_req, res) => {
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

// POST /api/samples/upload - Upload board image (legacy, through API)
samplesRouter.post('/upload', asyncHandler(async (req, res) => {
    const { image } = req.body;

    if (!image) {
        throw new BadRequestError('Image is required');
    }

    const data = await mlHttp.post('/upload/', { image });
    res.json(data);
}));

// GET /api/samples - List all samples from S3
// Optional query params:
//   ?modelId= to filter AI detections by model (can be pipeline:uuid for pipelines)
//   ?filter= to apply preset filter (unlabeled, false-positives, false-negatives, low-iou, disagreements)
//   ?tagId= to filter by tag
samplesRouter.get('/', asyncHandler(async (req, res) => {
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

// GET /api/samples/:id - Get single sample
samplesRouter.get('/:id', asyncHandler(async (req, res) => {
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

// GET /api/samples/:id/image - Redirect to CloudFront
samplesRouter.get('/:id/image', asyncHandler(async (req, res) => {
    const { id } = req.params;
    res.redirect(301, getCloudFrontUrl(id));
}));

// DELETE /api/samples - Delete multiple samples (images + detections)
samplesRouter.delete('/', asyncHandler(async (req, res) => {
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

// POST /api/samples/:id/label - Save manual label (for data labeling without ML detection)
samplesRouter.post('/:id/label', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const label = req.body as ManualLabelData;

    await saveManualLabel(id, label);
    res.json({ success: true });
}));

// GET /api/samples/:id/corners - Get corners for a sample
samplesRouter.get('/:id/corners', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const corners = await getCorners(id);
    res.json({ corners });
}));

// POST /api/samples/:id/corners - Save corners for a sample
samplesRouter.post('/:id/corners', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const corners = req.body as BoardCorners;
    await saveCorners(id, corners);
    res.json({ success: true });
}));

// DELETE /api/samples/:id/corners - Clear corners for a sample
samplesRouter.delete('/:id/corners', asyncHandler(async (req, res) => {
    const { id } = req.params;
    await clearCorners(id);
    res.json({ success: true });
}));

// ============================================================================
// Grid Labels (per sample)
// ============================================================================

import { getGridLabel, saveGridLabel, deleteGridLabel, getCornerDetectionsForSamples } from '../ml/database';
import { mlHttp } from '../../lib/http';

// GET /api/samples/:id/grid - Get grid label for a sample
samplesRouter.get('/:id/grid', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const label = await getGridLabel(id);
    res.json({ grid: label?.grid ?? null });
}));

// POST /api/samples/:id/grid - Save grid label for a sample
samplesRouter.post('/:id/grid', asyncHandler(async (req, res) => {
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

// DELETE /api/samples/:id/grid - Delete grid label for a sample
samplesRouter.delete('/:id/grid', asyncHandler(async (req, res) => {
    const { id } = req.params;
    await deleteGridLabel(id);
    res.json({ success: true });
}));

// POST /api/samples/:id/grid/predict - Predict grid colors using AI
samplesRouter.post('/:id/grid/predict', asyncHandler(async (req, res) => {
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

// Re-export S3 utilities for use by other features
export { listSamples, getSample, getCloudFrontUrl, getThumbnailUrl } from './s3';
