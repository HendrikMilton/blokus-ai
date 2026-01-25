import { Router } from 'express';
import { randomUUID } from 'crypto';
import { asyncHandler } from '../../lib/async-handler';
import { mlHttp } from '../../lib/http';
import type { DetectResponse } from '@blokus-ai/shared';
import { BadRequestError, NotFoundError } from '../../error';
import { getBoardDetectionBySampleId, upsertBoardDetection, getLabeledSampleIds, deleteBoardDetection, saveBoardDetectionCorrection, CorrectionData, getVerifiedSampleIds, setVerificationStatus, isDetectionVerified, saveManualLabel, ManualLabelData } from './database';
import { listSamples, getSample, getUploadPresignedUrl, getCloudFrontUrl, getThumbnailUrl, generateThumbnail } from './s3';

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

// GET /api/ml/samples - List all samples from S3
mlRouter.get('/samples', asyncHandler(async (_req, res) => {
    const s3Samples = await listSamples();

    const [labeledIds, verifiedIds] = await Promise.all([
        getLabeledSampleIds(),
        getVerifiedSampleIds(),
    ]);

    const samples = s3Samples.map(s => {
        let status = 'unlabeled';
        if (verifiedIds.has(s.id)) {
            status = 'verified';
        } else if (labeledIds.has(s.id)) {
            status = 'labeled';
        }

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

    const [detection, verified] = await Promise.all([
        getBoardDetectionBySampleId(id),
        isDetectionVerified(id),
    ]);

    let status = 'unlabeled';
    if (verified) {
        status = 'verified';
    } else if (detection) {
        status = 'labeled';
    }

    res.json({
        id: s3Sample.id,
        filename: `${s3Sample.id}.png`,
        status,
        created_at: s3Sample.lastModified.toISOString(),
        imageUrl: getCloudFrontUrl(s3Sample.id),
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

// GET /api/ml/detect/:id - Get cached detection result only
mlRouter.get('/detect/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const cached = await getBoardDetectionBySampleId(id);
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
