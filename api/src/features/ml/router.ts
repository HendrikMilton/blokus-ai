import { Router } from 'express';
import { asyncHandler } from '../../lib/async-handler';
import { mlHttp, HttpError } from '../../lib/http';
import { config } from '../../config';
import type { DetectResponse } from '@blokus-ai/shared';
import { getBoardDetectionBySampleId, upsertBoardDetection, insertBoardDetection, getLabeledSampleIds, deleteBoardDetection, saveBoardDetectionCorrection, CorrectionData } from './database';

export const mlRouter = Router();

const handleMlError = (error: unknown, res: import('express').Response) => {
    if (error instanceof HttpError) {
        res.status(error.status).json({ error: error.body?.message || error.body?.detail || 'ML service error' });
    } else {
        res.status(503).json({ error: 'ML service unavailable' });
    }
};

// POST /api/ml/upload - Upload board image
mlRouter.post('/upload', asyncHandler(async (req, res) => {
    const { image } = req.body;

    if (!image) {
        res.status(400).json({ error: 'Image is required' });
        return;
    }

    try {
        const data = await mlHttp.post('/upload/', { image });
        res.json(data);
    } catch (error) {
        handleMlError(error, res);
    }
}));

// GET /api/ml/samples - List all samples
mlRouter.get('/samples', asyncHandler(async (_req, res) => {
    try {
        const data = await mlHttp.get<{ samples: { id: string; status: string }[] }>('/samples/');

        // Update status based on detection results
        const labeledIds = await getLabeledSampleIds();
        for (const sample of data.samples) {
            if (labeledIds.has(sample.id)) {
                sample.status = 'labeled';
            }
        }

        res.json(data);
    } catch (error) {
        handleMlError(error, res);
    }
}));

// GET /api/ml/samples/:id - Get single sample
mlRouter.get('/samples/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    try {
        const data = await mlHttp.get<{ id: string; filename: string; status: string; created_at: string }>(`/samples/${id}`);

        // Check if we have detection results - if so, status is 'labeled'
        const detection = await getBoardDetectionBySampleId(id);
        if (detection) {
            data.status = 'labeled';
        }

        res.json(data);
    } catch (error) {
        handleMlError(error, res);
    }
}));

// GET /api/ml/samples/:id/image - Get sample image
mlRouter.get('/samples/:id/image', asyncHandler(async (req, res) => {
    const { id } = req.params;

    try {
        const response = await fetch(`${config.ML_SERVICE_URL}/samples/${id}/image`);

        if (!response.ok) {
            res.status(response.status).json({ error: 'Image not found' });
            return;
        }

        const contentType = response.headers.get('content-type') || 'image/png';
        res.setHeader('Content-Type', contentType);
        res.setHeader('Cache-Control', 'public, max-age=31536000');

        const buffer = await response.arrayBuffer();
        res.send(Buffer.from(buffer));
    } catch {
        res.status(503).json({ error: 'ML service unavailable' });
    }
}));

// POST /api/ml/detect/:id - Run board detection and store result
mlRouter.post('/detect/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    try {
        const data = await mlHttp.post<DetectResponse>(`/detect/${id}`, {});
        await upsertBoardDetection(data);
        res.json(data);
    } catch (error) {
        handleMlError(error, res);
    }
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
        res.status(404).json({ error: 'Detection not found' });
        return;
    }
    res.json({ success: true });
}));
