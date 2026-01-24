import { Router } from 'express';
import { asyncHandler } from '../../lib/async-handler';
import { mlHttp, HttpError } from '../../lib/http';
import { config } from '../../config';

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
        const data = await mlHttp.get('/samples/');
        res.json(data);
    } catch (error) {
        handleMlError(error, res);
    }
}));

// GET /api/ml/samples/:id - Get single sample
mlRouter.get('/samples/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    try {
        const data = await mlHttp.get(`/samples/${id}`);
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
