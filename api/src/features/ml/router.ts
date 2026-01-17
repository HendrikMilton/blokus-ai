import { Router } from 'express';
import { asyncHandler } from '../../lib/async-handler';
import { config } from '../../config';
import {
    DetectBoardRequest,
    DetectBoardResponse,
    AnalyzeBoardRequest,
    AnalyzeBoardResponse,
    TrainingSampleCreate,
    TrainingSampleUpdate,
    TrainRequest,
} from '@blokus-ai/shared';

export const mlRouter = Router();

// Helper to proxy requests to ML service
async function proxyToMlService(
    path: string,
    method: string,
    headers: Record<string, string>,
    body?: unknown,
): Promise<Response> {
    return fetch(`${config.ML_SERVICE_URL}${path}`, {
        method,
        headers: {
            'Content-Type': 'application/json',
            ...headers,
        },
        body: body ? JSON.stringify(body) : undefined,
    });
}

// POST /api/ml/detect-board - Detect board state from image
mlRouter.post('/detect-board', asyncHandler(async (req, res) => {
    const { image } = req.body as DetectBoardRequest;

    if (!image) {
        res.status(400).json({ error: 'Image is required' });
        return;
    }

    const response = await fetch(`${config.ML_SERVICE_URL}/detect-board/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
        },
        body: JSON.stringify({ image }),
    });

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service detect-board error:', error);
        res.status(response.status).json({ error: 'Board detection failed' });
        return;
    }

    const data: DetectBoardResponse = await response.json();
    res.json(data);
}));

// POST /api/ml/analyze - Analyze board and recommend move
mlRouter.post('/analyze', asyncHandler(async (req, res) => {
    const { boardState, selectedColor } = req.body as AnalyzeBoardRequest;

    if (!boardState || !selectedColor) {
        res.status(400).json({ error: 'Board state and selected color are required' });
        return;
    }

    const response = await fetch(`${config.ML_SERVICE_URL}/analyze/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
        },
        body: JSON.stringify({ boardState, selectedColor }),
    });

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service analyze error:', error);
        res.status(response.status).json({ error: 'Move analysis failed' });
        return;
    }

    const data: AnalyzeBoardResponse = await response.json();
    res.json(data);
}));

// ============================================================================
// Admin endpoints - proxy to ML service /admin/*
// ============================================================================

// POST /api/ml/admin/samples - Create training sample
mlRouter.post('/admin/samples', asyncHandler(async (req, res) => {
    const body = req.body as TrainingSampleCreate;

    if (!body.image) {
        res.status(400).json({ error: 'Image is required' });
        return;
    }

    const response = await proxyToMlService('/admin/samples', 'POST', {
        'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
    }, body);

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service admin/samples POST error:', error);
        res.status(response.status).json({ error: 'Failed to create sample' });
        return;
    }

    res.json(await response.json());
}));

// GET /api/ml/admin/samples - List training samples
mlRouter.get('/admin/samples', asyncHandler(async (req, res) => {
    const queryParams = new URLSearchParams();
    if (req.query.status) {
        queryParams.set('status', req.query.status as string);
    }
    if (req.query.page) {
        queryParams.set('page', req.query.page as string);
    }
    if (req.query.page_size) {
        queryParams.set('page_size', req.query.page_size as string);
    }

    const path = `/admin/samples${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    const response = await proxyToMlService(path, 'GET', {
        'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
    });

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service admin/samples GET error:', error);
        res.status(response.status).json({ error: 'Failed to list samples' });
        return;
    }

    res.json(await response.json());
}));

// GET /api/ml/admin/samples/:id - Get single sample
mlRouter.get('/admin/samples/:id', asyncHandler(async (req, res) => {
    const response = await proxyToMlService(`/admin/samples/${req.params.id}`, 'GET', {
        'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
    });

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service admin/samples/:id GET error:', error);
        res.status(response.status).json({ error: 'Failed to get sample' });
        return;
    }

    res.json(await response.json());
}));

// GET /api/ml/admin/samples/:id/image - Get sample image
mlRouter.get('/admin/samples/:id/image', asyncHandler(async (req, res) => {
    const response = await fetch(`${config.ML_SERVICE_URL}/admin/samples/${req.params.id}/image`, {
        method: 'GET',
        headers: {
            'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
        },
    });

    if (!response.ok) {
        res.status(response.status).json({ error: 'Failed to get image' });
        return;
    }

    // Forward the image response
    const contentType = response.headers.get('content-type') || 'image/jpeg';
    res.setHeader('Content-Type', contentType);
    res.setHeader('Cache-Control', 'public, max-age=86400');

    const buffer = await response.arrayBuffer();
    res.send(Buffer.from(buffer));
}));

// PATCH /api/ml/admin/samples/:id - Update sample
mlRouter.patch('/admin/samples/:id', asyncHandler(async (req, res) => {
    const body = req.body as TrainingSampleUpdate;

    const response = await proxyToMlService(`/admin/samples/${req.params.id}`, 'PATCH', {
        'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
    }, body);

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service admin/samples/:id PATCH error:', error);
        res.status(response.status).json({ error: 'Failed to update sample' });
        return;
    }

    res.json(await response.json());
}));

// DELETE /api/ml/admin/samples/:id - Delete sample
mlRouter.delete('/admin/samples/:id', asyncHandler(async (req, res) => {
    const response = await proxyToMlService(`/admin/samples/${req.params.id}`, 'DELETE', {
        'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
    });

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service admin/samples/:id DELETE error:', error);
        res.status(response.status).json({ error: 'Failed to delete sample' });
        return;
    }

    res.json(await response.json());
}));

// POST /api/ml/admin/samples/:id/predict - Re-run detection on sample
mlRouter.post('/admin/samples/:id/predict', asyncHandler(async (req, res) => {
    const response = await proxyToMlService(`/admin/samples/${req.params.id}/predict`, 'POST', {
        'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
    });

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service admin/samples/:id/predict POST error:', error);
        res.status(response.status).json({ error: 'Failed to repredict sample' });
        return;
    }

    res.json(await response.json());
}));

// GET /api/ml/admin/stats - Get admin statistics
mlRouter.get('/admin/stats', asyncHandler(async (req, res) => {
    const response = await proxyToMlService('/admin/stats', 'GET', {
        'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
    });

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service admin/stats GET error:', error);
        res.status(response.status).json({ error: 'Failed to get stats' });
        return;
    }

    res.json(await response.json());
}));

// POST /api/ml/admin/train - Train model
mlRouter.post('/admin/train', asyncHandler(async (req, res) => {
    const body = req.body as TrainRequest;

    const response = await proxyToMlService('/admin/train', 'POST', {
        'X-Correlation-ID': req.headers['x-correlation-id'] as string || crypto.randomUUID(),
    }, body);

    if (!response.ok) {
        const error = await response.text();
        console.error('ML service admin/train POST error:', error);
        res.status(response.status).json({ error: 'Failed to train model' });
        return;
    }

    res.json(await response.json());
}));
