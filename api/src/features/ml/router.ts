import { Router } from 'express';
import { asyncHandler } from '../../lib/async-handler';
import { config } from '../../config';
import {
    DetectBoardRequest,
    DetectBoardResponse,
    AnalyzeBoardRequest,
    AnalyzeBoardResponse,
} from '@blokus-ai/shared';

export const mlRouter = Router();

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
