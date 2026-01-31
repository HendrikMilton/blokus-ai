import { Router } from 'express';
import type { GridLabels, GridAnalysisResult, PieceDefinition } from '@blokus-ai/shared';
import { analyzeGrid, PIECE_CATALOG } from './service';

export const piecesRouter = Router();

// GET /api/pieces/catalog - Get all piece definitions for display
piecesRouter.get('/catalog', (_req, res) => {
    const catalog: PieceDefinition[] = PIECE_CATALOG.map(p => ({
        id: p.id,
        name: p.name,
        size: p.size,
        shape: p.orientations[0], // Use first orientation as display shape
    }));
    res.json(catalog);
});

// POST /api/pieces/analyze - Analyze a grid to find which pieces are on/off board
piecesRouter.post('/analyze', (req, res) => {
    const { grid } = req.body as { grid: GridLabels };

    if (!grid || !Array.isArray(grid) || grid.length !== 20) {
        res.status(400).json({ error: 'Invalid grid: must be 20x20 array' });
        return;
    }

    for (const row of grid) {
        if (!Array.isArray(row) || row.length !== 20) {
            res.status(400).json({ error: 'Invalid grid: each row must have 20 cells' });
            return;
        }
    }

    const result: GridAnalysisResult = analyzeGrid(grid);
    res.json(result);
});
