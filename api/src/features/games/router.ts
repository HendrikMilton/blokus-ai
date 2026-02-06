import { Router } from 'express';
import { asyncHandler } from '../../lib/async-handler';
import { NotFoundError } from '../../error';
import { createGame, listGames, getGame, deleteGame, CreateGameData } from './database';

export const gamesRouter = Router();

// POST /api/games - Save a completed game
gamesRouter.post('/', asyncHandler(async (req, res) => {
    const data = req.body as CreateGameData;
    const game = await createGame(data);
    res.status(201).json(game);
}));

// GET /api/games - List game summaries
gamesRouter.get('/', asyncHandler(async (_req, res) => {
    const games = await listGames();
    res.json({ games });
}));

// GET /api/games/:id - Get full game with moves
gamesRouter.get('/:id', asyncHandler(async (req, res) => {
    const game = await getGame(req.params.id);
    if (!game) {
        throw new NotFoundError('Game not found');
    }
    res.json(game);
}));

// DELETE /api/games/:id - Delete a game
gamesRouter.delete('/:id', asyncHandler(async (req, res) => {
    const deleted = await deleteGame(req.params.id);
    if (!deleted) {
        throw new NotFoundError('Game not found');
    }
    res.json({ success: true });
}));
