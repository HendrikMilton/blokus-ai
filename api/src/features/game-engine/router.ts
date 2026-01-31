import { Router } from 'express';
import type { GameConfig, GameState, Move, PlayerColor, GamePieceDefinition } from '@blokus-ai/shared';
import { BlokusGame } from './engine';
import { PIECE_CATALOG } from '../pieces/service';

export const gameRouter = Router();

// Store active games in memory (in production, use Redis or DB)
const activeGames = new Map<string, BlokusGame>();
let gameIdCounter = 0;

// Helper to serialize game state (convert Sets to arrays)
const serializeState = (game: BlokusGame): GameState => {
    const state = game.getState();
    return {
        board: state.board,
        players: {
            blue: { ...state.players.blue, remainingPieceIds: Array.from(state.players.blue.remainingPieces) },
            yellow: { ...state.players.yellow, remainingPieceIds: Array.from(state.players.yellow.remainingPieces) },
            red: { ...state.players.red, remainingPieceIds: Array.from(state.players.red.remainingPieces) },
            green: { ...state.players.green, remainingPieceIds: Array.from(state.players.green.remainingPieces) },
        },
        currentPlayer: state.currentPlayer,
        turnNumber: state.turnNumber,
        isGameOver: state.isGameOver,
        consecutivePasses: state.consecutivePasses,
    };
};

// GET /api/game/pieces - Get all piece definitions with orientations
gameRouter.get('/pieces', (_req, res) => {
    const pieces: GamePieceDefinition[] = PIECE_CATALOG.map(p => ({
        id: p.id,
        name: p.name,
        size: p.size,
        orientations: p.orientations,
    }));
    res.json(pieces);
});

// POST /api/game/new - Create a new game
gameRouter.post('/new', (req, res) => {
    const config = req.body as GameConfig;

    const gameId = `game-${++gameIdCounter}`;
    const game = new BlokusGame();

    activeGames.set(gameId, game);

    res.json({
        gameId,
        config,
        state: serializeState(game),
    });
});

// GET /api/game/:id - Get game state
gameRouter.get('/:id', (req, res) => {
    const game = activeGames.get(req.params.id);

    if (!game) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    res.json({
        state: serializeState(game),
        moveHistory: game.getMoveHistory(),
    });
});

// GET /api/game/:id/legal-moves - Get legal moves for current player
gameRouter.get('/:id/legal-moves', (req, res) => {
    const game = activeGames.get(req.params.id);

    if (!game) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const moves = game.getLegalMoves();
    res.json({ moves, count: moves.length });
});

// POST /api/game/:id/move - Make a move
gameRouter.post('/:id/move', (req, res) => {
    const game = activeGames.get(req.params.id);

    if (!game) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const move = req.body as Move;
    const result = game.makeMove(move);

    if (!result.success) {
        res.status(400).json({ success: false, error: result.error });
        return;
    }

    res.json({
        success: true,
        state: serializeState(game),
    });
});

// POST /api/game/:id/move-with-ai - Make a move and let AI respond
gameRouter.post('/:id/move-with-ai', (req, res) => {
    const game = activeGames.get(req.params.id);

    if (!game) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const { move, aiColors } = req.body as { move: Move; aiColors: PlayerColor[] };
    const aiColorSet = new Set(aiColors);

    // Make human's move
    const result = game.makeMove(move);

    if (!result.success) {
        res.status(400).json({ success: false, error: result.error });
        return;
    }

    const aiMoves: Move[] = [];

    // Let AI make moves until it's a human's turn again or game over
    while (!game.isGameOver() && aiColorSet.has(game.getCurrentPlayer())) {
        const legalMoves = game.getLegalMoves();

        if (legalMoves.length === 0) {
            game.pass();
        } else {
            // Random AI: pick a random move
            const randomMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
            game.makeMove(randomMove);
            aiMoves.push(randomMove);
        }
    }

    res.json({
        success: true,
        state: serializeState(game),
        aiMoves,
    });
});

// POST /api/game/:id/pass - Pass turn (when no legal moves)
gameRouter.post('/:id/pass', (req, res) => {
    const game = activeGames.get(req.params.id);

    if (!game) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const result = game.pass();

    if (!result.success) {
        res.status(400).json({ success: false, error: result.error });
        return;
    }

    res.json({
        success: true,
        state: serializeState(game),
    });
});

// POST /api/game/:id/pass-with-ai - Pass and let AI continue
gameRouter.post('/:id/pass-with-ai', (req, res) => {
    const game = activeGames.get(req.params.id);

    if (!game) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const { aiColors } = req.body as { aiColors: PlayerColor[] };
    const aiColorSet = new Set(aiColors);

    // Human passes
    const result = game.pass();

    if (!result.success) {
        res.status(400).json({ success: false, error: result.error });
        return;
    }

    const aiMoves: Move[] = [];

    // Let AI make moves until it's a human's turn again or game over
    while (!game.isGameOver() && aiColorSet.has(game.getCurrentPlayer())) {
        const legalMoves = game.getLegalMoves();

        if (legalMoves.length === 0) {
            game.pass();
        } else {
            const randomMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
            game.makeMove(randomMove);
            aiMoves.push(randomMove);
        }
    }

    res.json({
        success: true,
        state: serializeState(game),
        aiMoves,
    });
});

// GET /api/game/:id/summary - Get game summary (scores, winner)
gameRouter.get('/:id/summary', (req, res) => {
    const game = activeGames.get(req.params.id);

    if (!game) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const summary = game.getSummary();
    res.json(summary);
});

// DELETE /api/game/:id - Delete a game
gameRouter.delete('/:id', (req, res) => {
    const deleted = activeGames.delete(req.params.id);

    if (!deleted) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    res.json({ success: true });
});
