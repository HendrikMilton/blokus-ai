import { Router } from 'express';
import type { GameConfig, GameState, Move, PlayerColor, GamePieceDefinition } from '@blokus-ai/shared';
import { BlokusGame } from './engine';
import { PIECE_CATALOG } from '../pieces/service';
import { getAlphaZeroCheckpoint } from '../alphazero/database';
import { mlHttp } from '../../lib/http';

export const gameRouter = Router();

// Store active games in memory (in production, use Redis or DB)
interface StoredGame {
    game: BlokusGame;
    config: GameConfig;
}

const activeGames = new Map<string, StoredGame>();
let gameIdCounter = 0;

// ============================================================================
// AlphaZero Integration
// ============================================================================

interface AlphaZeroMoveResponse {
    piece_idx: number;
    orientation_idx: number;
    row: number;
    col: number;
    policy: number[];
    value: number;
}

// Piece IDs in same order as ML service (verified identical)
const PIECE_IDS = ['1', '2', 'I3', 'V3', 'I4', 'O4', 'T4', 'L4', 'S4',
    'F', 'I5', 'L5', 'N', 'P', 'T5', 'U', 'V', 'W', 'X', 'Y', 'Z'];

// Convert game state for ML service
const gameStateForML = (game: BlokusGame): Record<string, unknown> => {
    const state = game.getState();
    return {
        board: state.board,
        players: Object.fromEntries(
            Object.entries(state.players).map(([color, p]) => [
                color,
                {
                    remainingPieces: Array.from(p.remainingPieces),  // ML expects 'remainingPieces'
                    hasPlayed: p.hasPlayed,
                    isEliminated: p.isEliminated,
                    score: p.score,
                }
            ])
        ),
        currentPlayer: state.currentPlayer,
        turnNumber: state.turnNumber,
        isGameOver: state.isGameOver,
        consecutivePasses: state.consecutivePasses,
    };
};

// Convert ML response to game Move
const mlMoveToGameMove = (response: AlphaZeroMoveResponse, player: PlayerColor): Move => ({
    player,
    pieceId: PIECE_IDS[response.piece_idx],
    position: { row: response.row, col: response.col },
    orientationIndex: response.orientation_idx,
});

// Get AI model ID for a specific player color
const getAiModelForPlayer = (config: GameConfig, color: PlayerColor): string | undefined => {
    // Check per-player models first
    if (config.aiModels?.[color]) {
        return config.aiModels[color];
    }
    // Fall back to legacy single model
    return config.aiModelId;
};

// Get AI move from ML service
const getAlphaZeroMove = async (game: BlokusGame, checkpointId: string): Promise<Move | null> => {
    try {
        const checkpoint = await getAlphaZeroCheckpoint(checkpointId);
        if (!checkpoint) {
            console.warn(`AlphaZero checkpoint not found: ${checkpointId}`);
            return null;
        }

        const response = await mlHttp.post<AlphaZeroMoveResponse>(
            '/alphazero/move',
            {
                game_state: gameStateForML(game),
                model_path: checkpoint.modelPath,
                mcts_simulations: 100,  // Fast for interactive play
            }
        );

        return mlMoveToGameMove(response, game.getCurrentPlayer());
    } catch (err) {
        console.warn('AlphaZero inference failed:', err);
        return null;
    }
};

// Get supervised model move from ML service
const getSupervisedMove = async (game: BlokusGame): Promise<Move | null> => {
    try {
        const response = await mlHttp.post<AlphaZeroMoveResponse>(
            '/alphazero/move',
            {
                game_state: gameStateForML(game),
                model_path: 'models/supervised/best.pt',
                mcts_simulations: 100,
            }
        );

        return mlMoveToGameMove(response, game.getCurrentPlayer());
    } catch (err) {
        console.warn('Supervised model inference failed:', err);
        return null;
    }
};

// Get heuristic AI move from ML service
interface HeuristicMoveResponse {
    piece_idx: number;
    orientation_idx: number;
    row: number;
    col: number;
}

interface SmartMoveResponse extends HeuristicMoveResponse {
    time_ms: number;
    candidates_evaluated: number;
}

// Difficulty levels for heuristic AI
// Format: "heuristic" (default easy) or "heuristic:easy|medium|hard|expert"
type HeuristicDifficulty = 'easy' | 'medium' | 'hard' | 'expert';

const parseHeuristicModel = (modelId: string): { isHeuristic: boolean; difficulty: HeuristicDifficulty } => {
    if (!modelId.startsWith('heuristic')) {
        return { isHeuristic: false, difficulty: 'easy' };
    }
    if (modelId === 'heuristic') {
        return { isHeuristic: true, difficulty: 'easy' };
    }
    // Parse "heuristic:difficulty" format
    const parts = modelId.split(':');
    const difficulty = parts[1] as HeuristicDifficulty;
    if (['easy', 'medium', 'hard', 'expert'].includes(difficulty)) {
        return { isHeuristic: true, difficulty };
    }
    return { isHeuristic: true, difficulty: 'easy' };
};

const getHeuristicMove = async (game: BlokusGame, difficulty: HeuristicDifficulty = 'easy', targetPlayers?: number[]): Promise<Move | null> => {
    try {
        // Use SmartPlayer endpoint for medium+ difficulty
        if (difficulty !== 'easy') {
            const response = await mlHttp.post<SmartMoveResponse>(
                '/alphazero/smart-move',
                {
                    game_state: gameStateForML(game),
                    difficulty,
                    target_players: targetPlayers,
                }
            );

            return {
                player: game.getCurrentPlayer(),
                pieceId: PIECE_IDS[response.piece_idx],
                position: { row: response.row, col: response.col },
                orientationIndex: response.orientation_idx,
            };
        }

        // Use basic heuristic endpoint for easy
        const response = await mlHttp.post<HeuristicMoveResponse>(
            '/alphazero/heuristic-move',
            {
                game_state: gameStateForML(game),
                difficulty: 'easy',
                target_players: targetPlayers,
            }
        );

        return {
            player: game.getCurrentPlayer(),
            pieceId: PIECE_IDS[response.piece_idx],
            position: { row: response.row, col: response.col },
            orientationIndex: response.orientation_idx,
        };
    } catch (err) {
        console.warn('Heuristic inference failed:', err);
        return null;
    }
};

// Compute target player indices from config
const COLOR_TO_INDEX: Record<PlayerColor, number> = { blue: 0, yellow: 1, red: 2, green: 3 };

const getTargetPlayers = (config: GameConfig): number[] | undefined => {
    if (!config.aiTargetsHuman || !config.humanColors?.length) return undefined;
    return config.humanColors.map(c => COLOR_TO_INDEX[c]);
};

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

    activeGames.set(gameId, { game, config });

    res.json({
        gameId,
        config,
        state: serializeState(game),
    });
});

// GET /api/game/:id - Get game state
gameRouter.get('/:id', (req, res) => {
    const stored = activeGames.get(req.params.id);

    if (!stored) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    res.json({
        state: serializeState(stored.game),
        moveHistory: stored.game.getMoveHistory(),
    });
});

// GET /api/game/:id/legal-moves - Get legal moves for current player
gameRouter.get('/:id/legal-moves', (req, res) => {
    const stored = activeGames.get(req.params.id);

    if (!stored) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const moves = stored.game.getLegalMoves();
    res.json({ moves, count: moves.length });
});

// POST /api/game/:id/move - Make a move
gameRouter.post('/:id/move', (req, res) => {
    const stored = activeGames.get(req.params.id);

    if (!stored) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const move = req.body as Move;
    const result = stored.game.makeMove(move);

    if (!result.success) {
        res.status(400).json({ success: false, error: result.error });
        return;
    }

    res.json({
        success: true,
        state: serializeState(stored.game),
    });
});

// POST /api/game/:id/move-with-ai - Make a move and let AI respond
gameRouter.post('/:id/move-with-ai', async (req, res) => {
    const stored = activeGames.get(req.params.id);

    if (!stored) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const { game, config } = stored;
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
            continue;
        }

        let aiMove: Move | null = null;
        const currentPlayer = game.getCurrentPlayer();
        const modelId = getAiModelForPlayer(config, currentPlayer);
        const { isHeuristic, difficulty } = parseHeuristicModel(modelId || '');

        if (modelId === 'supervised') {
            // Supervised neural network model
            aiMove = await getSupervisedMove(game);
            if (!aiMove) {
                throw new Error('Supervised model failed to generate move');
            }
        } else if (isHeuristic) {
            // Heuristic AI (with difficulty level)
            aiMove = await getHeuristicMove(game, difficulty, getTargetPlayers(config));
            if (!aiMove) {
                throw new Error('Heuristic AI failed to generate move');
            }
        } else if (modelId) {
            // AlphaZero model selected - must use it, no fallback
            aiMove = await getAlphaZeroMove(game, modelId);
            if (!aiMove) {
                throw new Error('AlphaZero model failed to generate move');
            }
        } else {
            // Random AI
            aiMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
        }

        game.makeMove(aiMove);
        aiMoves.push(aiMove);
    }

    res.json({
        success: true,
        state: serializeState(game),
        aiMoves,
    });
});

// POST /api/game/:id/ai-move - Let AI make a move (for spectate mode)
gameRouter.post('/:id/ai-move', async (req, res) => {
    const stored = activeGames.get(req.params.id);

    if (!stored) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const { game, config } = stored;

    if (game.isGameOver()) {
        res.json({
            success: true,
            state: serializeState(game),
            aiMoves: [],
        });
        return;
    }

    const aiMoves: Move[] = [];
    const legalMoves = game.getLegalMoves();

    if (legalMoves.length === 0) {
        // No legal moves, pass
        game.pass();
    } else {
        let aiMove: Move | null = null;
        const currentPlayer = game.getCurrentPlayer();
        const modelId = getAiModelForPlayer(config, currentPlayer);
        const { isHeuristic, difficulty } = parseHeuristicModel(modelId || '');

        if (modelId === 'supervised') {
            // Supervised neural network model
            aiMove = await getSupervisedMove(game);
            if (!aiMove) {
                throw new Error('Supervised model failed to generate move');
            }
        } else if (isHeuristic) {
            // Heuristic AI (with difficulty level)
            aiMove = await getHeuristicMove(game, difficulty, getTargetPlayers(config));
            if (!aiMove) {
                throw new Error('Heuristic AI failed to generate move');
            }
        } else if (modelId) {
            // AlphaZero model selected - must use it, no fallback
            aiMove = await getAlphaZeroMove(game, modelId);
            if (!aiMove) {
                throw new Error('AlphaZero model failed to generate move');
            }
        } else {
            // Random AI
            aiMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
        }

        game.makeMove(aiMove);
        aiMoves.push(aiMove);
    }

    res.json({
        success: true,
        state: serializeState(game),
        aiMoves,
    });
});

// POST /api/game/:id/pass - Pass turn (when no legal moves)
gameRouter.post('/:id/pass', (req, res) => {
    const stored = activeGames.get(req.params.id);

    if (!stored) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const result = stored.game.pass();

    if (!result.success) {
        res.status(400).json({ success: false, error: result.error });
        return;
    }

    res.json({
        success: true,
        state: serializeState(stored.game),
    });
});

// POST /api/game/:id/pass-with-ai - Pass and let AI continue
gameRouter.post('/:id/pass-with-ai', async (req, res) => {
    const stored = activeGames.get(req.params.id);

    if (!stored) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const { game, config } = stored;
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
            continue;
        }

        let aiMove: Move | null = null;
        const currentPlayer = game.getCurrentPlayer();
        const modelId = getAiModelForPlayer(config, currentPlayer);
        const { isHeuristic, difficulty } = parseHeuristicModel(modelId || '');

        if (modelId === 'supervised') {
            // Supervised neural network model
            aiMove = await getSupervisedMove(game);
            if (!aiMove) {
                throw new Error('Supervised model failed to generate move');
            }
        } else if (isHeuristic) {
            // Heuristic AI (with difficulty level)
            aiMove = await getHeuristicMove(game, difficulty, getTargetPlayers(config));
            if (!aiMove) {
                throw new Error('Heuristic AI failed to generate move');
            }
        } else if (modelId) {
            // AlphaZero model selected - must use it, no fallback
            aiMove = await getAlphaZeroMove(game, modelId);
            if (!aiMove) {
                throw new Error('AlphaZero model failed to generate move');
            }
        } else {
            // Random AI
            aiMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
        }

        game.makeMove(aiMove);
        aiMoves.push(aiMove);
    }

    res.json({
        success: true,
        state: serializeState(game),
        aiMoves,
    });
});

// GET /api/game/:id/summary - Get game summary (scores, winner)
gameRouter.get('/:id/summary', (req, res) => {
    const stored = activeGames.get(req.params.id);

    if (!stored) {
        res.status(404).json({ error: 'Game not found' });
        return;
    }

    const summary = stored.game.getSummary();
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
