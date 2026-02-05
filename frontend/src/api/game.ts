import { http } from './config';
import type {
    GameConfig,
    GameState,
    Move,
    GameSummary,
    PlayerColor,
    GamePieceDefinition,
} from '@blokus-ai/shared';

export interface NewGameResponse {
    gameId: string;
    config: GameConfig;
    state: GameState;
}

export interface MoveResponse {
    success: boolean;
    error?: string;
    state?: GameState;
    aiMoves?: Move[];
}

export interface LegalMovesResponse {
    moves: Move[];
    count: number;
}

export const GAME_API = {
    // Get all piece definitions with orientations
    getPieces: async (): Promise<GamePieceDefinition[]> => {
        return http.get<GamePieceDefinition[]>('/api/game/pieces');
    },

    // Create a new game
    newGame: async (config: GameConfig): Promise<NewGameResponse> => {
        return http.post<NewGameResponse>('/api/game/new', config);
    },

    // Get game state
    getGame: async (gameId: string): Promise<{ state: GameState; moveHistory: Move[] }> => {
        return http.get(`/api/game/${gameId}`);
    },

    // Get legal moves for current player
    getLegalMoves: async (gameId: string): Promise<LegalMovesResponse> => {
        return http.get(`/api/game/${gameId}/legal-moves`);
    },

    // Make a move
    makeMove: async (gameId: string, move: Move): Promise<MoveResponse> => {
        return http.post(`/api/game/${gameId}/move`, move);
    },

    // Make a move and let AI respond
    makeMoveWithAI: async (
        gameId: string,
        move: Move,
        aiColors: PlayerColor[]
    ): Promise<MoveResponse> => {
        return http.post(`/api/game/${gameId}/move-with-ai`, { move, aiColors });
    },

    // Pass turn
    pass: async (gameId: string): Promise<MoveResponse> => {
        return http.post(`/api/game/${gameId}/pass`, {});
    },

    // Let AI make a move (for spectate mode)
    aiMove: async (gameId: string): Promise<MoveResponse> => {
        return http.post(`/api/game/${gameId}/ai-move`, {});
    },

    // Pass and let AI continue
    passWithAI: async (gameId: string, aiColors: PlayerColor[]): Promise<MoveResponse> => {
        return http.post(`/api/game/${gameId}/pass-with-ai`, { aiColors });
    },

    // Get game summary
    getSummary: async (gameId: string): Promise<GameSummary> => {
        return http.get(`/api/game/${gameId}/summary`);
    },

    // Delete game
    deleteGame: async (gameId: string): Promise<void> => {
        await http.delete(`/api/game/${gameId}`);
    },
};

// ============================================================================
// AlphaZero Training API
// ============================================================================

export interface AlphaZeroConfig {
    gamesPerGeneration?: number;
    mctsSimulations?: number;
    mctsCPuct?: number;
    temperatureThreshold?: number;
    dirichletAlpha?: number;
    dirichletEpsilon?: number;
    numParallelGames?: number;
    mctsBatchSize?: number;
    inferenceBatchSize?: number;
    batchSize?: number;
    learningRate?: number;
    lrDecay?: string;
    trainingStepsPerGen?: number;
    trainingStepsMin?: number;
    trainingStepsMax?: number;
    adaptiveTrainingSteps?: boolean;
    replayBufferSize?: number;
    mixedPrecision?: boolean;
    numResBlocks?: number;
    channels?: number;
    promotionThreshold?: number;
    maxGenerations?: number;
    // Heuristic pretraining
    useHeuristicPretraining?: boolean;
    pretrainPositions?: number;
    pretrainSteps?: number;
    pretrainBatchSize?: number;
    pretrainLr?: number;
    // Research-backed improvements (KataGo, Gumbel MuZero, etc.)
    // ELO-based evaluation (replaces threshold-based promotion)
    useEloEvaluation?: boolean;
    eloInitial?: number;
    eloK?: number;
    eloPromotionThreshold?: number;
    eloGamesPerEval?: number;
    eloCompareGenerations?: number;
    // Gumbel MCTS (guaranteed policy improvement)
    useGumbelMcts?: boolean;
    gumbelMaxConsideredActions?: number;
    gumbelCVisit?: number;
    gumbelCScale?: number;
    // Playout cap randomization (KataGo - 50x efficiency)
    usePlayoutCapRandomization?: boolean;
    playoutCapFraction?: number;
    playoutCapFull?: number;
    playoutCapFast?: number;
    // Auxiliary prediction heads (denser gradients)
    useAuxiliaryLosses?: boolean;
    auxOwnershipWeight?: number;
    auxScoreWeight?: number;
    auxOpponentPolicyWeight?: number;
    // Policy target pruning
    usePolicyTargetPruning?: boolean;
    // Warm-start rollout blending
    useWarmStartRollouts?: boolean;
    warmStartInitialBlend?: number;
    warmStartAnnealGenerations?: number;
    // Sublinear replay window (prevents early overfitting)
    useSublinearWindow?: boolean;
    sublinearWindowC?: number;
    sublinearWindowAlpha?: number;
    sublinearWindowBeta?: number;
}

export interface AlphaZeroRun {
    id: string;
    name: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
    config: AlphaZeroConfig;
    currentGeneration: number;
    gamesGenerated: number;
    trainingSteps: number;
    policyLoss: number | null;
    valueLoss: number | null;
    bestWinRate: number | null;
    startedAt: string | null;
    completedAt: string | null;
    errorMessage: string | null;
    createdAt: string;
    updatedAt: string;
}

export interface AlphaZeroCheckpoint {
    id: string;
    runId: string;
    generation: number;
    step: number;
    policyLoss: number | null;
    valueLoss: number | null;
    evalWinRate: number | null;
    evalGames: number | null;
    heuristicWinRate: number | null;
    isBest: boolean;
    modelPath: string;
    createdAt: string;
}

export const ALPHAZERO_API = {
    // Get default config
    getDefaultConfig: async (): Promise<AlphaZeroConfig> => {
        return http.get('/api/ml/alphazero/config/default');
    },

    // Start a new training run
    startTraining: async (name: string, config?: Partial<AlphaZeroConfig>): Promise<AlphaZeroRun> => {
        return http.post('/api/ml/alphazero/runs', { name, config });
    },

    // List all training runs
    listRuns: async (): Promise<{ runs: AlphaZeroRun[] }> => {
        return http.get('/api/ml/alphazero/runs');
    },

    // Get a single run
    getRun: async (runId: string): Promise<AlphaZeroRun> => {
        return http.get(`/api/ml/alphazero/runs/${runId}`);
    },

    // Stop a running training
    stopTraining: async (runId: string): Promise<AlphaZeroRun> => {
        return http.post(`/api/ml/alphazero/runs/${runId}/stop`, {});
    },

    // Delete a training run
    deleteRun: async (runId: string): Promise<void> => {
        await http.delete(`/api/ml/alphazero/runs/${runId}`);
    },

    // Get all current training progress (for page refresh)
    getAllProgress: async (): Promise<Record<string, Record<string, unknown>>> => {
        return http.get('/api/ml/alphazero/progress');
    },

    // List checkpoints
    listCheckpoints: async (runId?: string, bestOnly?: boolean): Promise<{ checkpoints: AlphaZeroCheckpoint[] }> => {
        const params = new URLSearchParams();
        if (runId) {
            params.set('runId', runId);
        }
        if (bestOnly) {
            params.set('bestOnly', 'true');
        }
        const query = params.toString();
        return http.get(`/api/ml/alphazero/checkpoints${query ? `?${query}` : ''}`);
    },

    // Activate a checkpoint as best
    activateCheckpoint: async (checkpointId: string): Promise<void> => {
        await http.post(`/api/ml/alphazero/checkpoints/${checkpointId}/activate`, {});
    },

    // Export model to ONNX format
    exportOnnx: async (runId: string, checkpointId?: string): Promise<{
        run_id: string;
        onnx_path: string;
        file_size_mb: number;
        input_shapes: Record<string, number[]>;
        output_shapes: Record<string, number[]>;
        output_names: string[];
    }> => {
        const params = checkpointId ? `?checkpointId=${checkpointId}` : '';
        return http.get(`/api/ml/alphazero/runs/${runId}/export/onnx${params}`);
    },
};
