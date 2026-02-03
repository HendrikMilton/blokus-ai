import { useState, useEffect, useMemo, useCallback } from 'react';
import toast from 'react-hot-toast';
import { Play, RotateCcw, FlipHorizontal2, Trophy, Users, Brain, Gamepad2, Plus, Square, X, Download } from 'lucide-react';
import { PLAYER_COLORS } from '@blokus-ai/shared';
import type {
    GameState,
    Move,
    PlayerColor,
    GamePieceDefinition,
    AlphaZeroProgressPayload,
} from '@blokus-ai/shared';
import { Button } from '../../../components/ui/Button';
import { Select } from '../../../components/ui/Select';
import { Table, Column } from '../../../components/ui/Table';
import { Modal } from '../../../components/ui/Modal';
import { Input } from '../../../components/ui/Input';
import { useSocketEvent } from '../../../hooks/useSocketEvent';
import { GAME_API, ALPHAZERO_API } from '../../../api/game';
import type { AlphaZeroRun, AlphaZeroCheckpoint, AlphaZeroConfig } from '../../../api/game';

// Mock AI model for random baseline (unused, kept for reference)
// interface MockAIModel {
//     id: string;
//     name: string;
//     status: 'ready';
// }

// Format seconds to human readable time
const formatEta = (seconds: number): string => {
    if (seconds <= 0) {
        return '--';
    }
    if (seconds < 60) {
        return `${Math.round(seconds)}s`;
    }
    if (seconds < 3600) {
        return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    }
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
};

// Phase display names
const PHASE_NAMES: Record<string, string> = {
    'self_play': 'Self-Play',
    'training': 'Training',
    'evaluation': 'Evaluation',
};

// Tab type
type Tab = 'play' | 'train';

// Game mode options with AI selection
const GAME_MODE_OPTIONS = [
    { value: 'vs-ai', label: 'vs AI' },
    { value: '1v1', label: '1v1 (Local)' },
];

// Game mode selection
type GameMode = 'vs-ai' | '1v1';

// Board cell component
const BoardCell = ({
    color,
    isHighlighted,
    isInvalidHighlight,
    isValidHighlight,
    validHintCount,
    isClickable,
    onClick,
}: {
    color: PlayerColor | null;
    isHighlighted: boolean;
    isInvalidHighlight: boolean;
    isValidHighlight: boolean;
    validHintCount: number;
    isClickable: boolean;
    onClick: () => void;
}) => {
    // Calculate hint opacity based on how many placements overlap (more = brighter)
    const hintOpacity = validHintCount > 0 ? Math.min(0.15 + validHintCount * 0.1, 0.5) : 0;

    // Determine ring color: green if valid placement, red if invalid, white otherwise
    const ringColor = isValidHighlight
        ? 'ring-green-500'
        : isInvalidHighlight
            ? 'ring-red-500'
            : isHighlighted
                ? 'ring-white'
                : '';

    return (
        <button
            onClick={onClick}
            className={`
                w-full h-full transition-all relative
                ${ringColor ? `ring-2 ${ringColor} ring-inset` : ''}
                ${isClickable ? 'cursor-pointer hover:brightness-125' : ''}
            `}
            style={{
                backgroundColor: color ? PLAYER_COLORS[color] : '#1a1a2e',
            }}
        >
            {validHintCount > 0 && !isHighlighted && !isInvalidHighlight && !isValidHighlight && (
                <div
                    className="absolute inset-0 bg-green-500"
                    style={{ opacity: hintOpacity }}
                />
            )}
        </button>
    );
};

// Piece shape preview
const PiecePreview = ({
    piece,
    orientationIndex,
    color,
    isSelected,
    isUsed,
    onClick,
}: {
    piece: GamePieceDefinition;
    orientationIndex: number;
    color: PlayerColor;
    isSelected: boolean;
    isUsed: boolean;
    onClick: () => void;
}) => {
    const orientation = piece.orientations[orientationIndex] || piece.orientations[0];

    // Calculate bounding box
    const minRow = Math.min(...orientation.map(([r]) => r));
    const maxRow = Math.max(...orientation.map(([r]) => r));
    const minCol = Math.min(...orientation.map(([, c]) => c));
    const maxCol = Math.max(...orientation.map(([, c]) => c));

    const rows = maxRow - minRow + 1;
    const cols = maxCol - minCol + 1;

    const filledCells = new Set(
        orientation.map(([r, c]) => `${r - minRow},${c - minCol}`)
    );

    return (
        <button
            onClick={onClick}
            disabled={isUsed}
            className={`
                p-1 rounded transition-all
                ${isSelected ? 'bg-white/20 ring-2 ring-white' : 'hover:bg-white/10'}
                ${isUsed ? 'opacity-30 cursor-not-allowed' : 'cursor-pointer'}
            `}
            title={`${piece.name} (${piece.size} cells)`}
        >
            <div
                className="grid gap-px"
                style={{
                    gridTemplateColumns: `repeat(${cols}, 8px)`,
                    gridTemplateRows: `repeat(${rows}, 8px)`,
                }}
            >
                {Array.from({ length: rows * cols }).map((_, idx) => {
                    const r = Math.floor(idx / cols);
                    const c = idx % cols;
                    const isFilled = filledCells.has(`${r},${c}`);
                    return (
                        <div
                            key={idx}
                            className="rounded-[1px]"
                            style={{
                                backgroundColor: isFilled ? PLAYER_COLORS[color] : 'transparent',
                            }}
                        />
                    );
                })}
            </div>
        </button>
    );
};

// Player score display
const PlayerScore = ({
    color,
    score,
    remainingPieces,
    totalPieces,
    isCurrentTurn,
    isHuman,
    isEliminated,
    aiModelName,
}: {
    color: PlayerColor;
    score: number;
    remainingPieces: number;
    totalPieces: number;
    isCurrentTurn: boolean;
    isHuman: boolean;
    isEliminated: boolean;
    aiModelName?: string;
}) => {
    const playedPieces = totalPieces - remainingPieces;
    // Total squares = 89 (sum of all piece sizes: 1+2+3+3+4+4+4+4+4+5+5+5+5+5+5+5+5+5+5+5+5 = 89)
    const totalSquares = 89;
    const playedSquares = score; // Score equals squares placed
    const remainingSquares = totalSquares - playedSquares;

    return (
        <div
            className={`
                flex items-center gap-2 px-3 py-2 rounded-lg transition-all
                ${isCurrentTurn ? 'ring-2 ring-white bg-white/10' : 'bg-surface-dim'}
                ${isEliminated ? 'opacity-50' : ''}
            `}
        >
            <div
                className="w-4 h-4 rounded"
                style={{ backgroundColor: PLAYER_COLORS[color] }}
            />
            <span className="text-sm font-medium text-foreground capitalize">{color}</span>
            <span className="text-xs text-secondary">
                {isHuman ? '(You)' : aiModelName ? `(${aiModelName})` : '(AI)'}
            </span>
            {isEliminated && (
                <span className="text-xs text-red-400">Out</span>
            )}
            <div className="ml-auto text-right">
                <div className="text-sm text-foreground">{score} pts</div>
                <div className="text-xs text-secondary">
                    {playedPieces}/{totalPieces} pcs · {remainingSquares} sq left
                </div>
            </div>
        </div>
    );
};

// Columns are defined inside the component to have access to handlers

export const GymnasiumPage = () => {
    // Tab state
    const [activeTab, setActiveTab] = useState<Tab>('play');

    // AlphaZero state
    const [alphaZeroRuns, setAlphaZeroRuns] = useState<AlphaZeroRun[]>([]);
    const [alphaZeroCheckpoints, setAlphaZeroCheckpoints] = useState<AlphaZeroCheckpoint[]>([]);
    const [isLoadingRuns, setIsLoadingRuns] = useState(false);
    const [showTrainModal, setShowTrainModal] = useState(false);
    const [newTrainName, setNewTrainName] = useState('');
    const [trainConfig, setTrainConfig] = useState<Partial<AlphaZeroConfig>>({
        mctsSimulations: 400,
        gamesPerGeneration: 100,
        maxGenerations: 100,
        batchSize: 4096,
        numResBlocks: 15,
        channels: 256,
        numParallelGames: 8,
        mctsBatchSize: 16,
        inferenceBatchSize: 32,
    });
    const [selectedRunForDetails, setSelectedRunForDetails] = useState<AlphaZeroRun | null>(null);
    const [trainingProgress, setTrainingProgress] = useState<Record<string, AlphaZeroProgressPayload>>({});

    // AI model selection
    const [selectedAIModel, setSelectedAIModel] = useState<string>('random');

    // Build model options from checkpoints - show best checkpoint per run (highest win rate)
    const aiModelOptions = useMemo(() => {
        const options: { value: string; label: string }[] = [{ value: 'random', label: 'Random AI' }];

        // Group checkpoints by run and pick the one with highest win rate
        const checkpointsByRun = new Map<string, typeof alphaZeroCheckpoints[0]>();
        for (const cp of alphaZeroCheckpoints) {
            const existing = checkpointsByRun.get(cp.runId);
            const cpWinRate = cp.evalWinRate ?? 0;
            const existingWinRate = existing?.evalWinRate ?? 0;
            if (!existing || cpWinRate > existingWinRate) {
                checkpointsByRun.set(cp.runId, cp);
            }
        }

        // Add options for each run's best checkpoint
        for (const [runId, cp] of checkpointsByRun) {
            const run = alphaZeroRuns.find(r => r.id === runId);
            if (run) {
                options.push({
                    value: cp.id,
                    label: `${run.name} (Gen ${cp.generation}, ${cp.evalWinRate !== null ? `${(cp.evalWinRate * 100).toFixed(0)}%` : 'N/A'})`,
                });
            }
        }
        return options;
    }, [alphaZeroRuns, alphaZeroCheckpoints]);

    const selectedModelName = aiModelOptions.find(m => m.value === selectedAIModel)?.label || 'Random AI';

    // Game state
    const [gameId, setGameId] = useState<string | null>(null);
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [pieces, setPieces] = useState<GamePieceDefinition[]>([]);
    const [legalMoves, setLegalMoves] = useState<Move[]>([]);

    // UI state
    const [gameMode, setGameMode] = useState<GameMode>('vs-ai');
    const [selectedPieceId, setSelectedPieceId] = useState<string | null>(null);
    const [orientationIndex, setOrientationIndex] = useState(0);
    const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    // Load AlphaZero runs and checkpoints
    const loadAlphaZeroData = useCallback(async () => {
        setIsLoadingRuns(true);
        try {
            const [runsRes, checkpointsRes] = await Promise.all([
                ALPHAZERO_API.listRuns(),
                ALPHAZERO_API.listCheckpoints(), // All checkpoints
            ]);
            setAlphaZeroRuns(runsRes.runs);
            setAlphaZeroCheckpoints(checkpointsRes.checkpoints);
        } catch (err) {
            console.error('Failed to load AlphaZero data:', err);
        } finally {
            setIsLoadingRuns(false);
        }
    }, []);

    // Load AlphaZero data on mount
    useEffect(() => {
        loadAlphaZeroData();
    }, [loadAlphaZeroData]);

    // Listen for AlphaZero progress events
    useSocketEvent('alphazero-progress', (data) => {
        setTrainingProgress(prev => ({
            ...prev,
            [data.runId]: data,
        }));
    }, []);

    // Listen for run completion/failure to reload data
    useSocketEvent('alphazero-run-completed', () => {
        loadAlphaZeroData();
        toast.success('Training completed!');
    }, [loadAlphaZeroData]);

    useSocketEvent('alphazero-run-failed', (data) => {
        loadAlphaZeroData();
        toast.error(`Training failed: ${data.errorMessage}`);
    }, [loadAlphaZeroData]);

    useSocketEvent('alphazero-run-stopped', () => {
        loadAlphaZeroData();
        toast('Training stopped');
    }, [loadAlphaZeroData]);

    useSocketEvent('alphazero-checkpoint-created', (data) => {
        if (data.checkpoint.isBest) {
            loadAlphaZeroData();
            toast.success(`New best model! Win rate: ${((data.checkpoint.evalWinRate ?? 0) * 100).toFixed(0)}%`);
        }
    }, [loadAlphaZeroData]);

    // Start new training
    const handleStartTraining = async () => {
        if (!newTrainName.trim()) {
            toast.error('Please enter a name for the training run');
            return;
        }

        try {
            const run = await ALPHAZERO_API.startTraining(newTrainName.trim(), trainConfig);
            toast.success(`Training "${run.name}" started!`);
            setShowTrainModal(false);
            setNewTrainName('');
            // Reset config to defaults
            setTrainConfig({
                mctsSimulations: 400,
                gamesPerGeneration: 100,
                maxGenerations: 100,
                batchSize: 4096,
                numResBlocks: 15,
                channels: 256,
            });
            loadAlphaZeroData();
        } catch {
            toast.error('Failed to start training');
        }
    };

    // Stop training
    const handleStopTraining = async (runId: string) => {
        try {
            await ALPHAZERO_API.stopTraining(runId);
            toast.success('Training stopped');
            loadAlphaZeroData();
        } catch {
            toast.error('Failed to stop training');
        }
    };

    // Delete training run
    const handleDeleteRun = async (runId: string) => {
        if (!confirm('Delete this training run and all its checkpoints?')) {
            return;
        }

        try {
            await ALPHAZERO_API.deleteRun(runId);
            toast.success('Training run deleted');
            loadAlphaZeroData();
        } catch {
            toast.error('Failed to delete training run');
        }
    };

    // Export model to ONNX
    const [isExporting, setIsExporting] = useState(false);
    const handleExportOnnx = async (runId: string) => {
        setIsExporting(true);
        try {
            const result = await ALPHAZERO_API.exportOnnx(runId);
            toast.success(`Model exported: ${result.file_size_mb.toFixed(1)} MB`);
        } catch {
            toast.error('Failed to export model to ONNX');
        } finally {
            setIsExporting(false);
        }
    };

    // AI Runs table columns (defined here to access handlers)
    const aiRunColumns: Column<AlphaZeroRun>[] = useMemo(() => [
        { key: 'name', header: 'Name', sortable: true },
        {
            key: 'status',
            header: 'Status',
            render: (run) => (
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    run.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                        run.status === 'running' ? 'bg-yellow-500/20 text-yellow-400' :
                            run.status === 'pending' ? 'bg-blue-500/20 text-blue-400' :
                                run.status === 'stopped' ? 'bg-gray-500/20 text-gray-400' :
                                    'bg-red-500/20 text-red-400'
                }`}>
                    {run.status}
                </span>
            ),
        },
        {
            key: 'currentGeneration',
            header: 'Gen',
            render: (run) => run.currentGeneration > 0 ? `${run.currentGeneration}` : '-',
        },
        {
            key: 'gamesGenerated',
            header: 'Games',
            render: (run) => run.gamesGenerated > 0 ? run.gamesGenerated.toLocaleString() : '-',
        },
        {
            key: 'bestWinRate',
            header: 'Win Rate',
            render: (run) => run.bestWinRate !== null ? `${(run.bestWinRate * 100).toFixed(0)}%` : '-',
        },
        {
            key: 'createdAt',
            header: 'Created',
            render: (run) => new Date(run.createdAt).toLocaleDateString(),
            sortable: true,
        },
        {
            key: 'actions',
            header: '',
            width: '80px',
            render: (run) => (
                <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
                    {run.status === 'running' && (
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleStopTraining(run.id)}
                            title="Stop training"
                        >
                            <Square className="w-4 h-4" />
                        </Button>
                    )}
                    {run.status !== 'running' && (
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleDeleteRun(run.id)}
                            title="Delete run"
                        >
                            <X className="w-4 h-4" />
                        </Button>
                    )}
                </div>
            ),
        },
    ], [handleStopTraining, handleDeleteRun]);

    // Determine which colors are human/AI based on mode
    const humanColors: PlayerColor[] = useMemo(() => {
        if (gameMode === 'vs-ai') {
            return ['blue', 'red']; // Human plays blue and red (opposite corners)
        }
        return ['blue', 'yellow', 'red', 'green']; // All human in 1v1 (local multiplayer)
    }, [gameMode]);

    const aiColors: PlayerColor[] = useMemo(() => {
        if (gameMode === 'vs-ai') {
            return ['yellow', 'green'];
        }
        return [];
    }, [gameMode]);

    const isHumanTurn = gameState && humanColors.includes(gameState.currentPlayer);

    // Load pieces on mount
    useEffect(() => {
        GAME_API.getPieces().then(setPieces).catch(() => {
            toast.error('Failed to load pieces');
        });
    }, []);

    // Track whether we've loaded legal moves for the current turn
    const [legalMovesLoaded, setLegalMovesLoaded] = useState(false);

    // Load legal moves when game state changes
    useEffect(() => {
        if (gameId && gameState && !gameState.isGameOver && isHumanTurn) {
            setLegalMovesLoaded(false); // Reset for new turn
            GAME_API.getLegalMoves(gameId).then(res => {
                setLegalMoves(res.moves);
                setLegalMovesLoaded(true);
            }).catch(() => {
                setLegalMoves([]);
                setLegalMovesLoaded(true);
            });
        } else {
            setLegalMoves([]);
            setLegalMovesLoaded(false);
        }
    }, [gameId, gameState, isHumanTurn]);

    // Start new game
    const startGame = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await GAME_API.newGame({
                mode: gameMode,
                humanColors,
                aiColors,
                aiModelId: selectedAIModel !== 'random' ? selectedAIModel : undefined,
            });
            setGameId(response.gameId);
            setGameState(response.state);
            setSelectedPieceId(null);
            setOrientationIndex(0);
            toast.success('Game started!');
        } catch {
            toast.error('Failed to start game');
        } finally {
            setIsLoading(false);
        }
    }, [gameMode, humanColors, aiColors, selectedAIModel]);

    // Get selected piece
    const selectedPiece = useMemo(() => {
        return pieces.find(p => p.id === selectedPieceId) || null;
    }, [pieces, selectedPieceId]);

    // Rotate piece (cycle through orientations, staying in same "flip group")
    const rotatePiece = useCallback(() => {
        if (!selectedPiece) {
            return;
        }
        const total = selectedPiece.orientations.length;
        // For pieces with 8 orientations: 0-3 are rotations, 4-7 are flipped rotations
        // For pieces with fewer, just cycle through all
        if (total <= 4) {
            setOrientationIndex(prev => (prev + 1) % total);
        } else {
            // Stay in current half (normal or flipped)
            const isFlipped = orientationIndex >= 4;
            const base = isFlipped ? 4 : 0;
            const maxInGroup = isFlipped ? total - 4 : Math.min(4, total);
            const currentInGroup = orientationIndex - base;
            const nextInGroup = (currentInGroup + 1) % maxInGroup;
            setOrientationIndex(base + nextInGroup);
        }
    }, [selectedPiece, orientationIndex]);

    // Flip piece (toggle between normal and flipped versions)
    const flipPiece = useCallback(() => {
        if (!selectedPiece) {
            return;
        }
        const total = selectedPiece.orientations.length;
        if (total <= 4) {
            // Piece has no distinct flip (symmetric), just rotate
            setOrientationIndex(prev => (prev + 1) % total);
        } else {
            // Toggle between normal (0-3) and flipped (4-7)
            const isFlipped = orientationIndex >= 4;
            if (isFlipped) {
                // Go back to normal, same rotation position
                setOrientationIndex(orientationIndex - 4);
            } else {
                // Go to flipped version
                const flippedIndex = orientationIndex + 4;
                setOrientationIndex(flippedIndex < total ? flippedIndex : 4);
            }
        }
    }, [selectedPiece, orientationIndex]);

    // Get cells that would be filled by placing piece at position
    const getPlacementCells = useCallback((row: number, col: number): { row: number; col: number }[] | null => {
        if (!selectedPiece) {
            return null;
        }
        const orientation = selectedPiece.orientations[orientationIndex];
        if (!orientation) {
            return null;
        }

        return orientation.map(([r, c]) => ({
            row: row + r,
            col: col + c,
        }));
    }, [selectedPiece, orientationIndex]);

    // Check if placement is valid
    const isValidPlacement = useCallback((row: number, col: number): boolean => {
        if (!gameState || !selectedPiece) {
            return false;
        }

        return legalMoves.some(move =>
            move.pieceId === selectedPieceId &&
            move.orientationIndex === orientationIndex &&
            move.position.row === row &&
            move.position.col === col
        );
    }, [gameState, selectedPiece, selectedPieceId, orientationIndex, legalMoves]);

    // Make a move
    const makeMove = useCallback(async (row: number, col: number) => {
        if (!gameId || !gameState || !selectedPieceId || gameState.isGameOver) {
            return;
        }

        const move: Move = {
            player: gameState.currentPlayer,
            pieceId: selectedPieceId,
            position: { row, col },
            orientationIndex,
        };

        setIsLoading(true);
        try {
            const response = await GAME_API.makeMoveWithAI(gameId, move, aiColors);
            if (response.success && response.state) {
                setGameState(response.state);
                setSelectedPieceId(null);
                setOrientationIndex(0);

                if (response.aiMoves && response.aiMoves.length > 0) {
                    toast.success(`AI played ${response.aiMoves.length} move(s)`);
                }
            } else {
                toast.error(response.error || 'Invalid move');
            }
        } catch {
            toast.error('Failed to make move');
        } finally {
            setIsLoading(false);
        }
    }, [gameId, gameState, selectedPieceId, orientationIndex, aiColors]);

    // Handle cell click
    const handleCellClick = useCallback((row: number, col: number) => {
        if (!isHumanTurn || !selectedPieceId) {
            return;
        }

        if (isValidPlacement(row, col)) {
            makeMove(row, col);
        }
    }, [isHumanTurn, selectedPieceId, isValidPlacement, makeMove]);

    // Pass turn
    const handlePass = useCallback(async (isAutoPass = false) => {
        if (!gameId || !gameState || gameState.isGameOver) {
            return;
        }

        setIsLoading(true);
        try {
            const response = await GAME_API.passWithAI(gameId, aiColors);
            if (response.success && response.state) {
                setGameState(response.state);
                if (isAutoPass) {
                    toast(`No moves for ${gameState.currentPlayer} - auto-passed`, { icon: '⏭️' });
                } else {
                    toast.success('Passed turn');
                }
            } else {
                toast.error(response.error || 'Cannot pass');
            }
        } catch {
            toast.error('Failed to pass');
        } finally {
            setIsLoading(false);
        }
    }, [gameId, gameState, aiColors]);

    // Auto-pass when no legal moves are available
    useEffect(() => {
        // Only auto-pass if:
        // 1. Game is active
        // 2. It's a human's turn
        // 3. Legal moves have been loaded for this turn
        // 4. There are no legal moves
        // 5. Not currently loading (to prevent race conditions)
        if (
            gameId &&
            gameState &&
            !gameState.isGameOver &&
            isHumanTurn &&
            legalMovesLoaded &&
            legalMoves.length === 0 &&
            !isLoading
        ) {
            // Small delay to let UI render the "no moves" state briefly
            const timer = setTimeout(() => {
                handlePass(true);
            }, 800);
            return () => clearTimeout(timer);
        }
    }, [gameId, gameState, isHumanTurn, legalMoves, legalMovesLoaded, isLoading, handlePass]);

    // Check if current hover position is a valid placement
    const isHoverValid = useMemo(() => {
        if (!hoveredCell || !gameState || !selectedPiece) {
            return false;
        }
        return legalMoves.some(move =>
            move.pieceId === selectedPieceId &&
            move.orientationIndex === orientationIndex &&
            move.position.row === hoveredCell.row &&
            move.position.col === hoveredCell.col
        );
    }, [hoveredCell, gameState, selectedPiece, selectedPieceId, orientationIndex, legalMoves]);

    // Get highlighted cells for preview (valid and invalid/out-of-bounds)
    const { highlightedCells, invalidCells } = useMemo(() => {
        if (!hoveredCell || !selectedPieceId || !selectedPiece || !isHumanTurn) {
            return { highlightedCells: new Set<string>(), invalidCells: new Set<string>() };
        }

        const cells = getPlacementCells(hoveredCell.row, hoveredCell.col);
        if (!cells) {
            return { highlightedCells: new Set<string>(), invalidCells: new Set<string>() };
        }

        const valid = new Set<string>();
        const invalid = new Set<string>();

        for (const c of cells) {
            const key = `${c.row},${c.col}`;
            if (c.row >= 0 && c.row < 20 && c.col >= 0 && c.col < 20) {
                valid.add(key);
            } else {
                invalid.add(key);
            }
        }

        // If any cell is out of bounds, mark all as invalid visually
        if (invalid.size > 0) {
            return { highlightedCells: new Set<string>(), invalidCells: valid };
        }

        return { highlightedCells: valid, invalidCells: new Set<string>() };
    }, [hoveredCell, selectedPiece, isHumanTurn, getPlacementCells]);

    // Get cell counts for valid placements (green hint) - current orientation only
    // Count how many valid placements include each cell (for opacity stacking)
    const validPlacementCounts = useMemo(() => {
        if (!selectedPiece || !isHumanTurn) {
            return new Map<string, number>();
        }

        const counts = new Map<string, number>();

        // Get legal moves for this piece with current orientation
        const relevantMoves = legalMoves.filter(
            move => move.pieceId === selectedPieceId && move.orientationIndex === orientationIndex
        );

        // Count how many placements cover each cell
        for (const move of relevantMoves) {
            const orientation = selectedPiece.orientations[move.orientationIndex];
            if (orientation) {
                for (const [r, c] of orientation) {
                    const row = move.position.row + r;
                    const col = move.position.col + c;
                    const key = `${row},${col}`;
                    counts.set(key, (counts.get(key) || 0) + 1);
                }
            }
        }

        return counts;
    }, [selectedPiece, selectedPieceId, orientationIndex, legalMoves, isHumanTurn]);

    // Render board
    const renderBoard = () => {
        if (!gameState) {
            return null;
        }

        return (
            <div
                className="grid bg-surface rounded-lg p-2 gap-0.5"
                style={{
                    gridTemplateColumns: 'repeat(20, 1fr)',
                    gridTemplateRows: 'repeat(20, 1fr)',
                    aspectRatio: '1 / 1',
                }}
            >
                {gameState.board.map((row, ri) =>
                    row.map((cell, ci) => (
                        <div
                            key={`${ri}-${ci}`}
                            className="relative"
                            onMouseEnter={() => setHoveredCell({ row: ri, col: ci })}
                            onMouseLeave={() => setHoveredCell(null)}
                        >
                            <BoardCell
                                color={cell}
                                isHighlighted={highlightedCells.has(`${ri},${ci}`)}
                                isInvalidHighlight={invalidCells.has(`${ri},${ci}`)}
                                isValidHighlight={highlightedCells.has(`${ri},${ci}`) && isHoverValid}
                                validHintCount={validPlacementCounts.get(`${ri},${ci}`) || 0}
                                isClickable={selectedPieceId !== null && isValidPlacement(ri, ci)}
                                onClick={() => handleCellClick(ri, ci)}
                            />
                        </div>
                    ))
                )}
            </div>
        );
    };

    // Render piece palette for current player
    const renderPiecePalette = () => {
        if (!gameState || !isHumanTurn) {
            return null;
        }

        const currentPlayer = gameState.currentPlayer;
        const remainingPieceIds = new Set(gameState.players[currentPlayer].remainingPieceIds);

        return (
            <div className="space-y-2">
                <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium">Your Pieces ({currentPlayer})</h3>
                    {selectedPiece && (
                        <div className="flex gap-1">
                            <Button
                                size="sm"
                                variant="secondary"
                                onClick={rotatePiece}
                                title="Rotate (R)"
                            >
                                <RotateCcw className="w-4 h-4" />
                            </Button>
                            <Button
                                size="sm"
                                variant="secondary"
                                onClick={flipPiece}
                                title="Flip (F)"
                            >
                                <FlipHorizontal2 className="w-4 h-4" />
                            </Button>
                        </div>
                    )}
                </div>
                <div className="flex flex-wrap gap-1 p-2 bg-surface-dim rounded-lg max-h-64 overflow-y-auto">
                    {pieces
                        .sort((a, b) => b.size - a.size)
                        .map(piece => (
                            <PiecePreview
                                key={piece.id}
                                piece={piece}
                                orientationIndex={selectedPieceId === piece.id ? orientationIndex : 0}
                                color={currentPlayer}
                                isSelected={selectedPieceId === piece.id}
                                isUsed={!remainingPieceIds.has(piece.id)}
                                onClick={() => {
                                    setSelectedPieceId(piece.id);
                                    setOrientationIndex(0);
                                }}
                            />
                        ))}
                </div>
            </div>
        );
    };

    // Game over screen
    const renderGameOver = () => {
        if (!gameState?.isGameOver) {
            return null;
        }

        const scores = Object.entries(gameState.players)
            .map(([color, player]) => ({ color: color as PlayerColor, score: player.score }))
            .sort((a, b) => b.score - a.score);

        return (
            <div className="absolute inset-0 bg-black/80 flex items-center justify-center rounded-lg">
                <div className="bg-surface p-6 rounded-lg text-center space-y-4">
                    <Trophy className="w-12 h-12 mx-auto text-yellow-500" />
                    <h2 className="text-xl font-bold text-foreground">Game Over!</h2>
                    <div className="space-y-2">
                        {scores.map(({ color, score }, idx) => (
                            <div
                                key={color}
                                className={`flex items-center gap-3 px-4 py-2 rounded ${idx === 0 ? 'bg-yellow-500/20' : ''}`}
                            >
                                <span className="text-sm font-medium w-4 text-foreground">{idx + 1}.</span>
                                <div
                                    className="w-4 h-4 rounded"
                                    style={{ backgroundColor: PLAYER_COLORS[color] }}
                                />
                                <span className="capitalize text-foreground">{color}</span>
                                <span className="ml-auto font-bold text-foreground">{score}</span>
                            </div>
                        ))}
                    </div>
                    <Button onClick={startGame}>Play Again</Button>
                </div>
            </div>
        );
    };

    // Get sorted available pieces for keyboard navigation
    const availablePieces = useMemo(() => {
        if (!gameState || !isHumanTurn) {
            return [];
        }
        const remainingIds = new Set(gameState.players[gameState.currentPlayer].remainingPieceIds);
        return pieces
            .filter(p => remainingIds.has(p.id))
            .sort((a, b) => b.size - a.size);
    }, [gameState, pieces, isHumanTurn]);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Don't trigger if typing in an input
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
                return;
            }

            // Only handle game shortcuts when on play tab
            if (activeTab !== 'play') {
                return;
            }

            switch (e.key.toLowerCase()) {
            case 'r':
                if (selectedPieceId) {
                    rotatePiece();
                }
                break;
            case 'f':
                if (selectedPieceId) {
                    flipPiece();
                }
                break;
            case 'escape':
                setSelectedPieceId(null);
                setOrientationIndex(0);
                break;
            case 'arrowleft':
            case 'arrowup': {
                // Previous piece
                if (availablePieces.length === 0) {
                    break;
                }
                e.preventDefault();
                const currentIdx = selectedPieceId
                    ? availablePieces.findIndex(p => p.id === selectedPieceId)
                    : 0;
                const prevIdx = currentIdx <= 0 ? availablePieces.length - 1 : currentIdx - 1;
                setSelectedPieceId(availablePieces[prevIdx].id);
                setOrientationIndex(0);
                break;
            }
            case 'arrowright':
            case 'arrowdown': {
                // Next piece
                if (availablePieces.length === 0) {
                    break;
                }
                e.preventDefault();
                const currentIdx = selectedPieceId
                    ? availablePieces.findIndex(p => p.id === selectedPieceId)
                    : -1;
                const nextIdx = (currentIdx + 1) % availablePieces.length;
                setSelectedPieceId(availablePieces[nextIdx].id);
                setOrientationIndex(0);
                break;
            }
            case 'n':
                // New game
                if (!isLoading) {
                    startGame();
                }
                break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [rotatePiece, flipPiece, availablePieces, selectedPieceId, isLoading, startGame, activeTab]);

    // Render Play tab content
    const renderPlayTab = () => (
        <>
            {!gameId ? (
                <div className="space-y-6">
                    {/* Game setup */}
                    <div className="bg-surface border border-border rounded-lg p-6 max-w-md">
                        <h2 className="text-lg font-medium text-foreground mb-4">Start New Game</h2>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-secondary mb-1">Game Mode</label>
                                <Select
                                    value={gameMode}
                                    options={GAME_MODE_OPTIONS}
                                    onChange={v => setGameMode(v as GameMode)}
                                    className="w-full"
                                />
                            </div>
                            {gameMode === 'vs-ai' && (
                                <div>
                                    <label className="block text-sm font-medium text-secondary mb-1">AI Opponent</label>
                                    <Select
                                        value={selectedAIModel}
                                        options={aiModelOptions}
                                        onChange={(v) => setSelectedAIModel(String(v))}
                                        className="w-full"
                                    />
                                </div>
                            )}
                            <Button onClick={startGame} disabled={isLoading} className="w-full">
                                <Play className="w-4 h-4 mr-2" />
                                Start Game
                            </Button>
                        </div>
                    </div>

                    {/* Empty state */}
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                        <Users className="w-16 h-16 text-muted mb-4" />
                        <p className="text-foreground font-medium">No game in progress</p>
                        <p className="text-sm text-muted mt-1">Configure and start a new game above</p>
                    </div>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    {/* Board */}
                    <div className="lg:col-span-3 relative">
                        <div className="aspect-square max-w-2xl mx-auto relative">
                            {renderBoard()}
                            {renderGameOver()}
                            {/* AI Thinking Overlay */}
                            {isLoading && !gameState?.isGameOver && (
                                <div className="absolute inset-0 bg-black/40 flex items-center justify-center rounded-lg backdrop-blur-[1px]">
                                    <div className="bg-surface/90 px-6 py-4 rounded-lg flex items-center gap-3 shadow-lg">
                                        <div className="flex gap-1">
                                            <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                            <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                            <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                                        </div>
                                        <span className="text-foreground font-medium">AI is thinking...</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Sidebar */}
                    <div className="space-y-4">
                        {/* Scores */}
                        <div className="space-y-2">
                            <h3 className="text-sm font-medium text-secondary">Players</h3>
                            {gameState && (['blue', 'yellow', 'red', 'green'] as PlayerColor[]).map(color => (
                                <PlayerScore
                                    key={color}
                                    color={color}
                                    score={gameState.players[color].score}
                                    remainingPieces={gameState.players[color].remainingPieceIds.length}
                                    totalPieces={pieces.length}
                                    isCurrentTurn={gameState.currentPlayer === color}
                                    isHuman={humanColors.includes(color)}
                                    isEliminated={gameState.players[color].isEliminated}
                                    aiModelName={!humanColors.includes(color) ? selectedModelName : undefined}
                                />
                            ))}
                        </div>

                        {/* Piece selection */}
                        {renderPiecePalette()}

                        {/* Actions */}
                        {gameState && !gameState.isGameOver && (
                            <div className="space-y-2">
                                {isHumanTurn && legalMoves.length === 0 && (
                                    <Button
                                        variant="secondary"
                                        className="w-full"
                                        onClick={() => handlePass(false)}
                                        disabled={isLoading}
                                    >
                                        Pass (Auto-passing...)
                                    </Button>
                                )}
                                <Button
                                    variant="ghost"
                                    className="w-full"
                                    onClick={() => {
                                        setGameId(null);
                                        setGameState(null);
                                    }}
                                >
                                    <RotateCcw className="w-4 h-4 mr-2" />
                                    New Game
                                </Button>
                            </div>
                        )}

                        {/* Turn indicator */}
                        {gameState && !gameState.isGameOver && (
                            <div className="text-center text-sm">
                                {isLoading ? (
                                    <div className="flex items-center justify-center gap-2 text-primary">
                                        <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
                                        <span className="font-medium">AI thinking...</span>
                                    </div>
                                ) : (
                                    <span className="text-muted">
                                        Turn {gameState.turnNumber} •{' '}
                                        {isHumanTurn ? 'Your turn' : 'AI thinking...'}
                                    </span>
                                )}
                            </div>
                        )}

                        {/* Keyboard shortcuts */}
                        <div className="pt-2 border-t border-border">
                            <p className="text-xs text-muted mb-2">Keyboard Shortcuts</p>
                            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                                <div className="flex justify-between">
                                    <span className="text-secondary">R</span>
                                    <span className="text-muted">Rotate</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-secondary">F</span>
                                    <span className="text-muted">Flip</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-secondary">←→</span>
                                    <span className="text-muted">Prev/Next</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-secondary">Esc</span>
                                    <span className="text-muted">Deselect</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-secondary">N</span>
                                    <span className="text-muted">New Game</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );

    // Render Train tab content
    const renderTrainTab = () => {
        // Get running runs with their progress
        const runningRuns = alphaZeroRuns.filter(r => r.status === 'running');

        return (
            <div className="space-y-6">
                {/* Header with action */}
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-lg font-medium text-foreground">AlphaZero Training</h2>
                        <p className="text-sm text-secondary mt-1">Train AI models using self-play reinforcement learning</p>
                    </div>
                    <Button onClick={() => setShowTrainModal(true)}>
                        <Plus className="w-4 h-4 mr-2" />
                    Train New Model
                    </Button>
                </div>

                {/* Live Progress Cards for Running Runs */}
                {runningRuns.length > 0 && (
                    <div className="space-y-4">
                        {runningRuns.map(run => {
                            const progress = trainingProgress[run.id];
                            return (
                                <div key={run.id} className="bg-surface border border-border rounded-lg p-4">
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="flex items-center gap-2">
                                            <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                                            <h3 className="font-medium text-foreground">{run.name}</h3>
                                        </div>
                                        <Button size="sm" variant="ghost" onClick={() => handleStopTraining(run.id)}>
                                            <Square className="w-3 h-3 mr-1" />
                                        Stop
                                        </Button>
                                    </div>

                                    {progress ? (
                                        <div className="space-y-3">
                                            {/* Generation and Phase Steps */}
                                            <div className="flex items-center justify-between text-sm">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-secondary">Gen {progress.generation}/{progress.maxGenerations}</span>
                                                    <span className="text-muted">•</span>
                                                    <div className="flex items-center gap-1 text-xs">
                                                        {['self_play', 'training', 'evaluation'].map((phase, idx) => (
                                                            <div key={phase} className="flex items-center">
                                                                {idx > 0 && <span className="text-muted mx-1">→</span>}
                                                                <span className={
                                                                    progress.phase === phase
                                                                        ? 'text-primary font-medium'
                                                                        : ['self_play', 'training', 'evaluation'].indexOf(progress.phase) > idx
                                                                            ? 'text-green-500'
                                                                            : 'text-muted'
                                                                }>
                                                                    {PHASE_NAMES[phase]}
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                                <span className="text-foreground font-medium">
                                                    {progress.phaseCurrent}/{progress.phaseTotal}
                                                </span>
                                            </div>

                                            {/* Phase Progress Bar */}
                                            <div className="h-2 bg-surface-dim rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-primary transition-all duration-300"
                                                    style={{ width: `${progress.phaseProgress}%` }}
                                                />
                                            </div>

                                            {/* Message */}
                                            <p className="text-xs text-muted truncate">{progress.message}</p>

                                            {/* Stats Grid */}
                                            <div className="grid grid-cols-4 gap-3 text-xs">
                                                <div>
                                                    <span className="text-muted block">Phase ETA</span>
                                                    <span className="text-foreground font-medium">{formatEta(progress.phaseEtaSec)}</span>
                                                </div>
                                                <div>
                                                    <span className="text-muted block">Gen ETA</span>
                                                    <span className="text-foreground font-medium">{formatEta(progress.generationEtaSec)}</span>
                                                </div>
                                                <div>
                                                    <span className="text-muted block">Total ETA</span>
                                                    <span className="text-foreground font-medium">{formatEta(progress.overallEtaSec)}</span>
                                                </div>
                                                <div>
                                                    <span className="text-muted block">Overall</span>
                                                    <span className="text-foreground font-medium">{progress.overallProgress.toFixed(1)}%</span>
                                                </div>
                                            </div>

                                            {/* Phase-specific metrics */}
                                            {progress.phase === 'self_play' && progress.gamesPerMin && (
                                                <div className="flex items-center gap-4 text-xs text-secondary pt-2 border-t border-border">
                                                    <span>{progress.gamesPerMin.toFixed(1)} games/min</span>
                                                    <span>{progress.avgGameTimeSec?.toFixed(1)}s/game</span>
                                                    {progress.lastGameMoves && <span>{progress.lastGameMoves} moves/game</span>}
                                                </div>
                                            )}

                                            {progress.phase === 'training' && progress.policyLoss && (
                                                <div className="flex items-center gap-4 text-xs text-secondary pt-2 border-t border-border">
                                                    <span>Policy: {progress.policyLoss.toFixed(4)}</span>
                                                    <span>Value: {progress.valueLoss?.toFixed(4)}</span>
                                                    <span>{progress.stepsPerSec?.toFixed(1)} steps/s</span>
                                                </div>
                                            )}

                                            {progress.phase === 'evaluation' && progress.currentWinRate !== undefined && (
                                                <div className="flex items-center gap-4 text-xs text-secondary pt-2 border-t border-border">
                                                    <span>Win rate: {(progress.currentWinRate * 100).toFixed(1)}%</span>
                                                    <span>Wins: {progress.evalWins}/{progress.evalGames}</span>
                                                    <span>Threshold: {((progress.promotionThreshold ?? 0.55) * 100).toFixed(0)}%</span>
                                                </div>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="text-sm text-secondary">Starting training...</div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}

                {/* Training Runs table */}
                <div className="bg-surface border border-border rounded-lg overflow-hidden">
                    <div className="px-4 py-3 border-b border-border">
                        <h3 className="text-sm font-medium text-foreground">Training Runs</h3>
                    </div>
                    {isLoadingRuns ? (
                        <div className="p-8 text-center text-secondary">Loading...</div>
                    ) : alphaZeroRuns.length === 0 ? (
                        <div className="p-8 text-center">
                            <Brain className="w-12 h-12 text-muted mx-auto mb-3" />
                            <p className="text-foreground font-medium">No training runs yet</p>
                            <p className="text-sm text-secondary mt-1">Start training to create AI models</p>
                        </div>
                    ) : (
                        <Table
                            columns={aiRunColumns}
                            data={alphaZeroRuns}
                            getRowKey={(run) => run.id}
                            onRowClick={(run) => setSelectedRunForDetails(run)}
                        />
                    )}
                </div>

                {/* Run Details Modal */}
                {selectedRunForDetails && (
                    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
                        <div className="bg-surface border border-border rounded-lg p-6 max-w-lg w-full mx-4">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-medium text-foreground">{selectedRunForDetails.name}</h3>
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => setSelectedRunForDetails(null)}
                                >
                                    <X className="w-4 h-4" />
                                </Button>
                            </div>
                            <div className="space-y-3 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-secondary">Status:</span>
                                    <span className={`font-medium ${
                                        selectedRunForDetails.status === 'completed' ? 'text-green-400' :
                                            selectedRunForDetails.status === 'running' ? 'text-yellow-400' :
                                                selectedRunForDetails.status === 'failed' ? 'text-red-400' :
                                                    'text-foreground'
                                    }`}>
                                        {selectedRunForDetails.status}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-secondary">Generation:</span>
                                    <span className="text-foreground">{selectedRunForDetails.currentGeneration}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-secondary">Games Generated:</span>
                                    <span className="text-foreground">{selectedRunForDetails.gamesGenerated.toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-secondary">Training Steps:</span>
                                    <span className="text-foreground">{selectedRunForDetails.trainingSteps.toLocaleString()}</span>
                                </div>
                                {selectedRunForDetails.policyLoss !== null && (
                                    <div className="flex justify-between">
                                        <span className="text-secondary">Policy Loss:</span>
                                        <span className="text-foreground">{selectedRunForDetails.policyLoss.toFixed(4)}</span>
                                    </div>
                                )}
                                {selectedRunForDetails.valueLoss !== null && (
                                    <div className="flex justify-between">
                                        <span className="text-secondary">Value Loss:</span>
                                        <span className="text-foreground">{selectedRunForDetails.valueLoss.toFixed(4)}</span>
                                    </div>
                                )}
                                {selectedRunForDetails.bestWinRate !== null && (
                                    <div className="flex justify-between">
                                        <span className="text-secondary">Best Win Rate:</span>
                                        <span className="text-foreground">{(selectedRunForDetails.bestWinRate * 100).toFixed(0)}%</span>
                                    </div>
                                )}
                                {selectedRunForDetails.errorMessage && (
                                    <div className="p-2 bg-red-500/10 rounded text-red-400 text-xs">
                                        {selectedRunForDetails.errorMessage}
                                    </div>
                                )}
                            </div>
                            <div className="mt-4 pt-4 border-t border-border">
                                <p className="text-xs text-secondary mb-2">Config:</p>
                                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-foreground">
                                    <div><span className="text-muted">Batch:</span> {selectedRunForDetails.config.batchSize}</div>
                                    <div><span className="text-muted">MCTS Sims:</span> {selectedRunForDetails.config.mctsSimulations}</div>
                                    <div><span className="text-muted">Games/Gen:</span> {selectedRunForDetails.config.gamesPerGeneration}</div>
                                    <div><span className="text-muted">Max Gen:</span> {selectedRunForDetails.config.maxGenerations}</div>
                                </div>
                            </div>
                            <div className="flex justify-end gap-2 mt-4">
                                {(selectedRunForDetails.status === 'completed' || selectedRunForDetails.status === 'stopped') && (
                                    <Button
                                        variant="secondary"
                                        onClick={() => handleExportOnnx(selectedRunForDetails.id)}
                                        disabled={isExporting}
                                    >
                                        <Download className="w-4 h-4 mr-2" />
                                        {isExporting ? 'Exporting...' : 'Export ONNX'}
                                    </Button>
                                )}
                                {selectedRunForDetails.status === 'running' && (
                                    <Button
                                        variant="secondary"
                                        onClick={() => {
                                            handleStopTraining(selectedRunForDetails.id);
                                            setSelectedRunForDetails(null);
                                        }}
                                    >
                                        <Square className="w-4 h-4 mr-2" />
                                    Stop
                                    </Button>
                                )}
                                <Button
                                    variant="ghost"
                                    onClick={() => setSelectedRunForDetails(null)}
                                >
                                Close
                                </Button>
                            </div>
                        </div>
                    </div>
                )}

                {/* New Training Modal */}
                <Modal
                    isOpen={showTrainModal}
                    onClose={() => setShowTrainModal(false)}
                    title="Start AlphaZero Training"
                    size="lg"
                >
                    <div className="space-y-4">
                        <Input
                            label="Training Run Name"
                            value={newTrainName}
                            onChange={(e) => setNewTrainName(e.target.value)}
                            placeholder="e.g., Alpha v1"
                        />

                        <div className="border-t border-border pt-4">
                            <h4 className="text-sm font-medium text-foreground mb-3">Self-Play Configuration</h4>
                            <div className="grid grid-cols-2 gap-3">
                                <Input
                                    label="MCTS Simulations"
                                    type="number"
                                    value={trainConfig.mctsSimulations ?? 400}
                                    onChange={(e) => setTrainConfig(c => ({ ...c, mctsSimulations: parseInt(e.target.value) || 400 }))}
                                />
                                <Input
                                    label="Games per Generation"
                                    type="number"
                                    value={trainConfig.gamesPerGeneration ?? 100}
                                    onChange={(e) => setTrainConfig(c => ({ ...c, gamesPerGeneration: parseInt(e.target.value) || 100 }))}
                                />
                                <Input
                                    label="Parallel Games"
                                    type="number"
                                    value={trainConfig.numParallelGames ?? 8}
                                    onChange={(e) => setTrainConfig(c => ({ ...c, numParallelGames: parseInt(e.target.value) || 8 }))}
                                />
                                <Input
                                    label="MCTS Batch Size"
                                    type="number"
                                    value={trainConfig.mctsBatchSize ?? 16}
                                    onChange={(e) => setTrainConfig(c => ({ ...c, mctsBatchSize: parseInt(e.target.value) || 16 }))}
                                />
                                <Input
                                    label="Inference Batch Size"
                                    type="number"
                                    value={trainConfig.inferenceBatchSize ?? 32}
                                    onChange={(e) => setTrainConfig(c => ({ ...c, inferenceBatchSize: parseInt(e.target.value) || 32 }))}
                                />
                            </div>
                        </div>

                        <div className="border-t border-border pt-4">
                            <h4 className="text-sm font-medium text-foreground mb-3">Training Configuration</h4>
                            <div className="grid grid-cols-2 gap-3">
                                <Input
                                    label="Batch Size"
                                    type="number"
                                    value={trainConfig.batchSize ?? 4096}
                                    onChange={(e) => setTrainConfig(c => ({ ...c, batchSize: parseInt(e.target.value) || 4096 }))}
                                />
                                <Input
                                    label="Max Generations"
                                    type="number"
                                    value={trainConfig.maxGenerations ?? 100}
                                    onChange={(e) => setTrainConfig(c => ({ ...c, maxGenerations: parseInt(e.target.value) || 100 }))}
                                />
                            </div>
                        </div>

                        <div className="border-t border-border pt-4">
                            <h4 className="text-sm font-medium text-foreground mb-3">Network Architecture</h4>
                            <div className="grid grid-cols-2 gap-3">
                                <Input
                                    label="Residual Blocks"
                                    type="number"
                                    value={trainConfig.numResBlocks ?? 15}
                                    onChange={(e) => setTrainConfig(c => ({ ...c, numResBlocks: parseInt(e.target.value) || 15 }))}
                                />
                                <Input
                                    label="Channels"
                                    type="number"
                                    value={trainConfig.channels ?? 256}
                                    onChange={(e) => setTrainConfig(c => ({ ...c, channels: parseInt(e.target.value) || 256 }))}
                                />
                            </div>
                        </div>

                        <div className="p-3 bg-surface-dim rounded-lg text-xs text-secondary">
                            <p className="font-medium text-foreground mb-1">Quick Presets</p>
                            <div className="flex gap-2 mt-2">
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => setTrainConfig({
                                        mctsSimulations: 20,
                                        gamesPerGeneration: 5,
                                        maxGenerations: 2,
                                        batchSize: 512,
                                        numResBlocks: 5,
                                        channels: 64,
                                        evalGames: 10,
                                        numParallelGames: 4,
                                        mctsBatchSize: 8,
                                        inferenceBatchSize: 16,
                                    })}
                                >
                                Quick Test
                                </Button>
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => setTrainConfig({
                                        mctsSimulations: 200,
                                        gamesPerGeneration: 100,
                                        maxGenerations: 10,
                                        batchSize: 4096,
                                        numResBlocks: 10,
                                        channels: 128,
                                        evalGames: 20,
                                        numParallelGames: 32,
                                        mctsBatchSize: 32,
                                        inferenceBatchSize: 128,
                                        trainingStepsPerGen: 500,
                                    })}
                                >
                                Medium
                                </Button>
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => setTrainConfig({
                                        mctsSimulations: 800,
                                        gamesPerGeneration: 200,
                                        maxGenerations: 100,
                                        batchSize: 4096,
                                        numResBlocks: 15,
                                        channels: 128,
                                        evalGames: 50,
                                        numParallelGames: 32,
                                        mctsBatchSize: 32,
                                        inferenceBatchSize: 128,
                                        trainingStepsPerGen: 2000,
                                    })}
                                >
                                Large
                                </Button>
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => setTrainConfig({
                                        mctsSimulations: 1600,
                                        gamesPerGeneration: 400,
                                        maxGenerations: 200,
                                        batchSize: 8192,
                                        numResBlocks: 20,
                                        channels: 256,
                                        evalGames: 100,
                                        numParallelGames: 128,
                                        mctsBatchSize: 128,
                                        inferenceBatchSize: 512,
                                        trainingStepsPerGen: 4000,
                                    })}
                                >
                                GH200
                                </Button>
                            </div>
                        </div>
                    </div>

                    <div className="flex justify-end gap-2 mt-6 pt-4 border-t border-border">
                        <Button
                            variant="ghost"
                            onClick={() => setShowTrainModal(false)}
                        >
                        Cancel
                        </Button>
                        <Button
                            onClick={handleStartTraining}
                            disabled={!newTrainName.trim()}
                        >
                            <Brain className="w-4 h-4 mr-2" />
                        Start Training
                        </Button>
                    </div>
                </Modal>

                {/* Available AI Models (for playing) */}
                {alphaZeroCheckpoints.length > 0 && (
                    <div className="bg-surface border border-border rounded-lg overflow-hidden">
                        <div className="px-4 py-3 border-b border-border">
                            <h3 className="text-sm font-medium text-foreground">Available AI Models</h3>
                            <p className="text-xs text-secondary mt-0.5">Best performing checkpoint from each training run</p>
                        </div>
                        <div className="p-4 grid gap-2">
                            {/* Show best checkpoint per run (highest win rate) */}
                            {Array.from(
                                alphaZeroCheckpoints.reduce((map, cp) => {
                                    const existing = map.get(cp.runId);
                                    const cpWinRate = cp.evalWinRate ?? 0;
                                    const existingWinRate = existing?.evalWinRate ?? 0;
                                    if (!existing || cpWinRate > existingWinRate) {
                                        map.set(cp.runId, cp);
                                    }
                                    return map;
                                }, new Map<string, typeof alphaZeroCheckpoints[0]>())
                            ).map(([runId, cp]) => {
                                const run = alphaZeroRuns.find(r => r.id === runId);
                                return (
                                    <div
                                        key={cp.id}
                                        className="flex items-center justify-between p-3 bg-surface-dim rounded-lg hover:bg-white/5 cursor-pointer"
                                        onClick={() => {
                                            setSelectedAIModel(cp.id);
                                            setActiveTab('play');
                                            toast.success(`Selected ${run?.name || 'model'} - ready to play!`);
                                        }}
                                    >
                                        <div>
                                            <p className="text-sm font-medium text-foreground">{run?.name || 'Unknown'}</p>
                                            <p className="text-xs text-secondary">
                                            Generation {cp.generation} • {cp.evalWinRate !== null ? `${(cp.evalWinRate * 100).toFixed(0)}% win rate` : 'N/A'}
                                            </p>
                                        </div>
                                        <Button size="sm" variant="secondary">
                                            <Play className="w-3 h-3 mr-1" />
                                        Play
                                        </Button>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="space-y-6">
            {/* Header with tabs */}
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-foreground">Gymnasium</h1>
                <div className="flex bg-surface-dim rounded-lg p-1">
                    <button
                        onClick={() => setActiveTab('play')}
                        className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                            activeTab === 'play'
                                ? 'bg-surface text-foreground shadow-sm'
                                : 'text-secondary hover:text-foreground'
                        }`}
                    >
                        <Gamepad2 className="w-4 h-4" />
                        Play
                    </button>
                    <button
                        onClick={() => setActiveTab('train')}
                        className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                            activeTab === 'train'
                                ? 'bg-surface text-foreground shadow-sm'
                                : 'text-secondary hover:text-foreground'
                        }`}
                    >
                        <Brain className="w-4 h-4" />
                        Train
                    </button>
                </div>
            </div>

            {/* Tab content */}
            {activeTab === 'play' ? renderPlayTab() : renderTrainTab()}
        </div>
    );
};
