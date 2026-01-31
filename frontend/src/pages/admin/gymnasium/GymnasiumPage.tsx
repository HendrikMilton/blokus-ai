import { useState, useEffect, useMemo, useCallback } from 'react';
import toast from 'react-hot-toast';
import { Play, RotateCcw, FlipHorizontal2, Trophy, Users } from 'lucide-react';
import { PLAYER_COLORS } from '@blokus-ai/shared';
import type {
    GameState,
    Move,
    PlayerColor,
    GamePieceDefinition,
} from '@blokus-ai/shared';
import { Button } from '../../../components/ui/Button';
import { Select } from '../../../components/ui/Select';
import { GAME_API } from '../../../api/game';

const GAME_MODE_OPTIONS = [
    { value: 'vs-ai', label: 'vs AI (Random)' },
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
}: {
    color: PlayerColor;
    score: number;
    remainingPieces: number;
    totalPieces: number;
    isCurrentTurn: boolean;
    isHuman: boolean;
    isEliminated: boolean;
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
                {isHuman ? '(You)' : '(AI)'}
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

export const GymnasiumPage = () => {
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
    }, [gameMode, humanColors, aiColors]);

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
    }, [rotatePiece, flipPiece, availablePieces, selectedPieceId, isLoading, startGame]);

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold">Gymnasium</h1>
                {!gameId && (
                    <div className="flex items-center gap-2">
                        <Select
                            value={gameMode}
                            options={GAME_MODE_OPTIONS}
                            onChange={v => setGameMode(v as GameMode)}
                            className="w-40"
                        />
                        <Button onClick={startGame} disabled={isLoading}>
                            <Play className="w-4 h-4 mr-2" />
                            Start Game
                        </Button>
                    </div>
                )}
            </div>

            {!gameId ? (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                    <Users className="w-16 h-16 text-muted mb-4" />
                    <p className="text-foreground font-medium">No game in progress</p>
                    <p className="text-sm text-muted mt-1">Select a mode and start a new game</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    {/* Board */}
                    <div className="lg:col-span-3 relative">
                        <div className="aspect-square max-w-2xl mx-auto">
                            {renderBoard()}
                            {renderGameOver()}
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
                                    onClick={startGame}
                                >
                                    <RotateCcw className="w-4 h-4 mr-2" />
                                    New Game
                                </Button>
                            </div>
                        )}

                        {/* Turn indicator */}
                        {gameState && !gameState.isGameOver && (
                            <div className="text-center text-sm text-muted">
                                Turn {gameState.turnNumber} •{' '}
                                {isHumanTurn ? 'Your turn' : 'AI thinking...'}
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
        </div>
    );
};
