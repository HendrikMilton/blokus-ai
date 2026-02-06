import { useState, useEffect, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, ArrowLeft, User, Bot, Crosshair } from 'lucide-react';
import { PLAYER_COLORS, BOARD_SIZE } from '@blokus-ai/shared';
import type { SavedGame, PlayerColor, GamePieceDefinition, CellState } from '@blokus-ai/shared';
import { Button } from '../../../components/ui/Button';
import { Table, Column } from '../../../components/ui/Table';
import { GAME_API } from '../../../api/game';
import { GAMES_API } from '../../../api/games';

// Replay board cell - simplified (no hover/click/hints)
const ReplayCell = ({
    color,
    isCurrentMove,
}: {
    color: CellState;
    isCurrentMove: boolean;
}) => (
    <div
        className={`w-full h-full ${isCurrentMove ? 'ring-2 ring-white ring-inset' : ''}`}
        style={{
            backgroundColor: color ? PLAYER_COLORS[color] : '#1a1a2e',
        }}
    />
);

// Move row type for the table
interface MoveRow {
    index: number;
    player: PlayerColor;
    pieceId: string;
    position: string;
}

// Format AI model name for display
const formatAiModel = (modelId: string): string => {
    if (modelId === 'heuristic') return 'Easy';
    if (modelId.startsWith('heuristic:')) {
        const level = modelId.split(':')[1];
        return level.charAt(0).toUpperCase() + level.slice(1);
    }
    if (modelId === 'random') return 'Random';
    // AlphaZero checkpoint - truncate
    return modelId.length > 12 ? `AZ:${modelId.slice(0, 8)}` : modelId;
};

// Compute placements from scores (handles ties)
const computePlacements = (scores: Record<PlayerColor, number>): Record<PlayerColor, number> => {
    const entries = (['blue', 'yellow', 'red', 'green'] as PlayerColor[])
        .map(c => ({ color: c, score: scores[c] }));
    entries.sort((a, b) => b.score - a.score);

    const placements: Record<string, number> = {};
    let rank = 1;
    for (let i = 0; i < entries.length; i++) {
        if (i > 0 && entries[i].score < entries[i - 1].score) {
            rank = i + 1;
        }
        placements[entries[i].color] = rank;
    }
    return placements as Record<PlayerColor, number>;
};

const PLACEMENT_LABELS: Record<number, string> = {
    1: '1st',
    2: '2nd',
    3: '3rd',
    4: '4th',
};

export const GameReplayPage = () => {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();

    const [game, setGame] = useState<SavedGame | null>(null);
    const [pieces, setPieces] = useState<GamePieceDefinition[]>([]);
    const [currentMoveIndex, setCurrentMoveIndex] = useState(-1); // -1 = empty board
    const [loading, setLoading] = useState(true);

    // Load game and pieces
    useEffect(() => {
        if (!id) {
            return;
        }
        setLoading(true);
        Promise.all([
            GAMES_API.getGame(id),
            GAME_API.getPieces(),
        ]).then(([gameData, piecesData]) => {
            setGame(gameData);
            setPieces(piecesData);
            setCurrentMoveIndex(gameData.moves.length - 1); // Start at final state
        }).catch(() => {
            navigate('/admin/gymnasium');
        }).finally(() => setLoading(false));
    }, [id, navigate]);

    // Build piece lookup map
    const pieceMap = useMemo(() => {
        const map = new Map<string, GamePieceDefinition>();
        for (const p of pieces) {
            map.set(p.id, p);
        }
        return map;
    }, [pieces]);

    // Reconstruct board state at currentMoveIndex
    const { board, currentMoveCells, scores } = useMemo(() => {
        const b: CellState[][] = Array.from({ length: BOARD_SIZE }, () =>
            Array.from({ length: BOARD_SIZE }, () => null)
        );
        const sc: Record<PlayerColor, number> = { blue: 0, yellow: 0, red: 0, green: 0 };
        const moveCells = new Set<string>();

        if (!game) {
            return { board: b, currentMoveCells: moveCells, scores: sc };
        }

        for (let i = 0; i <= currentMoveIndex && i < game.moves.length; i++) {
            const move = game.moves[i];
            const piece = pieceMap.get(move.pieceId);
            if (!piece) {
                continue;
            }

            const orientation = piece.orientations[move.orientationIndex];
            if (!orientation) {
                continue;
            }

            const cells: string[] = [];
            for (const [dr, dc] of orientation) {
                const row = move.position.row + dr;
                const col = move.position.col + dc;
                if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
                    b[row][col] = move.player;
                    cells.push(`${row},${col}`);
                }
            }
            sc[move.player] += piece.size;

            // Only highlight cells of the current (last applied) move
            if (i === currentMoveIndex) {
                for (const c of cells) {
                    moveCells.add(c);
                }
            }
        }

        return { board: b, currentMoveCells: moveCells, scores: sc };
    }, [game, currentMoveIndex, pieceMap]);

    // Final placements (based on final scores, not current position)
    const finalPlacements = useMemo(() => {
        if (!game) return { blue: 0, yellow: 0, red: 0, green: 0 } as Record<PlayerColor, number>;
        return computePlacements(game.scores);
    }, [game]);

    // Navigation
    const goFirst = useCallback(() => setCurrentMoveIndex(-1), []);
    const goPrev = useCallback(() => setCurrentMoveIndex(i => Math.max(-1, i - 1)), []);
    const goNext = useCallback(() => {
        if (!game) {
            return;
        }
        setCurrentMoveIndex(i => Math.min(game.moves.length - 1, i + 1));
    }, [game]);
    const goLast = useCallback(() => {
        if (!game) {
            return;
        }
        setCurrentMoveIndex(game.moves.length - 1);
    }, [game]);

    // Keyboard navigation
    useEffect(() => {
        const handleKey = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
                return;
            }
            switch (e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                goPrev();
                break;
            case 'ArrowRight':
                e.preventDefault();
                goNext();
                break;
            case 'Home':
                e.preventDefault();
                goFirst();
                break;
            case 'End':
                e.preventDefault();
                goLast();
                break;
            }
        };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, [goFirst, goPrev, goNext, goLast]);

    // Move history table columns
    const moveColumns: Column<MoveRow>[] = useMemo(() => [
        {
            key: 'index',
            header: '#',
            width: '40px',
            render: (m) => <span className="text-xs text-muted">{m.index + 1}</span>,
        },
        {
            key: 'player',
            header: 'Player',
            width: '60px',
            render: (m) => (
                <div className="flex items-center gap-1.5">
                    <div
                        className="w-2.5 h-2.5 rounded-sm"
                        style={{ backgroundColor: PLAYER_COLORS[m.player] }}
                    />
                </div>
            ),
        },
        {
            key: 'pieceId',
            header: 'Piece',
            render: (m) => {
                const piece = pieceMap.get(m.pieceId);
                return <span className="text-xs">{piece?.name || m.pieceId}</span>;
            },
        },
        {
            key: 'position',
            header: 'Position',
            width: '60px',
            render: (m) => <span className="text-xs text-secondary">{m.position}</span>,
        },
    ], [pieceMap]);

    // Move rows for table
    const moveRows: MoveRow[] = useMemo(() => {
        if (!game) {
            return [];
        }
        return game.moves.map((m, i) => ({
            index: i,
            player: m.player,
            pieceId: m.pieceId,
            position: `${m.position.row},${m.position.col}`,
        }));
    }, [game]);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <div className="text-secondary">Loading game...</div>
            </div>
        );
    }

    if (!game) {
        return null;
    }

    const totalMoves = game.moves.length;
    const humanSet = new Set(game.humanColors);
    const isAtFinalState = currentMoveIndex === totalMoves - 1;

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-3">
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => navigate('/admin/gymnasium')}
                >
                    <ArrowLeft className="w-4 h-4" />
                </Button>
                <div>
                    <h1 className="text-lg font-bold text-foreground">
                        Game Replay
                    </h1>
                    <p className="text-xs text-secondary">
                        {game.mode} &middot; {new Date(game.createdAt).toLocaleString()} &middot; {totalMoves} moves
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Board */}
                <div className="lg:col-span-3">
                    <div className="aspect-square max-w-2xl mx-auto">
                        <div
                            className="grid bg-surface rounded-lg p-2 gap-0.5 h-full"
                            style={{
                                gridTemplateColumns: `repeat(${BOARD_SIZE}, 1fr)`,
                                gridTemplateRows: `repeat(${BOARD_SIZE}, 1fr)`,
                            }}
                        >
                            {board.map((row, ri) =>
                                row.map((cell, ci) => (
                                    <ReplayCell
                                        key={`${ri}-${ci}`}
                                        color={cell}
                                        isCurrentMove={currentMoveCells.has(`${ri},${ci}`)}
                                    />
                                ))
                            )}
                        </div>
                    </div>
                </div>

                {/* Sidebar */}
                <div className="space-y-4">
                    {/* Navigation */}
                    <div className="bg-surface border border-border rounded-lg p-3 space-y-3">
                        <div className="text-center text-sm font-medium text-foreground">
                            Move {currentMoveIndex + 1} / {totalMoves}
                        </div>
                        <div className="flex items-center justify-center gap-1">
                            <Button size="sm" variant="ghost" onClick={goFirst} disabled={currentMoveIndex <= -1} title="First (Home)">
                                <ChevronsLeft className="w-4 h-4" />
                            </Button>
                            <Button size="sm" variant="ghost" onClick={goPrev} disabled={currentMoveIndex <= -1} title="Previous (Left)">
                                <ChevronLeft className="w-4 h-4" />
                            </Button>
                            <Button size="sm" variant="ghost" onClick={goNext} disabled={currentMoveIndex >= totalMoves - 1} title="Next (Right)">
                                <ChevronRight className="w-4 h-4" />
                            </Button>
                            <Button size="sm" variant="ghost" onClick={goLast} disabled={currentMoveIndex >= totalMoves - 1} title="Last (End)">
                                <ChevronsRight className="w-4 h-4" />
                            </Button>
                        </div>
                    </div>

                    {/* Game Info */}
                    <div className="bg-surface border border-border rounded-lg p-3 space-y-2">
                        <h3 className="text-xs font-medium text-secondary uppercase tracking-wide">Game Info</h3>
                        <div className="flex flex-wrap gap-1.5">
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-500/15 text-blue-400">
                                {game.mode.toUpperCase()}
                            </span>
                            {game.aiTargetsHuman && (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-red-500/15 text-red-400">
                                    <Crosshair className="w-3 h-3" />
                                    Targeting
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Scores & Players */}
                    <div className="space-y-1.5">
                        <h3 className="text-xs font-medium text-secondary uppercase tracking-wide">Players & Scores</h3>
                        {(['blue', 'yellow', 'red', 'green'] as PlayerColor[]).map(color => {
                            const isHuman = humanSet.has(color);
                            const aiModel = game.aiModels?.[color];
                            const placement = finalPlacements[color];
                            const isWinner = game.winner === color;

                            return (
                                <div
                                    key={color}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded ${
                                        isWinner ? 'bg-yellow-500/10 ring-1 ring-yellow-500/30' : 'bg-surface-dim'
                                    }`}
                                >
                                    <div
                                        className="w-3 h-3 rounded-sm flex-shrink-0"
                                        style={{ backgroundColor: PLAYER_COLORS[color] }}
                                    />
                                    <div className="flex flex-col flex-1 min-w-0">
                                        <div className="flex items-center gap-1.5">
                                            <span className="text-sm capitalize text-foreground">{color}</span>
                                            {isHuman ? (
                                                <span className="inline-flex items-center gap-0.5 px-1.5 py-0 rounded text-[10px] font-medium bg-emerald-500/15 text-emerald-400">
                                                    <User className="w-2.5 h-2.5" />
                                                    You
                                                </span>
                                            ) : (
                                                <span className="inline-flex items-center gap-0.5 px-1.5 py-0 rounded text-[10px] font-medium bg-purple-500/15 text-purple-400">
                                                    <Bot className="w-2.5 h-2.5" />
                                                    {aiModel ? formatAiModel(aiModel) : 'Random'}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    {isAtFinalState && (
                                        <span className={`text-[10px] font-medium px-1.5 py-0 rounded ${
                                            placement === 1 ? 'bg-yellow-500/20 text-yellow-400' :
                                                placement === 2 ? 'bg-gray-400/15 text-gray-400' :
                                                    placement === 3 ? 'bg-amber-700/20 text-amber-600' :
                                                        'bg-surface text-muted'
                                        }`}>
                                            {PLACEMENT_LABELS[placement]}
                                        </span>
                                    )}
                                    <span className="text-sm font-medium text-foreground tabular-nums">{scores[color]}</span>
                                </div>
                            );
                        })}
                    </div>

                    {/* Move history */}
                    <div className="bg-surface border border-border rounded-lg overflow-hidden">
                        <div className="px-3 py-2 border-b border-border">
                            <h3 className="text-xs font-medium text-secondary">Move History</h3>
                        </div>
                        <div className="max-h-64 overflow-y-auto">
                            <Table
                                columns={moveColumns}
                                data={moveRows}
                                getRowKey={(m) => m.index}
                                onRowClick={(m) => setCurrentMoveIndex(m.index)}
                                getRowClassName={(m) =>
                                    m.index === currentMoveIndex ? 'bg-white/10' : ''
                                }
                            />
                        </div>
                    </div>

                    {/* Keyboard shortcuts */}
                    <div className="pt-2 border-t border-border">
                        <p className="text-xs text-muted mb-1">Keyboard</p>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
                            <div className="flex justify-between">
                                <span className="text-secondary">Left/Right</span>
                                <span className="text-muted">Prev/Next</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-secondary">Home/End</span>
                                <span className="text-muted">Start/End</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
