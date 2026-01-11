import { useState } from 'react';
import { BoardCell, BOARD_SIZE, PLAYER_COLORS, PlayerColor } from '../types/game';

interface SavedBoard {
    id: string;
    timestamp: Date;
    image: string;
    detectedState: BoardCell[][];
    correctedState?: BoardCell[][];
    notes?: string;
}

// Mock data for now - will be replaced with API calls
const MOCK_BOARDS: SavedBoard[] = [];

function BoardThumbnail({ board, onClick }: { board: SavedBoard; onClick: () => void }) {
    return (
        <button
            onClick={onClick}
            className="bg-surface rounded-lg border border-border overflow-hidden hover:border-primary transition-colors"
        >
            <div className="aspect-square relative">
                <img
                    src={board.image}
                    alt={`Board ${board.id}`}
                    className="w-full h-full object-cover"
                />
                <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-2 py-1">
                    <p className="text-xs text-white truncate">
                        {board.timestamp.toLocaleString()}
                    </p>
                </div>
            </div>
        </button>
    );
}

function MiniBoard({ state, size = 200 }: { state: BoardCell[][]; size?: number }) {
    const cellSize = size / BOARD_SIZE;

    return (
        <svg width={size} height={size} className="border border-border rounded">
            {/* Grid background */}
            <rect width={size} height={size} fill="#1a1a2e" />

            {/* Cells */}
            {state.map((row, r) =>
                row.map((cell, c) => (
                    <rect
                        key={`${r}-${c}`}
                        x={c * cellSize}
                        y={r * cellSize}
                        width={cellSize - 0.5}
                        height={cellSize - 0.5}
                        fill={cell.color ? PLAYER_COLORS[cell.color] : 'transparent'}
                        stroke="#333"
                        strokeWidth={0.25}
                    />
                ))
            )}
        </svg>
    );
}

function BoardDetailModal({
    board,
    onClose,
    onSave,
}: {
    board: SavedBoard;
    onClose: () => void;
    onSave: (updated: SavedBoard) => void;
}) {
    const [editingState, setEditingState] = useState<BoardCell[][]>(
        board.correctedState || board.detectedState
    );
    const [selectedColor, setSelectedColor] = useState<PlayerColor | null>(null);
    const [notes, setNotes] = useState(board.notes || '');

    const handleCellClick = (row: number, col: number) => {
        if (selectedColor === null) return;

        const newState = editingState.map((r, ri) =>
            r.map((cell, ci) => {
                if (ri === row && ci === col) {
                    // Toggle: if same color, clear it; otherwise set new color
                    const newColor = cell.color === selectedColor ? null : selectedColor;
                    return { color: newColor };
                }
                return cell;
            })
        );
        setEditingState(newState);
    };

    const handleSave = () => {
        onSave({
            ...board,
            correctedState: editingState,
            notes,
        });
        onClose();
    };

    const cellSize = 280 / BOARD_SIZE;

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
            <div className="bg-background rounded-xl max-w-2xl w-full max-h-[90vh] overflow-auto">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-border">
                    <h2 className="text-lg font-semibold text-foreground">Board Details</h2>
                    <button
                        onClick={onClose}
                        className="text-muted-foreground hover:text-foreground"
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                <div className="p-4 space-y-4">
                    {/* Original image */}
                    <div>
                        <h3 className="text-sm font-medium text-muted-foreground mb-2">Original Photo</h3>
                        <img
                            src={board.image}
                            alt="Original"
                            className="w-full max-w-sm rounded-lg"
                        />
                    </div>

                    {/* Side by side comparison */}
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <h3 className="text-sm font-medium text-muted-foreground mb-2">Detected</h3>
                            <MiniBoard state={board.detectedState} size={160} />
                        </div>
                        <div>
                            <h3 className="text-sm font-medium text-muted-foreground mb-2">
                                Corrected {editingState !== board.detectedState && '(edited)'}
                            </h3>
                            <MiniBoard state={editingState} size={160} />
                        </div>
                    </div>

                    {/* Editor */}
                    <div>
                        <h3 className="text-sm font-medium text-muted-foreground mb-2">Edit Labels</h3>

                        {/* Color picker */}
                        <div className="flex gap-2 mb-3">
                            {(Object.keys(PLAYER_COLORS) as PlayerColor[]).map((color) => (
                                <button
                                    key={color}
                                    onClick={() => setSelectedColor(color)}
                                    className={`w-8 h-8 rounded-full border-2 transition-all ${
                                        selectedColor === color ? 'border-white scale-110' : 'border-transparent'
                                    }`}
                                    style={{ backgroundColor: PLAYER_COLORS[color] }}
                                />
                            ))}
                            <button
                                onClick={() => setSelectedColor(null)}
                                className={`w-8 h-8 rounded-full border-2 bg-gray-700 flex items-center justify-center ${
                                    selectedColor === null ? 'border-white scale-110' : 'border-transparent'
                                }`}
                            >
                                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        {/* Editable grid */}
                        <div className="overflow-auto">
                            <svg
                                width={cellSize * BOARD_SIZE}
                                height={cellSize * BOARD_SIZE}
                                className="border border-border rounded cursor-pointer"
                            >
                                <rect width={cellSize * BOARD_SIZE} height={cellSize * BOARD_SIZE} fill="#1a1a2e" />
                                {editingState.map((row, r) =>
                                    row.map((cell, c) => (
                                        <rect
                                            key={`${r}-${c}`}
                                            x={c * cellSize}
                                            y={r * cellSize}
                                            width={cellSize - 0.5}
                                            height={cellSize - 0.5}
                                            fill={cell.color ? PLAYER_COLORS[cell.color] : 'transparent'}
                                            stroke="#444"
                                            strokeWidth={0.5}
                                            onClick={() => handleCellClick(r, c)}
                                            className="hover:opacity-80 cursor-pointer"
                                        />
                                    ))
                                )}
                            </svg>
                        </div>
                    </div>

                    {/* Notes */}
                    <div>
                        <h3 className="text-sm font-medium text-muted-foreground mb-2">Notes</h3>
                        <textarea
                            value={notes}
                            onChange={(e) => setNotes(e.target.value)}
                            placeholder="Add notes about detection issues..."
                            className="w-full bg-surface border border-border rounded-lg p-3 text-foreground placeholder:text-muted-foreground resize-none"
                            rows={3}
                        />
                    </div>

                    {/* Actions */}
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="flex-1 px-4 py-2 rounded-lg border border-border text-foreground hover:bg-surface transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            className="flex-1 px-4 py-2 rounded-lg bg-primary text-white hover:bg-primary/90 transition-colors"
                        >
                            Save Corrections
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export function MLAdminPage() {
    const [boards, setBoards] = useState<SavedBoard[]>(MOCK_BOARDS);
    const [selectedBoard, setSelectedBoard] = useState<SavedBoard | null>(null);

    const handleSaveBoard = (updated: SavedBoard) => {
        setBoards((prev) =>
            prev.map((b) => (b.id === updated.id ? updated : b))
        );
    };

    return (
        <div className="min-h-screen bg-background">
            {/* Header */}
            <div className="bg-surface border-b border-border px-4 py-4">
                <h1 className="text-xl font-bold text-foreground">ML Admin</h1>
                <p className="text-sm text-muted-foreground mt-1">
                    Review and correct board detection labels
                </p>
            </div>

            {/* Stats */}
            <div className="p-4 grid grid-cols-3 gap-3">
                <div className="bg-surface rounded-lg p-3 border border-border">
                    <p className="text-2xl font-bold text-foreground">{boards.length}</p>
                    <p className="text-xs text-muted-foreground">Total Boards</p>
                </div>
                <div className="bg-surface rounded-lg p-3 border border-border">
                    <p className="text-2xl font-bold text-green-500">
                        {boards.filter((b) => b.correctedState).length}
                    </p>
                    <p className="text-xs text-muted-foreground">Labeled</p>
                </div>
                <div className="bg-surface rounded-lg p-3 border border-border">
                    <p className="text-2xl font-bold text-yellow-500">
                        {boards.filter((b) => !b.correctedState).length}
                    </p>
                    <p className="text-xs text-muted-foreground">Pending</p>
                </div>
            </div>

            {/* Board grid */}
            <div className="p-4">
                {boards.length === 0 ? (
                    <div className="text-center py-12">
                        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-surface flex items-center justify-center">
                            <svg className="w-8 h-8 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                        </div>
                        <p className="text-muted-foreground">No boards captured yet</p>
                        <p className="text-sm text-muted-foreground/60 mt-1">
                            Take photos of your Blokus board to start labeling
                        </p>
                    </div>
                ) : (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                        {boards.map((board) => (
                            <BoardThumbnail
                                key={board.id}
                                board={board}
                                onClick={() => setSelectedBoard(board)}
                            />
                        ))}
                    </div>
                )}
            </div>

            {/* Detail modal */}
            {selectedBoard && (
                <BoardDetailModal
                    board={selectedBoard}
                    onClose={() => setSelectedBoard(null)}
                    onSave={handleSaveBoard}
                />
            )}
        </div>
    );
}
