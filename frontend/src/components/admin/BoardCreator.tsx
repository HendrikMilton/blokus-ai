/**
 * BoardCreator - Create board states from scratch for synthetic training data
 */

import { useState, useCallback } from 'react';
import { BoardCell, BOARD_SIZE } from '@blokus-ai/shared';
import { BoardEditor } from './BoardEditor';
import { ColorPicker, SelectedColor } from './ColorPicker';
import { useBoardEditor } from '../../hooks/useBoardEditor';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import { Button } from '../ui/Button';
import { Modal } from '../ui/Modal';

interface BoardCreatorProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (boardState: BoardCell[][]) => void | Promise<void>;
}

function createEmptyBoard(): BoardCell[][] {
    return Array.from({ length: BOARD_SIZE }, () =>
        Array.from({ length: BOARD_SIZE }, () => ({ color: null }))
    );
}

export function BoardCreator({ isOpen, onClose, onSave }: BoardCreatorProps) {
    const [selectedColor, setSelectedColor] = useState<SelectedColor>('blue');
    const [isSaving, setIsSaving] = useState(false);
    const [colorblindMode, setColorblindMode] = useState(false);

    const {
        boardState,
        setBoardState,
        undo,
        redo,
        canUndo,
        canRedo,
        reset,
    } = useBoardEditor({
        initialState: createEmptyBoard(),
    });

    useKeyboardShortcuts({
        onColorSelect: setSelectedColor,
        onUndo: undo,
        onRedo: redo,
        enabled: isOpen,
    });

    const handleSave = useCallback(async () => {
        setIsSaving(true);
        try {
            await onSave(boardState);
            onClose();
        } finally {
            setIsSaving(false);
        }
    }, [boardState, onSave, onClose]);

    const handleExportJSON = useCallback(() => {
        const json = JSON.stringify(boardState, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `board-state-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, [boardState]);

    const countCells = () => {
        const counts = { blue: 0, yellow: 0, red: 0, green: 0, empty: 0 };
        for (const row of boardState) {
            for (const cell of row) {
                if (cell.color) {
                    counts[cell.color]++;
                } else {
                    counts.empty++;
                }
            }
        }
        return counts;
    };

    const cellCounts = countCells();
    const totalColored = cellCounts.blue + cellCounts.yellow + cellCounts.red + cellCounts.green;

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="lg" title="Create Board State">
            <div className="flex flex-col gap-4">
                {/* Info */}
                <p className="text-sm text-muted-foreground">
                    Create a board state from scratch. This is useful for generating synthetic training data
                    to test edge cases or specific patterns.
                </p>

                {/* Toolbar */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <button
                            onClick={undo}
                            disabled={!canUndo}
                            className="p-2 rounded-md bg-surface border border-border disabled:opacity-50"
                            title="Undo (Ctrl+Z)"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                            </svg>
                        </button>
                        <button
                            onClick={redo}
                            disabled={!canRedo}
                            className="p-2 rounded-md bg-surface border border-border disabled:opacity-50"
                            title="Redo (Ctrl+Shift+Z)"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6" />
                            </svg>
                        </button>
                        <button
                            onClick={() => reset()}
                            className="px-3 py-2 text-sm rounded-md bg-surface border border-border text-muted-foreground hover:text-foreground"
                        >
                            Clear All
                        </button>
                    </div>

                    <button
                        onClick={() => setColorblindMode(!colorblindMode)}
                        className={`p-2 rounded-md border ${
                            colorblindMode
                                ? 'bg-primary/20 border-primary text-primary'
                                : 'bg-surface border-border text-muted-foreground'
                        }`}
                        title="Toggle colorblind patterns"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                    </button>
                </div>

                {/* Board editor */}
                <div className="flex justify-center">
                    <BoardEditor
                        boardState={boardState}
                        selectedColor={selectedColor}
                        onChange={setBoardState}
                        colorblindMode={colorblindMode}
                        size={400}
                    />
                </div>

                {/* Color picker */}
                <ColorPicker
                    selectedColor={selectedColor}
                    onColorSelect={setSelectedColor}
                    showShortcuts
                    colorblindMode={colorblindMode}
                />

                {/* Cell counts */}
                <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
                    <span>Total colored: {totalColored}</span>
                    <span className="text-blue-400">Blue: {cellCounts.blue}</span>
                    <span className="text-yellow-400">Yellow: {cellCounts.yellow}</span>
                    <span className="text-red-400">Red: {cellCounts.red}</span>
                    <span className="text-green-400">Green: {cellCounts.green}</span>
                </div>

                {/* Actions */}
                <div className="flex justify-between pt-4 border-t border-border">
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleExportJSON}
                    >
                        Export JSON
                    </Button>

                    <div className="flex gap-2">
                        <Button variant="secondary" size="sm" onClick={onClose}>
                            Cancel
                        </Button>
                        <Button
                            size="sm"
                            onClick={handleSave}
                            loading={isSaving}
                            disabled={totalColored === 0}
                        >
                            Save as Training Sample
                        </Button>
                    </div>
                </div>
            </div>
        </Modal>
    );
}
