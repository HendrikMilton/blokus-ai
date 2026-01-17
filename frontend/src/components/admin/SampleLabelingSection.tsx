/**
 * SampleLabelingSection - Full-width inline labeling interface
 */

import { useState, useEffect, useCallback } from 'react';
import {
    TrainingSample,
    TrainingSampleUpdate,
    SampleStatus,
    BoardCell,
    BOARD_SIZE,
} from '@blokus-ai/shared';
import { Button } from '../ui/Button';
import { BoardEditor } from './BoardEditor';
import { ColorPicker, SelectedColor } from './ColorPicker';
import { useBoardEditor } from '../../hooks/useBoardEditor';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import { ADMIN_API } from '../../api/admin';

interface SampleLabelingSectionProps {
    sample: TrainingSample;
    onUpdate: (id: string, update: TrainingSampleUpdate) => Promise<void>;
    onDelete: (id: string) => Promise<void>;
    onRepredict: (id: string) => Promise<void>;
    onClose: () => void;
    onNavigate?: (direction: 'prev' | 'next') => void;
    hasNext?: boolean;
    hasPrev?: boolean;
}

function createEmptyBoard(): BoardCell[][] {
    return Array.from({ length: BOARD_SIZE }, () =>
        Array.from({ length: BOARD_SIZE }, () => ({ color: null }))
    );
}

export function SampleLabelingSection({
    sample,
    onUpdate,
    onDelete,
    onRepredict,
    onClose,
    onNavigate,
    hasNext,
    hasPrev,
}: SampleLabelingSectionProps) {
    const [selectedColor, setSelectedColor] = useState<SelectedColor>('blue');
    const [notes, setNotes] = useState('');
    const [status, setStatus] = useState<SampleStatus>('pending');
    const [isDeleting, setIsDeleting] = useState(false);
    const [isRepredicting, setIsRepredicting] = useState(false);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [colorblindMode, setColorblindMode] = useState(false);

    // Determine initial labeled state
    const initialState = sample.labeled_state || sample.detected_state || createEmptyBoard();

    const handleSave = useCallback(async (state: BoardCell[][]) => {
        await onUpdate(sample.id, { labeled_state: state });
    }, [sample.id, onUpdate]);

    const {
        boardState,
        setBoardState,
        undo,
        redo,
        canUndo,
        canRedo,
        isDirty,
        isSaving,
        reset,
    } = useBoardEditor({
        initialState,
        onSave: handleSave,
        autoSaveDelay: 1000,
    });

    // Reset state when sample changes
    useEffect(() => {
        setNotes(sample.notes || '');
        setStatus(sample.status);
        reset(sample.labeled_state || sample.detected_state);
    }, [sample.id]);

    // Keyboard shortcuts
    useKeyboardShortcuts({
        onColorSelect: setSelectedColor,
        onUndo: undo,
        onRedo: redo,
        onNextSample: hasNext ? () => onNavigate?.('next') : undefined,
        onPrevSample: hasPrev ? () => onNavigate?.('prev') : undefined,
        onClose,
        enabled: true,
    });

    const handleStatusChange = async (newStatus: SampleStatus) => {
        setStatus(newStatus);
        await onUpdate(sample.id, { status: newStatus });
    };

    const handleNotesChange = async () => {
        if (notes === sample.notes) return;
        await onUpdate(sample.id, { notes });
    };

    const handleDelete = async () => {
        setIsDeleting(true);
        try {
            await onDelete(sample.id);
            setShowDeleteConfirm(false);
            onClose();
        } finally {
            setIsDeleting(false);
        }
    };

    const handleRepredict = async () => {
        setIsRepredicting(true);
        try {
            await onRepredict(sample.id);
        } finally {
            setIsRepredicting(false);
        }
    };

    const copyDetectedToLabeled = () => {
        setBoardState(sample.detected_state);
    };

    const clearLabels = () => {
        setBoardState(createEmptyBoard());
    };

    const imageUrl = ADMIN_API.getSampleImageUrl(sample.id);

    return (
        <div className="bg-surface border border-border rounded-lg overflow-hidden">
            {/* Header bar */}
            <div className="flex items-center justify-between px-4 py-3 bg-muted/30 border-b border-border">
                <div className="flex items-center gap-4">
                    <h3 className="font-medium text-foreground">
                        Sample {sample.id.slice(0, 8)}
                    </h3>

                    {/* Status selector */}
                    <select
                        value={status}
                        onChange={(e) => handleStatusChange(e.target.value as SampleStatus)}
                        className="px-2 py-1 text-sm bg-background border border-border rounded-md"
                    >
                        <option value="pending">Pending</option>
                        <option value="labeled">Labeled</option>
                        <option value="verified">Verified</option>
                    </select>

                    {/* Save indicator */}
                    <div className="flex items-center gap-1.5 text-sm">
                        {isSaving ? (
                            <span className="text-yellow-400 flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                                Saving...
                            </span>
                        ) : isDirty ? (
                            <span className="text-muted-foreground flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-muted-foreground" />
                                Unsaved
                            </span>
                        ) : (
                            <span className="text-green-400 flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-green-400" />
                                Saved
                            </span>
                        )}
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    {/* Navigation */}
                    {onNavigate && (
                        <div className="flex gap-1 mr-2">
                            <button
                                onClick={() => onNavigate('prev')}
                                disabled={!hasPrev}
                                className="p-1.5 rounded-md bg-background border border-border hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
                                title="Previous sample (←)"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                </svg>
                            </button>
                            <button
                                onClick={() => onNavigate('next')}
                                disabled={!hasNext}
                                className="p-1.5 rounded-md bg-background border border-border hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
                                title="Next sample (→)"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                        </div>
                    )}

                    {/* Colorblind toggle */}
                    <button
                        onClick={() => setColorblindMode(!colorblindMode)}
                        className={`p-1.5 rounded-md border ${
                            colorblindMode
                                ? 'bg-primary/20 border-primary text-primary'
                                : 'bg-background border-border text-muted-foreground hover:text-foreground'
                        }`}
                        title="Toggle colorblind mode"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                    </button>

                    {/* Close button */}
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded-md bg-background border border-border hover:bg-muted"
                        title="Close (Esc)"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>
            </div>

            {/* Main content */}
            <div className="p-6">
                <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
                    {/* Original image */}
                    <div className="space-y-3">
                        <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                            Original Image
                        </h4>
                        <div className="aspect-square bg-black/20 rounded-lg overflow-hidden border border-border">
                            <img
                                src={imageUrl}
                                alt="Original board"
                                className="w-full h-full object-contain"
                            />
                        </div>
                    </div>

                    {/* Detected state */}
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                                Detected State
                            </h4>
                            <button
                                onClick={handleRepredict}
                                disabled={isRepredicting}
                                className="text-xs text-primary hover:text-primary/80 disabled:opacity-50"
                            >
                                {isRepredicting ? 'Detecting...' : 'Re-detect'}
                            </button>
                        </div>
                        <div className="aspect-square">
                            <BoardEditor
                                boardState={sample.detected_state}
                                selectedColor="clear"
                                onChange={() => {}}
                                readOnly
                                colorblindMode={colorblindMode}
                            />
                        </div>
                        <button
                            onClick={copyDetectedToLabeled}
                            className="w-full text-sm text-muted-foreground hover:text-foreground py-2 border border-dashed border-border rounded-md hover:border-primary/50 transition-colors"
                        >
                            Copy to labeled →
                        </button>
                    </div>

                    {/* Labeled state (editable) */}
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                                Labeled State
                            </h4>
                            <div className="flex items-center gap-1">
                                <button
                                    onClick={undo}
                                    disabled={!canUndo}
                                    className="p-1 rounded text-muted-foreground hover:text-foreground disabled:opacity-30"
                                    title="Undo (Ctrl+Z)"
                                >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                                    </svg>
                                </button>
                                <button
                                    onClick={redo}
                                    disabled={!canRedo}
                                    className="p-1 rounded text-muted-foreground hover:text-foreground disabled:opacity-30"
                                    title="Redo (Ctrl+Shift+Z)"
                                >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6" />
                                    </svg>
                                </button>
                                <button
                                    onClick={clearLabels}
                                    className="text-xs text-red-400 hover:text-red-300 ml-2"
                                >
                                    Clear
                                </button>
                            </div>
                        </div>
                        <div className="aspect-square">
                            <BoardEditor
                                boardState={boardState}
                                selectedColor={selectedColor}
                                onChange={setBoardState}
                                colorblindMode={colorblindMode}
                            />
                        </div>
                    </div>

                    {/* Right sidebar - color picker, notes, actions */}
                    <div className="space-y-6">
                        {/* Color picker */}
                        <div className="space-y-3">
                            <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                                Color Palette
                            </h4>
                            <ColorPicker
                                selectedColor={selectedColor}
                                onColorSelect={setSelectedColor}
                                showShortcuts
                                colorblindMode={colorblindMode}
                            />
                        </div>

                        {/* Notes */}
                        <div className="space-y-3">
                            <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                                Notes
                            </h4>
                            <textarea
                                value={notes}
                                onChange={(e) => setNotes(e.target.value)}
                                onBlur={handleNotesChange}
                                placeholder="Add notes about this sample..."
                                className="w-full px-3 py-2 text-sm bg-background border border-border rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-primary/50"
                                rows={3}
                            />
                        </div>

                        {/* Keyboard shortcuts help */}
                        <div className="space-y-2 text-xs text-muted-foreground">
                            <h4 className="font-medium uppercase tracking-wide">Shortcuts</h4>
                            <div className="grid grid-cols-2 gap-1">
                                <span>1-4</span><span>Select color</span>
                                <span>0</span><span>Eraser</span>
                                <span>Ctrl+Z</span><span>Undo</span>
                                <span>Ctrl+Shift+Z</span><span>Redo</span>
                                <span>← →</span><span>Navigate samples</span>
                                <span>Esc</span><span>Close</span>
                            </div>
                        </div>

                        {/* Delete action */}
                        <div className="pt-4 border-t border-border">
                            {showDeleteConfirm ? (
                                <div className="space-y-3">
                                    <p className="text-sm text-muted-foreground">
                                        Delete this sample? This cannot be undone.
                                    </p>
                                    <div className="flex gap-2">
                                        <Button
                                            variant="destructive"
                                            size="sm"
                                            onClick={handleDelete}
                                            loading={isDeleting}
                                            className="flex-1"
                                        >
                                            Delete
                                        </Button>
                                        <Button
                                            variant="secondary"
                                            size="sm"
                                            onClick={() => setShowDeleteConfirm(false)}
                                            className="flex-1"
                                        >
                                            Cancel
                                        </Button>
                                    </div>
                                </div>
                            ) : (
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => setShowDeleteConfirm(true)}
                                    className="text-red-400 hover:text-red-300 hover:bg-red-400/10"
                                >
                                    Delete Sample
                                </Button>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
