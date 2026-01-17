/**
 * SampleDetailModal - View and edit a training sample
 */

import { useState, useEffect, useCallback } from 'react';
import {
    TrainingSample,
    TrainingSampleUpdate,
    SampleStatus,
    BoardCell,
    BOARD_SIZE,
} from '@blokus-ai/shared';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { BoardEditor } from './BoardEditor';
import { ColorPicker, SelectedColor } from './ColorPicker';
import { useBoardEditor } from '../../hooks/useBoardEditor';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import { ADMIN_API } from '../../api/admin';

interface SampleDetailModalProps {
    sample: TrainingSample | null;
    isOpen: boolean;
    onClose: () => void;
    onUpdate: (id: string, update: TrainingSampleUpdate) => Promise<void>;
    onDelete: (id: string) => Promise<void>;
    onRepredict: (id: string) => Promise<void>;
    onNavigate?: (direction: 'prev' | 'next') => void;
    hasNext?: boolean;
    hasPrev?: boolean;
}

function createEmptyBoard(): BoardCell[][] {
    return Array.from({ length: BOARD_SIZE }, () =>
        Array.from({ length: BOARD_SIZE }, () => ({ color: null }))
    );
}

export function SampleDetailModal({
    sample,
    isOpen,
    onClose,
    onUpdate,
    onDelete,
    onRepredict,
    onNavigate,
    hasNext,
    hasPrev,
}: SampleDetailModalProps) {
    const [selectedColor, setSelectedColor] = useState<SelectedColor>('blue');
    const [notes, setNotes] = useState('');
    const [status, setStatus] = useState<SampleStatus>('pending');
    const [isDeleting, setIsDeleting] = useState(false);
    const [isRepredicting, setIsRepredicting] = useState(false);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [colorblindMode, setColorblindMode] = useState(false);

    // Determine initial labeled state - use labeled_state if exists, otherwise detected_state
    const initialState = sample?.labeled_state || sample?.detected_state || createEmptyBoard();

    const handleSave = useCallback(async (state: BoardCell[][]) => {
        if (!sample) {
            return;
        }
        await onUpdate(sample.id, { labeled_state: state });
    }, [sample, onUpdate]);

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
        if (sample) {
            setNotes(sample.notes || '');
            setStatus(sample.status);
            reset(sample.labeled_state || sample.detected_state);
        }
    }, [sample?.id]); // Only run when sample ID changes

    // Keyboard shortcuts
    useKeyboardShortcuts({
        onColorSelect: setSelectedColor,
        onUndo: undo,
        onRedo: redo,
        onNextSample: hasNext ? () => onNavigate?.('next') : undefined,
        onPrevSample: hasPrev ? () => onNavigate?.('prev') : undefined,
        enabled: isOpen,
    });

    const handleStatusChange = async (newStatus: SampleStatus) => {
        if (!sample) {
            return;
        }
        setStatus(newStatus);
        await onUpdate(sample.id, { status: newStatus });
    };

    const handleNotesChange = async () => {
        if (!sample || notes === sample.notes) {
            return;
        }
        await onUpdate(sample.id, { notes });
    };

    const handleDelete = async () => {
        if (!sample) {
            return;
        }
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
        if (!sample) {
            return;
        }
        setIsRepredicting(true);
        try {
            await onRepredict(sample.id);
        } finally {
            setIsRepredicting(false);
        }
    };

    const copyDetectedToLabeled = () => {
        if (!sample) {
            return;
        }
        setBoardState(sample.detected_state);
    };

    const clearLabels = () => {
        setBoardState(createEmptyBoard());
    };

    if (!sample) {
        return null;
    }

    const imageUrl = ADMIN_API.getSampleImageUrl(sample.id);
    const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="xl" title={`Sample ${sample.id.slice(0, 8)}`}>
            <div className="flex flex-col gap-4 max-h-[80vh] overflow-y-auto">
                {/* Mobile warning */}
                {isMobile && (
                    <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-md">
                        <p className="text-sm text-yellow-400">
                            For labeling, please use this page on desktop. Full editing capabilities require keyboard shortcuts and precise mouse control.
                        </p>
                    </div>
                )}

                {/* Top bar with actions */}
                <div className="flex flex-wrap items-center gap-2">
                    {/* Status selector */}
                    <select
                        value={status}
                        onChange={(e) => handleStatusChange(e.target.value as SampleStatus)}
                        className="px-2 py-1.5 text-sm bg-surface border border-border rounded-md"
                    >
                        <option value="pending">Pending</option>
                        <option value="labeled">Labeled</option>
                        <option value="verified">Verified</option>
                    </select>

                    {/* Colorblind mode toggle */}
                    <button
                        onClick={() => setColorblindMode(!colorblindMode)}
                        className={`px-2 py-1.5 text-sm rounded-md border ${
                            colorblindMode
                                ? 'bg-primary/20 border-primary text-primary'
                                : 'bg-surface border-border text-muted-foreground'
                        }`}
                        title="Toggle colorblind-friendly patterns"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                    </button>

                    <div className="flex-1" />

                    {/* Navigation */}
                    {onNavigate && (
                        <div className="flex gap-1">
                            <button
                                onClick={() => onNavigate('prev')}
                                disabled={!hasPrev}
                                className="p-1.5 rounded-md bg-surface border border-border disabled:opacity-50"
                                title="Previous sample (←)"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                </svg>
                            </button>
                            <button
                                onClick={() => onNavigate('next')}
                                disabled={!hasNext}
                                className="p-1.5 rounded-md bg-surface border border-border disabled:opacity-50"
                                title="Next sample (→)"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                        </div>
                    )}

                    {/* Save indicator */}
                    <div className="flex items-center gap-1 text-xs">
                        {isSaving ? (
                            <span className="text-yellow-400 flex items-center gap-1">
                                <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                                Saving...
                            </span>
                        ) : isDirty ? (
                            <span className="text-muted-foreground flex items-center gap-1">
                                <span className="w-2 h-2 rounded-full bg-muted-foreground" />
                                Unsaved
                            </span>
                        ) : (
                            <span className="text-green-400 flex items-center gap-1">
                                <span className="w-2 h-2 rounded-full bg-green-400" />
                                Saved
                            </span>
                        )}
                    </div>
                </div>

                {/* Main content - side by side on desktop */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {/* Original image */}
                    <div className="space-y-2">
                        <h4 className="text-sm font-medium text-muted-foreground">Original Image</h4>
                        <img
                            src={imageUrl}
                            alt="Original board"
                            className="w-full aspect-square object-cover rounded-lg border border-border"
                        />
                    </div>

                    {/* Detected state (read-only) */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <h4 className="text-sm font-medium text-muted-foreground">Detected State</h4>
                            <button
                                onClick={handleRepredict}
                                disabled={isRepredicting}
                                className="text-xs text-primary hover:text-primary/80"
                            >
                                {isRepredicting ? 'Detecting...' : 'Re-detect'}
                            </button>
                        </div>
                        <BoardEditor
                            boardState={sample.detected_state}
                            selectedColor="clear"
                            onChange={() => {}}
                            readOnly
                            colorblindMode={colorblindMode}
                            size={300}
                        />
                        <button
                            onClick={copyDetectedToLabeled}
                            className="w-full text-xs text-muted-foreground hover:text-foreground py-1"
                        >
                            Copy to labeled →
                        </button>
                    </div>

                    {/* Labeled state (editable) */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <h4 className="text-sm font-medium text-muted-foreground">Labeled State</h4>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={undo}
                                    disabled={!canUndo}
                                    className="p-1 rounded text-muted-foreground hover:text-foreground disabled:opacity-50"
                                    title="Undo (Ctrl+Z)"
                                >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                                    </svg>
                                </button>
                                <button
                                    onClick={redo}
                                    disabled={!canRedo}
                                    className="p-1 rounded text-muted-foreground hover:text-foreground disabled:opacity-50"
                                    title="Redo (Ctrl+Shift+Z)"
                                >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6" />
                                    </svg>
                                </button>
                                <button
                                    onClick={clearLabels}
                                    className="text-xs text-red-400 hover:text-red-300"
                                >
                                    Clear
                                </button>
                            </div>
                        </div>
                        <BoardEditor
                            boardState={boardState}
                            selectedColor={selectedColor}
                            onChange={setBoardState}
                            colorblindMode={colorblindMode}
                            size={300}
                        />
                    </div>
                </div>

                {/* Color picker */}
                <div className="space-y-2">
                    <h4 className="text-sm font-medium text-muted-foreground">Color Palette</h4>
                    <ColorPicker
                        selectedColor={selectedColor}
                        onColorSelect={setSelectedColor}
                        showShortcuts={!isMobile}
                        colorblindMode={colorblindMode}
                    />
                </div>

                {/* Notes */}
                <div className="space-y-2">
                    <h4 className="text-sm font-medium text-muted-foreground">Notes</h4>
                    <textarea
                        value={notes}
                        onChange={(e) => setNotes(e.target.value)}
                        onBlur={handleNotesChange}
                        placeholder="Add notes about this sample..."
                        className="w-full px-3 py-2 text-sm bg-surface border border-border rounded-md resize-none"
                        rows={2}
                    />
                </div>

                {/* Footer actions */}
                <div className="flex justify-between pt-4 border-t border-border">
                    <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => setShowDeleteConfirm(true)}
                    >
                        Delete Sample
                    </Button>

                    <Button variant="secondary" size="sm" onClick={onClose}>
                        Close
                    </Button>
                </div>
            </div>

            {/* Delete confirmation */}
            {showDeleteConfirm && (
                <div className="fixed inset-0 z-50 flex items-center justify-center">
                    <div
                        className="absolute inset-0 bg-black/50"
                        onClick={() => setShowDeleteConfirm(false)}
                    />
                    <div className="relative bg-background p-6 rounded-lg max-w-sm mx-4">
                        <h3 className="text-lg font-medium mb-2">Delete Sample?</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                            This action cannot be undone. The image and all labels will be permanently deleted.
                        </p>
                        <div className="flex justify-end gap-2">
                            <Button
                                variant="secondary"
                                size="sm"
                                onClick={() => setShowDeleteConfirm(false)}
                            >
                                Cancel
                            </Button>
                            <Button
                                variant="destructive"
                                size="sm"
                                onClick={handleDelete}
                                loading={isDeleting}
                            >
                                Delete
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </Modal>
    );
}
