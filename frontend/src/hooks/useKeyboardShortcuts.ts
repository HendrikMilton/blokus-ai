/**
 * useKeyboardShortcuts - Hook for handling keyboard shortcuts
 */

import { useEffect, useCallback } from 'react';
import { SelectedColor } from '../components/admin/ColorPicker';

interface KeyboardShortcutsConfig {
    onColorSelect?: (color: SelectedColor) => void;
    onUndo?: () => void;
    onRedo?: () => void;
    onSave?: () => void;
    onNextSample?: () => void;
    onPrevSample?: () => void;
    onDelete?: () => void;
    onClose?: () => void;
    enabled?: boolean;
}

export const useKeyboardShortcuts = ({
    onColorSelect,
    onUndo,
    onRedo,
    onSave,
    onNextSample,
    onPrevSample,
    onDelete,
    onClose,
    enabled = true,
}: KeyboardShortcutsConfig) => {
    const handleKeyDown = useCallback((e: KeyboardEvent) => {
        // Ignore if typing in an input or textarea
        if (
            e.target instanceof HTMLInputElement ||
            e.target instanceof HTMLTextAreaElement
        ) {
            return;
        }

        // Ctrl/Cmd + Z = Undo
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            onUndo?.();
            return;
        }

        // Ctrl/Cmd + Shift + Z = Redo
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
            e.preventDefault();
            onRedo?.();
            return;
        }

        // Ctrl/Cmd + Y = Redo (alternative)
        if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
            e.preventDefault();
            onRedo?.();
            return;
        }

        // Ctrl/Cmd + S = Save
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            onSave?.();
            return;
        }

        // Arrow keys for navigation
        if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
            if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                e.preventDefault();
                onPrevSample?.();
                return;
            }
        }

        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
            if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                e.preventDefault();
                onNextSample?.();
                return;
            }
        }

        // Escape key to close
        if (e.key === 'Escape') {
            e.preventDefault();
            onClose?.();
            return;
        }

        // Delete key
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                // Only trigger delete if we're not in an editable element
                onDelete?.();
                return;
            }
        }

        // Number keys for color selection
        const colorMap: Record<string, SelectedColor> = {
            1: 'blue',
            2: 'yellow',
            3: 'red',
            4: 'green',
            0: 'clear',
        };

        if (e.key in colorMap && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.preventDefault();
            onColorSelect?.(colorMap[e.key]);
        }
    }, [onColorSelect, onUndo, onRedo, onSave, onNextSample, onPrevSample, onDelete, onClose]);

    useEffect(() => {
        if (!enabled) {
            return;
        }

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [enabled, handleKeyDown]);
};

// Type-only export for component props
export type { KeyboardShortcutsConfig };
