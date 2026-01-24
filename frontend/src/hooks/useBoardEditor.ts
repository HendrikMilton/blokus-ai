/**
 * useBoardEditor - Hook for managing board state with undo/redo and auto-save
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { BoardCell, BOARD_SIZE } from '@blokus-ai/shared';

interface UseBoardEditorOptions {
    initialState: BoardCell[][];
    onSave?: (state: BoardCell[][]) => void | Promise<void>;
    autoSaveDelay?: number;
    maxHistorySize?: number;
}

interface UseBoardEditorReturn {
    boardState: BoardCell[][];
    setBoardState: (state: BoardCell[][]) => void;
    undo: () => void;
    redo: () => void;
    canUndo: boolean;
    canRedo: boolean;
    isDirty: boolean;
    isSaving: boolean;
    saveNow: () => Promise<void>;
    reset: (newState?: BoardCell[][]) => void;
}

const createEmptyBoard = (): BoardCell[][] => Array.from({ length: BOARD_SIZE }, () =>
    Array.from({ length: BOARD_SIZE }, () => ({ color: null }))
);

const cloneBoard = (board: BoardCell[][]): BoardCell[][] => board.map(row => row.map(cell => ({ ...cell })));

const boardsEqual = (a: BoardCell[][], b: BoardCell[][]): boolean => {
    for (let i = 0; i < BOARD_SIZE; i++) {
        for (let j = 0; j < BOARD_SIZE; j++) {
            if (a[i][j].color !== b[i][j].color) {
                return false;
            }
        }
    }
    return true;
};

export const useBoardEditor = ({
    initialState,
    onSave,
    autoSaveDelay = 1000,
    maxHistorySize = 50,
}: UseBoardEditorOptions): UseBoardEditorReturn => {
    const [boardState, setBoardStateInternal] = useState<BoardCell[][]>(
        () => initialState || createEmptyBoard()
    );
    const [history, setHistory] = useState<BoardCell[][][]>([]);
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [isDirty, setIsDirty] = useState(false);
    const [isSaving, setIsSaving] = useState(false);

    const savedStateRef = useRef<BoardCell[][]>(initialState || createEmptyBoard());
    const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // Auto-save effect
    useEffect(() => {
        if (!isDirty || !onSave) {
            return;
        }

        if (saveTimeoutRef.current) {
            clearTimeout(saveTimeoutRef.current);
        }

        saveTimeoutRef.current = setTimeout(async () => {
            setIsSaving(true);
            try {
                await onSave(boardState);
                savedStateRef.current = cloneBoard(boardState);
                setIsDirty(false);
            } catch (error) {
                console.error('Auto-save failed:', error);
            } finally {
                setIsSaving(false);
            }
        }, autoSaveDelay);

        return () => {
            if (saveTimeoutRef.current) {
                clearTimeout(saveTimeoutRef.current);
            }
        };
    }, [boardState, isDirty, onSave, autoSaveDelay]);

    const setBoardState = useCallback((newState: BoardCell[][]) => {
        setBoardStateInternal(prevState => {
            // Only add to history if state actually changed
            if (boardsEqual(prevState, newState)) {
                return prevState;
            }

            // Add previous state to history
            setHistory(prev => {
                const newHistory = prev.slice(0, historyIndex + 1);
                newHistory.push(cloneBoard(prevState));
                // Trim history if too large
                if (newHistory.length > maxHistorySize) {
                    return newHistory.slice(-maxHistorySize);
                }
                return newHistory;
            });
            setHistoryIndex(prev => Math.min(prev + 1, maxHistorySize - 1));
            setIsDirty(true);

            return newState;
        });
    }, [historyIndex, maxHistorySize]);

    const undo = useCallback(() => {
        if (historyIndex < 0) {
            return;
        }

        const previousState = history[historyIndex];
        if (previousState) {
            // Save current state for redo
            setHistory(prev => {
                const newHistory = [...prev];
                newHistory[historyIndex + 1] = cloneBoard(boardState);
                return newHistory;
            });

            setBoardStateInternal(previousState);
            setHistoryIndex(prev => prev - 1);
            setIsDirty(!boardsEqual(previousState, savedStateRef.current));
        }
    }, [history, historyIndex, boardState]);

    const redo = useCallback(() => {
        const nextState = history[historyIndex + 2];
        if (nextState) {
            setBoardStateInternal(nextState);
            setHistoryIndex(prev => prev + 1);
            setIsDirty(!boardsEqual(nextState, savedStateRef.current));
        }
    }, [history, historyIndex]);

    const saveNow = useCallback(async () => {
        if (!onSave || !isDirty) {
            return;
        }

        if (saveTimeoutRef.current) {
            clearTimeout(saveTimeoutRef.current);
            saveTimeoutRef.current = null;
        }

        setIsSaving(true);
        try {
            await onSave(boardState);
            savedStateRef.current = cloneBoard(boardState);
            setIsDirty(false);
        } finally {
            setIsSaving(false);
        }
    }, [boardState, isDirty, onSave]);

    const reset = useCallback((newState?: BoardCell[][]) => {
        const state = newState || createEmptyBoard();
        setBoardStateInternal(state);
        savedStateRef.current = cloneBoard(state);
        setHistory([]);
        setHistoryIndex(-1);
        setIsDirty(false);
    }, []);

    // Update when initialState changes
    useEffect(() => {
        if (initialState && !boardsEqual(initialState, savedStateRef.current)) {
            reset(initialState);
        }
    }, [initialState, reset]);

    const canUndo = historyIndex >= 0;
    const canRedo = historyIndex < history.length - 1;

    return {
        boardState,
        setBoardState,
        undo,
        redo,
        canUndo,
        canRedo,
        isDirty,
        isSaving,
        saveNow,
        reset,
    };
};
