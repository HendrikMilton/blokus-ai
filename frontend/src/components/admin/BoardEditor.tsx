/**
 * BoardEditor - Interactive 20x20 grid for labeling board states
 */

import { useCallback, useRef, useState, useEffect } from 'react';
import { BoardCell, PlayerColor, PLAYER_COLORS, BOARD_SIZE } from '@blokus-ai/shared';
import { SelectedColor } from './ColorPicker';

interface BoardEditorProps {
    boardState: BoardCell[][];
    selectedColor: SelectedColor;
    onChange: (newState: BoardCell[][]) => void;
    readOnly?: boolean;
    colorblindMode?: boolean;
}

const COLOR_PATTERNS: Record<PlayerColor, { pattern: string; size?: string }> = {
    blue: {
        pattern: 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,255,255,0.4) 2px, rgba(255,255,255,0.4) 4px)',
    },
    yellow: {
        pattern: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.2) 2px, rgba(0,0,0,0.2) 4px)',
    },
    red: {
        pattern: 'repeating-linear-gradient(-45deg, transparent, transparent 2px, rgba(255,255,255,0.4) 2px, rgba(255,255,255,0.4) 4px)',
    },
    green: {
        pattern: 'radial-gradient(circle, rgba(255,255,255,0.4) 1.5px, transparent 1.5px)',
        size: '6px 6px',
    },
};

export function BoardEditor({
    boardState,
    selectedColor,
    onChange,
    readOnly = false,
    colorblindMode = false,
}: BoardEditorProps) {
    const [isDragging, setIsDragging] = useState(false);
    const lastCellRef = useRef<{ row: number; col: number } | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [containerSize, setContainerSize] = useState(400);

    // Measure container size
    useEffect(() => {
        const updateSize = () => {
            if (containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect();
                setContainerSize(Math.min(rect.width, rect.height));
            }
        };
        updateSize();
        window.addEventListener('resize', updateSize);
        return () => window.removeEventListener('resize', updateSize);
    }, []);

    const cellSize = containerSize / BOARD_SIZE;

    const paintCell = useCallback((row: number, col: number) => {
        if (readOnly) {
            return;
        }

        // Skip if same cell as last painted
        if (lastCellRef.current?.row === row && lastCellRef.current?.col === col) {
            return;
        }
        lastCellRef.current = { row, col };

        const newState = boardState.map((r, ri) =>
            r.map((cell, ci) => {
                if (ri === row && ci === col) {
                    return {
                        color: selectedColor === 'clear' ? null : selectedColor,
                    };
                }
                return cell;
            })
        );

        onChange(newState);
    }, [boardState, selectedColor, onChange, readOnly]);

    const getCellPosition = useCallback((e: React.MouseEvent | React.TouchEvent) => {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        let clientX: number, clientY: number;

        if ('touches' in e) {
            if (e.touches.length === 0) {
                return null;
            }
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        const x = clientX - rect.left;
        const y = clientY - rect.top;

        const col = Math.floor((x / rect.width) * BOARD_SIZE);
        const row = Math.floor((y / rect.height) * BOARD_SIZE);

        if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
            return { row, col };
        }
        return null;
    }, []);

    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        if (readOnly) {
            return;
        }
        e.preventDefault();
        setIsDragging(true);
        lastCellRef.current = null;

        const pos = getCellPosition(e);
        if (pos) {
            paintCell(pos.row, pos.col);
        }
    }, [readOnly, getCellPosition, paintCell]);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (!isDragging || readOnly) {
            return;
        }

        const pos = getCellPosition(e);
        if (pos) {
            paintCell(pos.row, pos.col);
        }
    }, [isDragging, readOnly, getCellPosition, paintCell]);

    const handleMouseUp = useCallback(() => {
        setIsDragging(false);
        lastCellRef.current = null;
    }, []);

    const handleTouchStart = useCallback((e: React.TouchEvent) => {
        if (readOnly) {
            return;
        }
        setIsDragging(true);
        lastCellRef.current = null;

        const pos = getCellPosition(e);
        if (pos) {
            paintCell(pos.row, pos.col);
        }
    }, [readOnly, getCellPosition, paintCell]);

    const handleTouchMove = useCallback((e: React.TouchEvent) => {
        if (!isDragging || readOnly) {
            return;
        }

        const pos = getCellPosition(e);
        if (pos) {
            paintCell(pos.row, pos.col);
        }
    }, [isDragging, readOnly, getCellPosition, paintCell]);

    const handleTouchEnd = useCallback(() => {
        setIsDragging(false);
        lastCellRef.current = null;
    }, []);

    return (
        <div
            ref={containerRef}
            className={`relative select-none touch-none w-full h-full ${readOnly ? '' : 'cursor-crosshair'}`}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
        >
            {/* Grid background */}
            <svg
                width="100%"
                height="100%"
                viewBox={`0 0 ${BOARD_SIZE} ${BOARD_SIZE}`}
                preserveAspectRatio="xMidYMid meet"
                className="absolute inset-0"
            >
                {/* Background */}
                <rect width={BOARD_SIZE} height={BOARD_SIZE} fill="#1a1a2e" />

                {/* Grid lines */}
                {Array.from({ length: BOARD_SIZE + 1 }).map((_, i) => (
                    <g key={i}>
                        <line
                            x1={i}
                            y1={0}
                            x2={i}
                            y2={BOARD_SIZE}
                            stroke="#333"
                            strokeWidth={0.02}
                        />
                        <line
                            x1={0}
                            y1={i}
                            x2={BOARD_SIZE}
                            y2={i}
                            stroke="#333"
                            strokeWidth={0.02}
                        />
                    </g>
                ))}

                {/* Cells */}
                {boardState.map((row, rowIdx) =>
                    row.map((cell, colIdx) => {
                        if (!cell.color) {
                            return null;
                        }

                        return (
                            <rect
                                key={`${rowIdx}-${colIdx}`}
                                x={colIdx + 0.02}
                                y={rowIdx + 0.02}
                                width={0.96}
                                height={0.96}
                                fill={PLAYER_COLORS[cell.color]}
                                rx={0.05}
                            />
                        );
                    })
                )}
            </svg>

            {/* Pattern overlay for colorblind mode */}
            {colorblindMode && (
                <div className="absolute inset-0 pointer-events-none">
                    {boardState.map((row, rowIdx) =>
                        row.map((cell, colIdx) => {
                            if (!cell.color) {
                                return null;
                            }

                            const pattern = COLOR_PATTERNS[cell.color];
                            return (
                                <div
                                    key={`pattern-${rowIdx}-${colIdx}`}
                                    className="absolute"
                                    style={{
                                        left: colIdx * cellSize,
                                        top: rowIdx * cellSize,
                                        width: cellSize,
                                        height: cellSize,
                                        backgroundImage: pattern.pattern,
                                        backgroundSize: pattern.size,
                                    }}
                                />
                            );
                        })
                    )}
                </div>
            )}
        </div>
    );
}
