import type { SampleStatusFlags, GridLabels, PlayerColor, BoardDetection } from '@blokus-ai/shared';
import { PLAYER_COLORS } from '@blokus-ai/shared';

// Grid cell colors for labeling
export const CELL_COLORS: { key: PlayerColor | null; label: string; color: string }[] = [
    { key: null, label: 'Empty', color: '#e5e7eb' },      // gray-200
    { key: 'blue', label: 'Blue', color: PLAYER_COLORS.blue },
    { key: 'yellow', label: 'Yellow', color: PLAYER_COLORS.yellow },
    { key: 'red', label: 'Red', color: PLAYER_COLORS.red },
    { key: 'green', label: 'Green', color: PLAYER_COLORS.green },
];

// Create empty 20x20 grid
export const createEmptyGrid = (): GridLabels =>
    Array.from({ length: 20 }, () => Array.from({ length: 20 }, () => null));

// Corner labels and colors
export const CORNER_CONFIG = [
    { key: 'topLeft' as const, label: 'TL', color: '#ef4444' },     // red
    { key: 'topRight' as const, label: 'TR', color: '#22c55e' },    // green
    { key: 'bottomRight' as const, label: 'BR', color: '#3b82f6' }, // blue
    { key: 'bottomLeft' as const, label: 'BL', color: '#eab308' },  // yellow
];

export type CornerKey = 'topLeft' | 'topRight' | 'bottomRight' | 'bottomLeft';
export type PartialCorners = Partial<Record<CornerKey, { x: number; y: number }>>;

export interface Sample {
    id: string;
    filename: string;
    status: SampleStatusFlags;
    created_at: string;
    imageUrl?: string;
    thumbnailUrl?: string;
}

export interface SamplesResponse {
    samples: Sample[];
}

export interface LabelData {
    prediction: BoardDetection | null;
    correction: BoardDetection | null;
    isModified: boolean;
}

export type DragHandle = 'nw' | 'ne' | 'sw' | 'se' | 'move' | null;
