/**
 * ColorPicker - Color selection buttons with keyboard shortcut hints
 */

import { PlayerColor, PLAYER_COLORS } from '@blokus-ai/shared';

export type SelectedColor = PlayerColor | 'clear';

interface ColorPickerProps {
    selectedColor: SelectedColor;
    onColorSelect: (color: SelectedColor) => void;
    showShortcuts?: boolean;
    colorblindMode?: boolean;
}

const COLOR_PATTERNS: Record<PlayerColor, string> = {
    blue: 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,255,255,0.3) 2px, rgba(255,255,255,0.3) 4px)',
    yellow: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.15) 2px, rgba(0,0,0,0.15) 4px)',
    red: 'repeating-linear-gradient(-45deg, transparent, transparent 2px, rgba(255,255,255,0.3) 2px, rgba(255,255,255,0.3) 4px)',
    green: 'radial-gradient(circle, rgba(255,255,255,0.3) 1px, transparent 1px)',
};

const SHORTCUT_KEYS: Record<SelectedColor, string> = {
    blue: '1',
    yellow: '2',
    red: '3',
    green: '4',
    clear: '0',
};

export function ColorPicker({
    selectedColor,
    onColorSelect,
    showShortcuts = true,
    colorblindMode = false,
}: ColorPickerProps) {
    const colors: SelectedColor[] = ['blue', 'yellow', 'red', 'green', 'clear'];

    return (
        <div className="flex flex-wrap gap-2">
            {colors.map((color) => {
                const isSelected = selectedColor === color;
                const isClear = color === 'clear';
                const bgColor = isClear ? '#374151' : PLAYER_COLORS[color as PlayerColor];
                const pattern = !isClear && colorblindMode ? COLOR_PATTERNS[color as PlayerColor] : undefined;

                return (
                    <button
                        key={color}
                        onClick={() => onColorSelect(color)}
                        className={`
                            relative flex items-center justify-center gap-2 px-3 py-2 rounded-lg
                            transition-all duration-150 min-w-[80px]
                            ${isSelected
                        ? 'ring-2 ring-white ring-offset-2 ring-offset-background scale-105'
                        : 'hover:scale-105 opacity-80 hover:opacity-100'
                    }
                        `}
                        style={{
                            backgroundColor: bgColor,
                            backgroundImage: pattern,
                            backgroundSize: color === 'green' && colorblindMode ? '6px 6px' : undefined,
                        }}
                        title={`${color} (${SHORTCUT_KEYS[color]})`}
                    >
                        <span className={`
                            text-sm font-medium capitalize
                            ${isClear || color === 'yellow' ? 'text-white drop-shadow-md' : 'text-white'}
                        `}>
                            {isClear ? 'Clear' : color}
                        </span>
                        {showShortcuts && (
                            <span className={`
                                text-xs px-1.5 py-0.5 rounded
                                ${isClear || color === 'yellow'
                                ? 'bg-black/30 text-white'
                                : 'bg-white/20 text-white'
                            }
                            `}>
                                {SHORTCUT_KEYS[color]}
                            </span>
                        )}
                    </button>
                );
            })}
        </div>
    );
}
