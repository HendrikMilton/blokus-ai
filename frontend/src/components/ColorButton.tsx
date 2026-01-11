import { PlayerColor, PLAYER_COLORS } from '../types/game';
import { PieceIcon } from './PieceIcon';

interface ColorButtonProps {
    color: PlayerColor;
    onClick: () => void;
}

const COLOR_NAMES: Record<PlayerColor, string> = {
    blue: 'Blue',
    yellow: 'Yellow',
    red: 'Red',
    green: 'Green',
};

export function ColorButton({ color, onClick }: ColorButtonProps) {
    return (
        <button
            onClick={onClick}
            className="relative flex flex-col items-center justify-center rounded-2xl p-6 transition-all duration-200 hover:scale-105 active:scale-95 min-h-[140px]"
            style={{ backgroundColor: PLAYER_COLORS[color] }}
        >
            <PieceIcon className="w-16 h-16 text-white mb-2" />
            <span className="text-white font-semibold text-lg">
                {COLOR_NAMES[color]}
            </span>
        </button>
    );
}
