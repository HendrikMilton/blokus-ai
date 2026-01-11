import { useNavigate } from 'react-router-dom';
import { ColorButton } from '../components/ColorButton';
import { useGame } from '../contexts/GameContext';
import { PlayerColor } from '../types/game';

const COLORS: PlayerColor[] = ['blue', 'yellow', 'red', 'green'];

export function ColorSelectionPage() {
    const navigate = useNavigate();
    const { selectColor } = useGame();

    const handleColorSelect = (color: PlayerColor) => {
        selectColor(color);
        navigate('/capture');
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-[calc(100vh-80px)] p-4">
            <p className="text-muted-foreground mb-8 text-center">
                Select your player color to begin
            </p>

            <div className="grid grid-cols-2 gap-4 w-full max-w-sm">
                {COLORS.map(color => (
                    <ColorButton
                        key={color}
                        color={color}
                        onClick={() => handleColorSelect(color)}
                    />
                ))}
            </div>
        </div>
    );
}
