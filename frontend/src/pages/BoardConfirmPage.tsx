import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGame } from '../contexts/GameContext';
import { BoardOverlay } from '../components/BoardOverlay';
import { Button } from '../components/ui';

export function BoardConfirmPage() {
    const navigate = useNavigate();
    const { state, analyze } = useGame();

    useEffect(() => {
        if (!state.capturedImage || !state.selectedColor) {
            navigate('/');
        }
    }, [state.capturedImage, state.selectedColor, navigate]);

    const handleRetake = () => {
        navigate('/capture');
    };

    const handleAnalyze = () => {
        analyze();
        navigate('/results');
    };

    if (!state.capturedImage || !state.boardState) {
        return null;
    }

    return (
        <div className="fixed inset-0 bg-background flex flex-col">
            {/* Board image with overlay */}
            <div className="flex-1 relative">
                <img
                    src={state.capturedImage}
                    alt="Captured board"
                    className="absolute inset-0 w-full h-full object-cover"
                />
                <BoardOverlay boardState={state.boardState} />
            </div>

            {/* Action buttons */}
            <div className="p-4 bg-background border-t border-border">
                <div className="flex gap-3 max-w-md mx-auto">
                    <Button
                        variant="secondary"
                        onClick={handleRetake}
                        className="flex-1"
                    >
                        Retake
                    </Button>
                    <Button
                        variant="primary"
                        onClick={handleAnalyze}
                        className="flex-1"
                    >
                        Analyze
                    </Button>
                </div>
            </div>
        </div>
    );
}
