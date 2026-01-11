import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { useGame } from '../contexts/GameContext';
import { BoardOverlay } from '../components/BoardOverlay';
import { Button, Loading } from '../components/ui';
import { PLAYER_COLORS } from '../types/game';

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

    const handleAnalyze = async () => {
        toast.success(`boardState: ${!!state.boardState}, isDetecting: ${state.isDetecting}`);
        try {
            toast.success('Starting analyze...');
            await analyze();
            toast.success('Analyze complete!');
            navigate('/results');
        } catch (error) {
            toast.error(`Analyze failed: ${error}`);
        }
    };

    if (!state.capturedImage || !state.selectedColor) {
        return null;
    }

    // Count detected pieces per color
    const pieceCount = state.boardState?.flat().filter(cell => cell.color).length ?? 0;

    return (
        <div className="fixed inset-0 bg-background flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-4 bg-background/80 backdrop-blur-sm border-b border-border">
                <button
                    onClick={handleRetake}
                    className="p-2 -ml-2 text-muted-foreground hover:text-foreground transition-colors"
                >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                </button>
                <h1 className="text-lg font-semibold text-foreground">Confirm Board</h1>
                <div
                    className="px-3 py-1 rounded-full text-sm font-medium text-white"
                    style={{ backgroundColor: PLAYER_COLORS[state.selectedColor] }}
                >
                    {state.selectedColor}
                </div>
            </div>

            {/* Board image with overlay */}
            <div className="flex-1 relative overflow-hidden">
                {state.isDetecting ? (
                    <div className="absolute inset-0 flex items-center justify-center bg-surface">
                        <div className="text-center">
                            <Loading />
                            <p className="mt-4 text-muted-foreground">Detecting board...</p>
                        </div>
                    </div>
                ) : (
                    <>
                        <img
                            src={state.capturedImage}
                            alt="Captured board"
                            className="absolute inset-0 w-full h-full object-cover"
                        />
                        {/* Semi-transparent backdrop for better grid visibility */}
                        <div className="absolute inset-0 bg-black/20" />
                        {state.boardState && <BoardOverlay boardState={state.boardState} />}
                    </>
                )}
            </div>

            {/* Detection summary */}
            {!state.isDetecting && state.boardState && (
                <div className="px-4 py-3 bg-surface border-t border-border">
                    <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>Detected {pieceCount} placed cells</span>
                    </div>
                </div>
            )}

            {/* Action buttons */}
            <div className="p-4 bg-background border-t border-border">
                <div className="flex gap-3 max-w-md mx-auto">
                    <Button
                        variant="secondary"
                        onClick={handleRetake}
                        className="flex-1"
                        disabled={state.isDetecting}
                    >
                        Retake
                    </Button>
                    <Button
                        variant="primary"
                        onClick={handleAnalyze}
                        className="flex-1"
                        disabled={state.isDetecting || !state.boardState}
                    >
                        {state.isDetecting ? 'Detecting...' : state.isAnalyzing ? 'Analyzing...' : !state.boardState ? 'No Board Detected' : 'Analyze'}
                    </Button>
                </div>
            </div>
        </div>
    );
}
