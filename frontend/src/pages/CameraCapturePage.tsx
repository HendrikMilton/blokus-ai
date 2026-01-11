import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGame } from '../contexts/GameContext';
import { useCamera } from '../hooks/useCamera';
import { PLAYER_COLORS } from '../types/game';

export function CameraCapturePage() {
    const navigate = useNavigate();
    const { state, captureBoard } = useGame();
    const { videoRef, canvasRef, isReady, error, startCamera, capture } = useCamera();

    useEffect(() => {
        if (!state.selectedColor) {
            navigate('/');
            return;
        }
        startCamera();
    }, [state.selectedColor, navigate, startCamera]);

    const handleCapture = () => {
        const imageData = capture();
        if (imageData) {
            captureBoard(imageData);
            navigate('/confirm');
        }
    };

    const handleBack = () => {
        navigate('/');
    };

    if (!state.selectedColor) {
        return null;
    }

    return (
        <div className="fixed inset-0 bg-black flex flex-col">
            {/* Back button and color indicator */}
            <div className="absolute top-4 left-4 right-4 z-10 flex justify-between items-center">
                <button
                    onClick={handleBack}
                    className="w-10 h-10 rounded-full bg-black/50 flex items-center justify-center text-white"
                >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                </button>

                <div
                    className="px-3 py-1 rounded-full text-white text-sm font-medium"
                    style={{ backgroundColor: PLAYER_COLORS[state.selectedColor] }}
                >
                    {state.selectedColor.charAt(0).toUpperCase() + state.selectedColor.slice(1)}
                </div>
            </div>

            {/* Camera viewfinder */}
            <div className="flex-1 relative overflow-hidden">
                {error ? (
                    <div className="absolute inset-0 flex items-center justify-center text-white text-center p-4">
                        <div>
                            <p className="text-lg mb-2">{error}</p>
                            <p className="text-sm text-gray-400">Please allow camera access to continue</p>
                        </div>
                    </div>
                ) : (
                    <video
                        ref={videoRef}
                        className="absolute inset-0 w-full h-full object-cover"
                        playsInline
                        muted
                    />
                )}
                <canvas ref={canvasRef} className="hidden" />

                {/* Guide overlay */}
                <div className="absolute inset-8 border-2 border-white/30 rounded-lg pointer-events-none" />
            </div>

            {/* Capture button */}
            <div className="absolute bottom-8 left-0 right-0 flex justify-center">
                <button
                    onClick={handleCapture}
                    disabled={!isReady}
                    className="w-20 h-20 rounded-full bg-white border-4 border-gray-300 disabled:opacity-50 transition-transform active:scale-95"
                >
                    <div className="w-full h-full rounded-full bg-white" />
                </button>
            </div>
        </div>
    );
}
