import { createContext, useContext, useState, ReactNode } from 'react';
import { GameState, PlayerColor, AnalysisResult } from '../types/game';
import { ML_API } from '../api/ml';

interface GameContextType {
    state: GameState;
    selectColor: (color: PlayerColor) => void;
    captureBoard: (imageData: string) => Promise<void>;
    analyze: () => Promise<void>;
    reset: () => void;
}

const initialState: GameState = {
    selectedColor: null,
    capturedImage: null,
    boardState: null,
    analysis: null,
    isDetecting: false,
    isAnalyzing: false,
};

const GameContext = createContext<GameContextType | null>(null);

export const GameProvider = ({ children }: { children: ReactNode }) => {
    const [state, setState] = useState<GameState>(initialState);

    const selectColor = (color: PlayerColor) => {
        setState(prev => ({ ...prev, selectedColor: color }));
    };

    const captureBoard = async (imageData: string) => {
        setState(prev => ({
            ...prev,
            capturedImage: imageData,
            isDetecting: true,
        }));

        try {
            const response = await ML_API.detectBoard(imageData);
            setState(prev => ({
                ...prev,
                boardState: response.boardState,
                isDetecting: false,
            }));
        } catch (error) {
            setState(prev => ({
                ...prev,
                isDetecting: false,
            }));
            throw error;
        }
    };

    const analyze = async () => {
        if (!state.selectedColor || !state.boardState) {
            return;
        }

        setState(prev => ({ ...prev, isAnalyzing: true }));

        try {
            const response = await ML_API.analyze(state.boardState, state.selectedColor);
            const analysis: AnalysisResult = {
                recommendedPiece: response.recommendedPiece,
                placement: response.placement,
                stats: response.stats,
            };
            setState(prev => ({
                ...prev,
                analysis,
                isAnalyzing: false,
            }));
        } catch (error) {
            setState(prev => ({
                ...prev,
                isAnalyzing: false,
            }));
            throw error;
        }
    };

    const reset = () => {
        setState(initialState);
    };

    return (
        <GameContext.Provider value={{ state, selectColor, captureBoard, analyze, reset }}>
            {children}
        </GameContext.Provider>
    );
};

export const useGame = () => {
    const context = useContext(GameContext);
    if (!context) {
        throw new Error('useGame must be used within a GameProvider');
    }
    return context;
};
