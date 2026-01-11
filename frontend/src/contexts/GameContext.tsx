import { createContext, useContext, useState, ReactNode } from 'react';
import { GameState, PlayerColor } from '../types/game';
import { generateMockBoardState, generateMockAnalysis } from '../utils/mockData';

interface GameContextType {
    state: GameState;
    selectColor: (color: PlayerColor) => void;
    captureBoard: (imageData: string) => void;
    analyze: () => void;
    reset: () => void;
}

const initialState: GameState = {
    selectedColor: null,
    capturedImage: null,
    boardState: null,
    analysis: null,
};

const GameContext = createContext<GameContextType | null>(null);

export function GameProvider({ children }: { children: ReactNode }) {
    const [state, setState] = useState<GameState>(initialState);

    const selectColor = (color: PlayerColor) => {
        setState(prev => ({ ...prev, selectedColor: color }));
    };

    const captureBoard = (imageData: string) => {
        setState(prev => ({
            ...prev,
            capturedImage: imageData,
            boardState: generateMockBoardState(),
        }));
    };

    const analyze = () => {
        if (state.selectedColor) {
            setState(prev => ({
                ...prev,
                analysis: generateMockAnalysis(state.selectedColor!),
            }));
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
}

export function useGame() {
    const context = useContext(GameContext);
    if (!context) {
        throw new Error('useGame must be used within a GameProvider');
    }
    return context;
}
