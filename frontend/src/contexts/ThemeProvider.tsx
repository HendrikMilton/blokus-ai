import { ReactNode } from 'react';
import { ThemeContext, useThemeState } from '../hooks/useTheme';

interface ThemeProviderProps {
    children: ReactNode;
}

export const ThemeProvider = ({ children }: ThemeProviderProps) => {
    const themeState = useThemeState();

    return (
        <ThemeContext.Provider value={themeState}>
            <div className={themeState.theme === 'dark' ? 'dark' : ''}>
                {children}
            </div>
        </ThemeContext.Provider>
    );
};
