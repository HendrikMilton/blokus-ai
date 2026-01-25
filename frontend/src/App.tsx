import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { GameProvider } from './contexts/GameContext';
import { ThemeProvider } from './contexts/ThemeProvider';
import { useTheme } from './hooks/useTheme';
import { routes } from './routes';
import { initializeSocket, connectSocket, disconnectSocket } from './api/config';

const SunIcon = () => (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
    </svg>
);

const MoonIcon = () => (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
    </svg>
);

const Header = () => {
    const { theme, toggleTheme } = useTheme();

    return (
        <header className="h-14 bg-surface border-b border-border px-6 flex items-center justify-between shadow-gcp sticky top-0 z-50">
            <Link to="/" className="text-xl font-medium text-foreground">
                Blokus AI
            </Link>
            <div className="flex items-center gap-4">
                <button
                    onClick={toggleTheme}
                    className="p-2 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                    {theme === 'dark' ? (
                        <SunIcon />
                    ) : (
                        <MoonIcon />
                    )}
                </button>
                <Link
                    to="/admin"
                    className="text-sm text-secondary hover:text-foreground transition-colors"
                >
                    Admin
                </Link>
            </div>
        </header>
    );
};

const Layout = ({ children }: { children: React.ReactNode }) => (
    <div className="min-h-screen bg-background">
        <Header />
        <main className="px-6 py-6">{children}</main>
    </div>
);

const App = () => {
    // Initialize and connect WebSocket on mount
    useEffect(() => {
        initializeSocket();
        connectSocket();

        return () => {
            disconnectSocket();
        };
    }, []);

    return (
        <ThemeProvider>
            <GameProvider>
                <BrowserRouter>
                    <Toaster />
                    <Routes>
                        {routes.map((route) => (
                            <Route
                                key={route.path}
                                path={route.path}
                                element={
                                    route.layout
                                        ? <Layout>{route.element}</Layout>
                                        : route.element
                                }
                            />
                        ))}
                    </Routes>
                </BrowserRouter>
            </GameProvider>
        </ThemeProvider>
    );
};

export default App;
