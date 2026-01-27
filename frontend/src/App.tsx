import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Sun, Moon } from 'lucide-react';
import { GameProvider } from './contexts/GameContext';
import { ThemeProvider } from './contexts/ThemeProvider';
import { useTheme } from './hooks/useTheme';
import { routes } from './routes';

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
                        <Sun className="w-5 h-5" />
                    ) : (
                        <Moon className="w-5 h-5" />
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
