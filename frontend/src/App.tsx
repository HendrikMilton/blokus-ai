import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Sun, Moon, Image, Gamepad2 } from 'lucide-react';
import { ThemeProvider } from './contexts/ThemeProvider';
import { useTheme } from './hooks/useTheme';
import { routes } from './routes';

const navItems = [
    { to: '/admin/images', icon: Image, label: 'Images' },
    { to: '/admin/gymnasium', icon: Gamepad2, label: 'Gymnasium' },
];

const Header = () => {
    const { theme, toggleTheme } = useTheme();
    const location = useLocation();

    return (
        <header className="h-14 bg-surface border-b border-border px-6 flex items-center justify-between shadow-gcp sticky top-0 z-50">
            <div className="flex items-center gap-6">
                <Link to="/admin" className="text-xl font-medium text-foreground">
                    Blokus AI
                </Link>
                <nav className="flex items-center gap-1">
                    {navItems.map(({ to, icon: Icon, label }) => {
                        const isActive = location.pathname.startsWith(to);
                        return (
                            <Link
                                key={to}
                                to={to}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                                    isActive
                                        ? 'bg-primary/10 text-primary'
                                        : 'text-secondary hover:text-foreground hover:bg-surface-dim'
                                }`}
                            >
                                <Icon className="w-4 h-4" />
                                {label}
                            </Link>
                        );
                    })}
                </nav>
            </div>
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
        </ThemeProvider>
    );
};

export default App;
