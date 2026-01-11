import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { GameProvider } from './contexts/GameContext';
import { ColorSelectionPage } from './pages/ColorSelectionPage';
import { CameraCapturePage } from './pages/CameraCapturePage';
import { BoardConfirmPage } from './pages/BoardConfirmPage';
import { ResultsPage } from './pages/ResultsPage';
import { MLAdminPage } from './pages/MLAdminPage';

function Header() {
    return (
        <header className="bg-surface border-b border-border px-4 py-4 flex items-center justify-between">
            <Link to="/" className="text-2xl font-bold text-foreground">
                Blokus AI
            </Link>
            <Link
                to="/admin"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
                Admin
            </Link>
        </header>
    );
}

function Layout({ children }: { children: React.ReactNode }) {
    return (
        <div className="min-h-screen bg-background">
            <Header />
            <main>{children}</main>
        </div>
    );
}

export default function App() {
    return (
        <GameProvider>
            <BrowserRouter>
                <Toaster />
                <Routes>
                    {/* Screen 1 has header via Layout */}
                    <Route path="/" element={<Layout><ColorSelectionPage /></Layout>} />
                    {/* Screens 2-4 are full-screen, no header */}
                    <Route path="/capture" element={<CameraCapturePage />} />
                    <Route path="/confirm" element={<BoardConfirmPage />} />
                    <Route path="/results" element={<ResultsPage />} />
                    <Route path="/admin" element={<Layout><MLAdminPage /></Layout>} />
                </Routes>
            </BrowserRouter>
        </GameProvider>
    );
}
