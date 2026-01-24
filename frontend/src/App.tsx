import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { GameProvider } from './contexts/GameContext';
import { ColorSelectionPage } from './pages/color-selection/ColorSelectionPage';
import { CameraCapturePage } from './pages/camera-capture/CameraCapturePage';
import { BoardConfirmPage } from './pages/board-confirm/BoardConfirmPage';
import { ResultsPage } from './pages/results/ResultsPage';
import { MLAdminPage } from './pages/ml-admin/MLAdminPage';

const Header = () => <header className="bg-surface border-b border-border px-4 py-4 flex items-center justify-between">
    <Link to="/" className="text-2xl font-bold text-foreground">
                Blokus AI
    </Link>
    <Link
        to="/admin"
        className="text-sm text-muted-foreground hover:text-foreground transition-colors"
    >
                Admin
    </Link>
</header>;

const Layout = ({ children }: { children: React.ReactNode }) => <div className="min-h-screen bg-background">
    <Header />
    <main>{children}</main>
</div>;

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
