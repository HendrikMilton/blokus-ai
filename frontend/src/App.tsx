import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { initializeSocket, connectSocket } from './api/config';
import { Header } from './components/ui';
import { UsersPage } from './pages/UsersPage';

function Home() {
    return (
        <div className="flex items-center justify-center py-20">
            <div className="text-center">
                <h1 className="text-4xl font-bold text-foreground">blokus-ai</h1>
                <p className="mt-4 text-muted-foreground">AI-powered Blokus game assistant</p>
                <Link
                    to="/users"
                    className="mt-6 inline-block px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90"
                >
                    Manage Users
                </Link>
            </div>
        </div>
    );
}

function Layout({ children }: { children: React.ReactNode }) {
    return (
        <div className="min-h-screen bg-background">
            <Header />
            <main className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
                {children}
            </main>
        </div>
    );
}

export default function App() {
    useEffect(() => {
        // Initialize and connect WebSocket
        initializeSocket();
        connectSocket();
    }, []);

    return (
        <BrowserRouter>
            <Toaster />
            <Layout>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/users" element={<UsersPage />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    );
}
