import { Link } from 'react-router-dom';
import { ChevronRight, Image, Gamepad2 } from 'lucide-react';

const navItems = [
    { to: '/admin/images', icon: Image, label: 'Training Images', description: 'Upload and manage board images' },
    { to: '/admin/gymnasium', icon: Gamepad2, label: 'Gymnasium', description: 'Play Blokus and train AI agents' },
];

export const AdminPage = () => (
    <div className="space-y-6">
        <div>
            <h1 className="text-xl font-medium text-foreground">ML Admin</h1>
            <p className="text-sm text-secondary mt-1">Manage training data and models</p>
        </div>

        <div className="grid gap-3">
            {navItems.map(({ to, icon: Icon, label, description }) => (
                <Link
                    key={to}
                    to={to}
                    className="flex items-center gap-4 p-4 bg-surface rounded-lg border border-border hover:shadow-gcp-hover transition-shadow"
                >
                    <div className="w-10 h-10 bg-surface-dim rounded-lg flex items-center justify-center text-secondary">
                        <Icon className="w-6 h-6" />
                    </div>
                    <div className="flex-1">
                        <p className="font-medium text-foreground">{label}</p>
                        <p className="text-sm text-secondary">{description}</p>
                    </div>
                    <ChevronRight className="w-4 h-4" />
                </Link>
            ))}
        </div>
    </div>
);
