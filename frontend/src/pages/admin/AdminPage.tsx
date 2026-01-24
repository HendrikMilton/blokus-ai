import { Link } from 'react-router-dom';

const ChevronRightIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
    </svg>
);

const ImageIcon = () => (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
);

const navItems = [
    { to: '/admin/images', icon: ImageIcon, label: 'Training Images', description: 'Upload and manage board images' },
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
                        <Icon />
                    </div>
                    <div className="flex-1">
                        <p className="font-medium text-foreground">{label}</p>
                        <p className="text-sm text-secondary">{description}</p>
                    </div>
                    <ChevronRightIcon />
                </Link>
            ))}
        </div>
    </div>
);
