import { Link } from 'react-router-dom';

interface HeaderProps {
    title?: string;
}

export const Header = ({ title = 'blokus-ai' }: HeaderProps) => {
    return (
        <header className="w-full border-b border-border bg-surface shadow-gcp">
            <nav className="max-w-5xl mx-auto flex justify-between items-center px-4 sm:px-6 h-14">
                {/* Logo */}
                <Link to="/" className="flex items-center gap-2.5">
                    <div className="h-8 w-8 bg-primary rounded-lg flex items-center justify-center shadow-sm">
                        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                    </div>
                    <span className="text-lg font-semibold text-foreground">
                        {title}
                    </span>
                </Link>

                {/* Navigation */}
                <div className="flex items-center gap-4">
                    <Link
                        to="/users"
                        className="text-sm text-secondary hover:text-foreground transition-colors"
                    >
                        Users
                    </Link>
                </div>
            </nav>
        </header>
    );
};
