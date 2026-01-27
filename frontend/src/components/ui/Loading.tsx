import { Loader2 } from 'lucide-react';

interface LoadingProps {
    size?: 'sm' | 'md' | 'lg';
    className?: string;
}

const sizes = {
    sm: 'h-4 w-4',
    md: 'h-8 w-8',
    lg: 'h-12 w-12',
};

export const Loading = ({ size = 'md', className = '' }: LoadingProps) => {
    return (
        <Loader2 className={`animate-spin text-primary ${sizes[size]} ${className}`} />
    );
};

export const LoadingPage = () => {
    return (
        <div className="flex items-center justify-center min-h-screen">
            <Loading size="lg" />
        </div>
    );
};

export const LoadingOverlay = () => {
    return (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
            <Loading size="lg" />
        </div>
    );
};
