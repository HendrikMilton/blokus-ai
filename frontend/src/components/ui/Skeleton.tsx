interface SkeletonProps {
    className?: string;
    variant?: 'text' | 'rectangular' | 'circular';
    width?: number | string;
    height?: number | string;
}

export const Skeleton = ({
    className = '',
    variant = 'text',
    width,
    height,
}: SkeletonProps) => {
    const variantClasses = {
        text: 'rounded-md',
        rectangular: 'rounded-lg',
        circular: 'rounded-full',
    };

    const style: React.CSSProperties = {};
    if (width) {
        style.width = typeof width === 'number' ? `${width}px` : width;
    }
    if (height) {
        style.height = typeof height === 'number' ? `${height}px` : height;
    }

    return (
        <div
            className={`relative overflow-hidden bg-muted/30 ${variantClasses[variant]} ${className}`}
            style={style}
        >
            <div
                className="absolute inset-0 animate-shimmer"
                style={{
                    background: 'linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.4) 50%, transparent 100%)',
                }}
            />
        </div>
    );
};

// Table skeleton for loading states
export const TableSkeleton = ({ rows = 5 }: { rows?: number }) => {
    return (
        <div className="w-full">
            {/* Header */}
            <div className="flex gap-4 py-3 px-4 border-b border-border bg-surface/50">
                <Skeleton className="h-4 w-12" />
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-4 w-48" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-4 w-24" />
            </div>
            {/* Rows */}
            {Array.from({ length: rows }).map((_, i) => (
                <div key={i} className="flex gap-4 py-4 px-4 border-b border-border">
                    <Skeleton className="h-4 w-12" />
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-4 w-48" />
                    <Skeleton className="h-4 w-16" />
                    <Skeleton className="h-4 w-16" />
                    <Skeleton className="h-4 w-24" />
                    <div className="flex gap-2">
                        <Skeleton className="h-6 w-12" />
                        <Skeleton className="h-6 w-16" />
                    </div>
                </div>
            ))}
        </div>
    );
};
