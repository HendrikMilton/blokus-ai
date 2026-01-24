/**
 * SampleGrid - Thumbnail grid of training samples with pagination
 */

import { TrainingSample, SampleStatus, PLAYER_COLORS, BOARD_SIZE } from '@blokus-ai/shared';
import { ADMIN_API } from '../../api/admin';

interface SampleGridProps {
    samples: TrainingSample[];
    selectedId: string | null;
    onSelect: (sample: TrainingSample) => void;
    loading?: boolean;
}

interface MiniboardProps {
    boardState: TrainingSample['detected_state'];
    size?: number;
}

const Miniboard = ({ boardState, size = 60 }: MiniboardProps) => <svg
    width={size}
    height={size}
    viewBox={`0 0 ${BOARD_SIZE} ${BOARD_SIZE}`}
    className="rounded"
>
    <rect width={BOARD_SIZE} height={BOARD_SIZE} fill="#1a1a2e" />
    {boardState.map((row, rowIdx) =>
        row.map((cell, colIdx) => {
            if (!cell.color) {
                return null;
            }
            return (
                <rect
                    key={`${rowIdx}-${colIdx}`}
                    x={colIdx}
                    y={rowIdx}
                    width={1}
                    height={1}
                    fill={PLAYER_COLORS[cell.color]}
                />
            );
        })
    )}
</svg>;

const STATUS_STYLES: Record<SampleStatus, { bg: string; text: string }> = {
    pending: { bg: 'bg-yellow-500/20', text: 'text-yellow-400' },
    labeled: { bg: 'bg-blue-500/20', text: 'text-blue-400' },
    verified: { bg: 'bg-green-500/20', text: 'text-green-400' },
};

const StatusBadge = ({ status }: { status: SampleStatus }) => {
    const styles = STATUS_STYLES[status];
    return (
        <span className={`px-2 py-0.5 text-xs rounded-full ${styles.bg} ${styles.text}`}>
            {status}
        </span>
    );
};

const SampleCard = ({
    sample,
    isSelected,
    onClick,
}: {
    sample: TrainingSample;
    isSelected: boolean;
    onClick: () => void;
}) => {
    const imageUrl = ADMIN_API.getSampleImageUrl(sample.id);
    const createdAt = new Date(sample.created_at);
    const timeAgo = getTimeAgo(createdAt);

    return (
        <button
            onClick={onClick}
            className={`
                relative flex flex-col bg-surface rounded-lg border overflow-hidden
                transition-all duration-150 text-left
                ${isSelected
            ? 'border-primary ring-2 ring-primary/50'
            : 'border-border hover:border-primary/50'
        }
            `}
        >
            {/* Thumbnail */}
            <div className="relative aspect-square bg-black/20">
                <img
                    src={imageUrl}
                    alt={`Sample ${sample.id.slice(0, 8)}`}
                    className="w-full h-full object-cover"
                    loading="lazy"
                />

                {/* Detected state overlay */}
                <div className="absolute bottom-1 right-1 shadow-lg rounded overflow-hidden">
                    <Miniboard boardState={sample.detected_state} size={40} />
                </div>
            </div>

            {/* Info */}
            <div className="p-2 flex flex-col gap-1">
                <div className="flex items-center justify-between">
                    <StatusBadge status={sample.status} />
                    <span className="text-xs text-muted-foreground">{timeAgo}</span>
                </div>
                {sample.notes && (
                    <p className="text-xs text-muted-foreground truncate">
                        {sample.notes}
                    </p>
                )}
            </div>
        </button>
    );
};

const getTimeAgo = (date: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) {
        return 'just now';
    }
    if (diffMins < 60) {
        return `${diffMins}m ago`;
    }
    if (diffHours < 24) {
        return `${diffHours}h ago`;
    }
    if (diffDays < 7) {
        return `${diffDays}d ago`;
    }

    return date.toLocaleDateString();
};

const LoadingSkeleton = () => <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
    {Array.from({ length: 8 }).map((_, i) => (
        <div
            key={i}
            className="flex flex-col bg-surface rounded-lg border border-border overflow-hidden animate-pulse"
        >
            <div className="aspect-square bg-muted" />
            <div className="p-2 space-y-2">
                <div className="flex justify-between">
                    <div className="h-5 w-16 bg-muted rounded-full" />
                    <div className="h-4 w-12 bg-muted rounded" />
                </div>
            </div>
        </div>
    ))}
</div>;

const EmptyState = () => <div className="flex flex-col items-center justify-center py-12 text-center">
    <div className="w-16 h-16 mb-4 rounded-full bg-surface flex items-center justify-center">
        <svg
            className="w-8 h-8 text-muted-foreground"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
        >
            <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
        </svg>
    </div>
    <h3 className="text-lg font-medium text-foreground mb-1">No samples yet</h3>
    <p className="text-sm text-muted-foreground">
                Upload board images to start labeling training data
    </p>
</div>;

export const SampleGrid = ({ samples, selectedId, onSelect, loading }: SampleGridProps) => {
    if (loading) {
        return <LoadingSkeleton />;
    }

    if (samples.length === 0) {
        return <EmptyState />;
    }

    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {samples.map((sample) => (
                <SampleCard
                    key={sample.id}
                    sample={sample}
                    isSelected={selectedId === sample.id}
                    onClick={() => onSelect(sample)}
                />
            ))}
        </div>
    );
};

interface PaginationProps {
    currentPage: number;
    totalPages: number;
    onPageChange: (page: number) => void;
}

export const Pagination = ({ currentPage, totalPages, onPageChange }: PaginationProps) => {
    if (totalPages <= 1) {
        return null;
    }

    return (
        <div className="flex items-center justify-center gap-2 mt-6">
            <button
                onClick={() => onPageChange(currentPage - 1)}
                disabled={currentPage <= 1}
                className="px-3 py-1.5 text-sm rounded-md bg-surface border border-border
                    disabled:opacity-50 disabled:cursor-not-allowed
                    hover:bg-surface/80 transition-colors"
            >
                Previous
            </button>

            <span className="px-3 py-1.5 text-sm text-muted-foreground">
                Page {currentPage} of {totalPages}
            </span>

            <button
                onClick={() => onPageChange(currentPage + 1)}
                disabled={currentPage >= totalPages}
                className="px-3 py-1.5 text-sm rounded-md bg-surface border border-border
                    disabled:opacity-50 disabled:cursor-not-allowed
                    hover:bg-surface/80 transition-colors"
            >
                Next
            </button>
        </div>
    );
};
