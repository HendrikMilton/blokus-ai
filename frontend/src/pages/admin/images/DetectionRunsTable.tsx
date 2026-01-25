import { useState } from 'react';
import type { DetectionRun, DetectionRunStatus } from '@blokus-ai/shared';
import { Table, Column } from '../../../components/ui/Table';
import { ConfirmModal } from '../../../components/ui/ConfirmModal';
import { http } from '../../../api/config';
import toast from 'react-hot-toast';

export interface DetectionProgress {
    processedCount: number;
    totalCount: number;
    successCount: number;
    failedCount: number;
}

interface Props {
    runs: DetectionRun[];
    loading: boolean;
    progress?: Record<string, DetectionProgress>; // keyed by run ID
    onDelete?: (id: string) => void;
}

const TrashIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
);

const statusStyles: Record<DetectionRunStatus, string> = {
    pending: 'bg-surface-dim text-secondary',
    running: 'bg-info/20 text-info',
    completed: 'bg-success/20 text-success',
    failed: 'bg-error/20 text-error',
};

const statusLabels: Record<DetectionRunStatus, string> = {
    pending: 'Pending',
    running: 'Running',
    completed: 'Completed',
    failed: 'Failed',
};

const SpinnerIcon = () => (
    <svg
        className="animate-spin h-3 w-3"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
    >
        <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
        />
        <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
    </svg>
);

const StatusChip = ({ status, progress }: { status: DetectionRunStatus; progress?: DetectionProgress }) => (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-xs font-medium rounded-full ${statusStyles[status]}`}>
        {status === 'running' && <SpinnerIcon />}
        {status === 'running' && progress
            ? `${progress.processedCount}/${progress.totalCount}`
            : statusLabels[status]
        }
    </span>
);

const formatDuration = (ms: number | null): string => {
    if (ms === null) {
        return '-';
    }
    if (ms < 1000) {
        return `${ms}ms`;
    }
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) {
        return `${seconds}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
};

const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
};

const formatPercent = (value: number | null): string => {
    if (value === null) {
        return '-';
    }
    return `${(value * 100).toFixed(1)}%`;
};

const getColumns = (
    progress?: Record<string, DetectionProgress>,
    onDeleteClick?: (run: DetectionRun) => void
): Column<DetectionRun>[] => [
    {
        key: 'status',
        header: 'Status',
        render: (row) => <StatusChip status={row.status} progress={progress?.[row.id]} />,
    },
    {
        key: 'totalCount',
        header: 'Total',
        render: (row) => row.totalCount,
    },
    {
        key: 'progress',
        header: 'Progress',
        render: (row) => {
            const liveProgress = progress?.[row.id];
            if (row.status === 'running' && liveProgress) {
                return `${liveProgress.processedCount}/${liveProgress.totalCount}`;
            }
            if (row.status === 'completed') {
                return `${row.processedCount}/${row.totalCount}`;
            }
            return '-';
        },
    },
    {
        key: 'successCount',
        header: 'Success',
        render: (row) => {
            const liveProgress = progress?.[row.id];
            if (row.status === 'running' && liveProgress) {
                return liveProgress.successCount;
            }
            return row.successCount ?? '-';
        },
    },
    {
        key: 'failedCount',
        header: 'Failed',
        render: (row) => {
            const liveProgress = progress?.[row.id];
            if (row.status === 'running' && liveProgress) {
                return liveProgress.failedCount;
            }
            return row.failedCount ?? '-';
        },
    },
    {
        key: 'createdAt',
        header: 'Created',
        render: (row) => formatDate(row.createdAt),
    },
    {
        key: 'durationMs',
        header: 'Duration',
        render: (row) => formatDuration(row.durationMs),
    },
    {
        key: 'avgIou',
        header: 'Avg IoU',
        render: (row) => formatPercent(row.avgIou),
    },
    {
        key: 'actions',
        header: '',
        render: (row) => (
            <button
                onClick={(e) => {
                    e.stopPropagation();
                    onDeleteClick?.(row);
                }}
                disabled={row.status === 'running'}
                className="p-1.5 text-secondary hover:text-error disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                title="Delete detection run"
            >
                <TrashIcon />
            </button>
        ),
    },
];

export const DetectionRunsTable = ({ runs, loading, progress, onDelete }: Props) => {
    const [deleteTarget, setDeleteTarget] = useState<DetectionRun | null>(null);
    const [isDeleting, setIsDeleting] = useState(false);

    const handleDelete = async () => {
        if (!deleteTarget) {
            return;
        }

        setIsDeleting(true);
        try {
            await http.delete(`/api/ml/detection-runs/${deleteTarget.id}`);
            toast.success('Detection run deleted and AI labels cleared');
            onDelete?.(deleteTarget.id);
            setDeleteTarget(null);
        } catch {
            toast.error('Failed to delete detection run');
        } finally {
            setIsDeleting(false);
        }
    };

    const columns = getColumns(progress, setDeleteTarget);

    return (
        <>
            <Table<DetectionRun>
                columns={columns}
                data={runs}
                keyField="id"
                loading={loading}
                emptyMessage="No detection runs yet. Click 'Run AI Detection' to start your first detection run."
            />
            <ConfirmModal
                isOpen={deleteTarget !== null}
                onClose={() => setDeleteTarget(null)}
                onConfirm={handleDelete}
                title="Delete Detection Run"
                message="Are you sure you want to delete this detection run? This will also clear all AI labels from images (human labels will be preserved). This action cannot be undone."
                confirmText="Delete"
                loading={isDeleting}
                variant="danger"
            />
        </>
    );
};
