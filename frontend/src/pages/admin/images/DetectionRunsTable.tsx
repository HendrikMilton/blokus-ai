import { useState } from 'react';
import type { DetectionRun, DetectionRunStatus } from '@blokus-ai/shared';
import { Trash2, Loader2 } from 'lucide-react';
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

const StatusChip = ({ status, progress }: { status: DetectionRunStatus; progress?: DetectionProgress }) => (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-xs font-medium rounded-full ${statusStyles[status]}`}>
        {status === 'running' && <Loader2 className="w-3 h-3 animate-spin" />}
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
                <Trash2 className="w-4 h-4" />
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
                getRowKey={(row) => row.id}
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
