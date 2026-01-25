import { useState } from 'react';
import type { TrainingRun, TrainingRunStatus } from '@blokus-ai/shared';
import { Table, Column } from '../../../components/ui/Table';
import { ConfirmModal } from '../../../components/ui/ConfirmModal';
import { http } from '../../../api/config';
import toast from 'react-hot-toast';

export interface TrainingProgress {
    phase: 'detection' | 'classifier';
    epoch: number;
    totalEpochs: number;
    loss: number;
    accuracy: number;
    // Preserved detection metrics for when classifier phase starts
    detectionAccuracy?: number;
    detectionLoss?: number;
}

interface Props {
    runs: TrainingRun[];
    loading: boolean;
    progress?: Record<string, TrainingProgress>; // keyed by run ID
    onDelete?: (id: string) => void;
}

const TrashIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
);

const statusStyles: Record<TrainingRunStatus, string> = {
    pending: 'bg-surface-dim text-secondary',
    running: 'bg-info/20 text-info',
    completed: 'bg-success/20 text-success',
    failed: 'bg-error/20 text-error',
};

const statusLabels: Record<TrainingRunStatus, string> = {
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

const phaseLabels: Record<string, string> = {
    detection: 'Detection',
    classifier: 'Classifier',
};

const StatusChip = ({ status, progress }: { status: TrainingRunStatus; progress?: TrainingProgress }) => (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-xs font-medium rounded-full ${statusStyles[status]}`}>
        {status === 'running' && <SpinnerIcon />}
        {status === 'running' && progress
            ? `${phaseLabels[progress.phase] || progress.phase} ${progress.epoch}/${progress.totalEpochs}`
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

const formatLoss = (value: number | null): string => {
    if (value === null) {
        return '-';
    }
    return value.toFixed(4);
};

const getColumns = (
    progress?: Record<string, TrainingProgress>,
    onDeleteClick?: (run: TrainingRun) => void
): Column<TrainingRun>[] => [
    {
        key: 'status',
        header: 'Status',
        render: (row) => <StatusChip status={row.status} progress={progress?.[row.id]} />,
    },
    {
        key: 'imageCount',
        header: 'Images',
        render: (row) => row.imageCount,
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
        key: 'accuracy',
        header: 'Accuracy',
        render: (row) => {
            // Show live accuracy during training (use preserved value during classifier phase)
            const liveProgress = progress?.[row.id];
            if (row.status === 'running' && liveProgress) {
                const value = liveProgress.phase === 'detection'
                    ? liveProgress.accuracy
                    : liveProgress.detectionAccuracy;
                if (value) return formatPercent(value);
            }
            return formatPercent(row.accuracy);
        },
    },
    {
        key: 'loss',
        header: 'Loss',
        render: (row) => {
            // Show live loss during training (use preserved value during classifier phase)
            const liveProgress = progress?.[row.id];
            if (row.status === 'running' && liveProgress) {
                const value = liveProgress.phase === 'detection'
                    ? liveProgress.loss
                    : liveProgress.detectionLoss;
                if (value) return formatLoss(value);
            }
            return formatLoss(row.loss);
        },
    },
    {
        key: 'classifierAccuracy',
        header: 'Cls Acc',
        render: (row) => {
            // Show live accuracy during classifier phase
            const liveProgress = progress?.[row.id];
            if (row.status === 'running' && liveProgress?.phase === 'classifier' && liveProgress?.accuracy) {
                return formatPercent(liveProgress.accuracy);
            }
            return formatPercent(row.classifierAccuracy);
        },
    },
    {
        key: 'classifierLoss',
        header: 'Cls Loss',
        render: (row) => {
            // Show live loss during classifier phase
            const liveProgress = progress?.[row.id];
            if (row.status === 'running' && liveProgress?.phase === 'classifier' && liveProgress?.loss) {
                return formatLoss(liveProgress.loss);
            }
            return formatLoss(row.classifierLoss);
        },
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
                title="Delete training run"
            >
                <TrashIcon />
            </button>
        ),
    },
];

export const TrainedModelsTable = ({ runs, loading, progress, onDelete }: Props) => {
    const [deleteTarget, setDeleteTarget] = useState<TrainingRun | null>(null);
    const [isDeleting, setIsDeleting] = useState(false);

    const handleDelete = async () => {
        if (!deleteTarget) {
            return;
        }

        setIsDeleting(true);
        try {
            await http.delete(`/api/ml/training-runs/${deleteTarget.id}`);
            toast.success('Training run deleted');
            onDelete?.(deleteTarget.id);
            setDeleteTarget(null);
        } catch {
            toast.error('Failed to delete training run');
        } finally {
            setIsDeleting(false);
        }
    };

    const columns = getColumns(progress, setDeleteTarget);

    return (
        <>
            <Table<TrainingRun>
                columns={columns}
                data={runs}
                keyField="id"
                loading={loading}
                emptyMessage="No training runs yet. Click 'Train Model' to start your first training run."
            />
            <ConfirmModal
                isOpen={deleteTarget !== null}
                onClose={() => setDeleteTarget(null)}
                onConfirm={handleDelete}
                title="Delete Training Run"
                message={'Are you sure you want to delete this training run? This will also delete the trained model files. This action cannot be undone.'}
                confirmText="Delete"
                loading={isDeleting}
                variant="danger"
            />
        </>
    );
};
