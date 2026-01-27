import { useState } from 'react';
import type { TrainingRun, TrainingRunStatus } from '@blokus-ai/shared';
import { Trash2, Loader2 } from 'lucide-react';
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

const phaseLabels: Record<string, string> = {
    detection: 'Detection',
    classifier: 'Classifier',
};

const StatusChip = ({ status, progress }: { status: TrainingRunStatus; progress?: TrainingProgress }) => (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-xs font-medium rounded-full ${statusStyles[status]}`}>
        {status === 'running' && <Loader2 className="w-3 h-3 animate-spin" />}
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
                if (value) {
                    return formatPercent(value);
                }
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
                if (value) {
                    return formatLoss(value);
                }
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
                <Trash2 className="w-4 h-4" />
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
                getRowKey={(row) => row.id}
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
