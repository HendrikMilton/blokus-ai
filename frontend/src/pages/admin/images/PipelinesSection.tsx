import { useState } from 'react';
import { Plus, Trash2, Edit2, Link } from 'lucide-react';
import type { ModelPipeline, TrainingRun } from '@blokus-ai/shared';
import { Button } from '../../../components/ui/Button';
import { Table, Column } from '../../../components/ui/Table';
import { ConfirmModal } from '../../../components/ui/ConfirmModal';

interface Props {
    pipelines: ModelPipeline[];
    trainingRuns: TrainingRun[];
    loading: boolean;
    onDelete: (id: string) => void;
    onEdit: (pipeline: ModelPipeline) => void;
    onCreate: () => void;
}

const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
};

export const PipelinesSection = ({ pipelines, trainingRuns, loading, onDelete, onEdit, onCreate }: Props) => {
    const [deleteId, setDeleteId] = useState<string | null>(null);
    const [isDeleting, setIsDeleting] = useState(false);

    // Create a map for quick lookup of training runs by ID
    const runMap = new Map(trainingRuns.map(r => [r.id, r]));

    const handleDelete = async () => {
        if (!deleteId) {
            return;
        }
        setIsDeleting(true);
        try {
            onDelete(deleteId);
        } finally {
            setIsDeleting(false);
            setDeleteId(null);
        }
    };

    const formatComponents = (componentIds: string[]) => {
        return componentIds.map((id, index) => {
            const run = runMap.get(id);
            const label = run
                ? `${formatDate(run.createdAt)} (${run.imageCount})`
                : id.slice(0, 8);
            return (
                <span key={id}>
                    {index > 0 && <span className="text-muted mx-1">&rarr;</span>}
                    <span className="text-foreground">{label}</span>
                </span>
            );
        });
    };

    const columns: Column<ModelPipeline>[] = [
        {
            key: 'name',
            header: 'Name',
            render: (pipeline) => (
                <div className="flex items-center gap-2">
                    <Link className="w-4 h-4 text-primary" />
                    <span className="font-medium text-foreground">{pipeline.name}</span>
                </div>
            ),
        },
        {
            key: 'components',
            header: 'Steps',
            noTruncate: true,
            render: (pipeline) => (
                <div className="text-sm">
                    {formatComponents(pipeline.componentIds)}
                </div>
            ),
        },
        {
            key: 'createdAt',
            header: 'Created',
            width: '140px',
            className: 'text-muted',
            render: (pipeline) => formatDate(pipeline.createdAt),
        },
        {
            key: 'actions',
            header: '',
            width: '100px',
            render: (pipeline) => (
                <div className="flex items-center justify-end gap-1">
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            onEdit(pipeline);
                        }}
                        className="p-1.5 rounded hover:bg-surface-dim text-muted hover:text-foreground transition-colors"
                        title="Edit pipeline"
                    >
                        <Edit2 className="w-4 h-4" />
                    </button>
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            setDeleteId(pipeline.id);
                        }}
                        className="p-1.5 rounded hover:bg-error/10 text-muted hover:text-error transition-colors"
                        title="Delete pipeline"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            ),
        },
    ];

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-foreground">Pipelines</h3>
                <Button
                    onClick={onCreate}
                    variant="secondary"
                    size="sm"
                    leftIcon={<Plus className="w-4 h-4" />}
                >
                    Create Pipeline
                </Button>
            </div>

            {loading ? (
                <div className="bg-surface rounded-lg border border-border overflow-hidden">
                    <div className="animate-pulse p-4 space-y-3">
                        {[...Array(2)].map((_, i) => (
                            <div key={i} className="h-10 bg-surface-dim rounded" />
                        ))}
                    </div>
                </div>
            ) : pipelines.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-center bg-surface rounded-lg border border-border">
                    <div className="w-12 h-12 bg-surface-dim rounded-full flex items-center justify-center mb-3">
                        <Link className="w-5 h-5 text-muted" />
                    </div>
                    <p className="text-secondary text-sm">No pipelines yet</p>
                    <p className="text-xs text-muted mt-1">Create a pipeline to chain models together</p>
                </div>
            ) : (
                <Table<ModelPipeline>
                    columns={columns}
                    data={pipelines}
                    getRowKey={(pipeline) => pipeline.id}
                    emptyMessage="No pipelines"
                />
            )}

            <ConfirmModal
                isOpen={deleteId !== null}
                onClose={() => setDeleteId(null)}
                onConfirm={handleDelete}
                title="Delete Pipeline"
                message="Are you sure you want to delete this pipeline? This action cannot be undone."
                confirmText="Delete"
                cancelText="Cancel"
                loading={isDeleting}
                variant="danger"
            />
        </div>
    );
};
