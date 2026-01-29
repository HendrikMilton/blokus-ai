import { useState } from 'react';
import { Plus, Trash2, Tag as TagIcon } from 'lucide-react';
import type { Tag } from '@blokus-ai/shared';
import { Button } from '../../../components/ui/Button';
import { Table, Column } from '../../../components/ui/Table';
import { ConfirmModal } from '../../../components/ui/ConfirmModal';

interface Props {
    tags: Tag[];
    loading: boolean;
    onDelete: (id: string) => void;
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

export const TagsSection = ({ tags, loading, onDelete, onCreate }: Props) => {
    const [deleteId, setDeleteId] = useState<string | null>(null);
    const [isDeleting, setIsDeleting] = useState(false);

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

    const columns: Column<Tag>[] = [
        {
            key: 'name',
            header: 'Name',
            render: (tag) => (
                <div className="flex items-center gap-2">
                    <TagIcon className="w-4 h-4 text-primary" />
                    <span className="font-medium text-foreground">{tag.name}</span>
                </div>
            ),
        },
        {
            key: 'createdAt',
            header: 'Created',
            width: '140px',
            className: 'text-muted',
            render: (tag) => formatDate(tag.createdAt),
        },
        {
            key: 'actions',
            header: '',
            width: '60px',
            render: (tag) => (
                <div className="flex items-center justify-end gap-1">
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            setDeleteId(tag.id);
                        }}
                        className="p-1.5 rounded hover:bg-error/10 text-muted hover:text-error transition-colors"
                        title="Delete tag"
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
                <h3 className="text-sm font-medium text-foreground">Tags</h3>
                <Button
                    onClick={onCreate}
                    variant="secondary"
                    size="sm"
                    leftIcon={<Plus className="w-4 h-4" />}
                >
                    Create Tag
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
            ) : tags.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-center bg-surface rounded-lg border border-border">
                    <div className="w-12 h-12 bg-surface-dim rounded-full flex items-center justify-center mb-3">
                        <TagIcon className="w-5 h-5 text-muted" />
                    </div>
                    <p className="text-secondary text-sm">No tags yet</p>
                    <p className="text-xs text-muted mt-1">Create tags to organize your samples</p>
                </div>
            ) : (
                <Table<Tag>
                    columns={columns}
                    data={tags}
                    getRowKey={(tag) => tag.id}
                    emptyMessage="No tags"
                />
            )}

            <ConfirmModal
                isOpen={deleteId !== null}
                onClose={() => setDeleteId(null)}
                onConfirm={handleDelete}
                title="Delete Tag"
                message="Are you sure you want to delete this tag? It will be removed from all samples."
                confirmText="Delete"
                cancelText="Cancel"
                loading={isDeleting}
                variant="danger"
            />
        </div>
    );
};
