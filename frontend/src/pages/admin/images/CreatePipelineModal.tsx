import { useState, useEffect } from 'react';
import { GripVertical, X, Plus } from 'lucide-react';
import type { TrainingRun, ModelPipeline } from '@blokus-ai/shared';
import { Modal } from '../../../components/ui/Modal';
import { Button } from '../../../components/ui/Button';
import { Input } from '../../../components/ui/Input';
import { Select } from '../../../components/ui/Select';
import { formatShortId } from '../../../utils/format';

interface Props {
    isOpen: boolean;
    onClose: () => void;
    onSave: (name: string, componentIds: string[]) => Promise<void>;
    trainingRuns: TrainingRun[];
    editingPipeline?: ModelPipeline | null;
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

const formatPercent = (value: number | null): string => {
    if (value === null) {
        return '-';
    }
    return `${(value * 100).toFixed(1)}%`;
};

export const CreatePipelineModal = ({ isOpen, onClose, onSave, trainingRuns, editingPipeline }: Props) => {
    const [name, setName] = useState('');
    const [componentIds, setComponentIds] = useState<string[]>([]);
    const [isSaving, setIsSaving] = useState(false);
    const [dragIndex, setDragIndex] = useState<number | null>(null);

    // Filter to only completed training runs
    const completedRuns = trainingRuns.filter(run => run.status === 'completed');

    // Reset form when modal opens/closes or editing pipeline changes
    useEffect(() => {
        if (isOpen) {
            if (editingPipeline) {
                setName(editingPipeline.name);
                setComponentIds(editingPipeline.componentIds);
            } else {
                setName('');
                setComponentIds([]);
            }
        }
    }, [isOpen, editingPipeline]);

    const handleAddComponent = (modelId: string) => {
        if (modelId && !componentIds.includes(modelId)) {
            setComponentIds([...componentIds, modelId]);
        }
    };

    const handleRemoveComponent = (index: number) => {
        setComponentIds(componentIds.filter((_, i) => i !== index));
    };

    const handleDragStart = (index: number) => {
        setDragIndex(index);
    };

    const handleDragOver = (e: React.DragEvent, index: number) => {
        e.preventDefault();
        if (dragIndex === null || dragIndex === index) {
            return;
        }

        const newComponents = [...componentIds];
        const [removed] = newComponents.splice(dragIndex, 1);
        newComponents.splice(index, 0, removed);
        setComponentIds(newComponents);
        setDragIndex(index);
    };

    const handleDragEnd = () => {
        setDragIndex(null);
    };

    const handleSave = async () => {
        if (!name.trim() || componentIds.length === 0) {
            return;
        }

        setIsSaving(true);
        try {
            await onSave(name.trim(), componentIds);
            onClose();
        } finally {
            setIsSaving(false);
        }
    };

    const handleClose = () => {
        setName('');
        setComponentIds([]);
        onClose();
    };

    // Get available models (not yet added to the pipeline)
    const availableModels = completedRuns.filter(run => !componentIds.includes(run.id));

    // Create a map for quick lookup of training runs by ID
    const runMap = new Map(trainingRuns.map(r => [r.id, r]));

    return (
        <Modal
            isOpen={isOpen}
            onClose={handleClose}
            title={editingPipeline ? 'Edit Pipeline' : 'Create Pipeline'}
            size="md"
        >
            <div className="space-y-4">
                {/* Name input */}
                <Input
                    label="Pipeline Name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g., Combined v1"
                />

                {/* Component list */}
                <div className="space-y-2">
                    <label className="block text-sm font-medium text-foreground">
                        Pipeline Steps
                    </label>
                    <p className="text-xs text-muted">
                        Models are tried in order. Detection stops at the first model that finds a board.
                    </p>

                    {componentIds.length === 0 ? (
                        <div className="flex items-center justify-center py-6 bg-surface-dim rounded-lg border border-dashed border-border">
                            <p className="text-sm text-muted">Add models to create a pipeline</p>
                        </div>
                    ) : (
                        <div className="space-y-1">
                            {componentIds.map((id, index) => {
                                const run = runMap.get(id);
                                return (
                                    <div
                                        key={id}
                                        draggable
                                        onDragStart={() => handleDragStart(index)}
                                        onDragOver={(e) => handleDragOver(e, index)}
                                        onDragEnd={handleDragEnd}
                                        className={`
                                            flex items-center gap-2 p-2 bg-surface rounded-lg border border-border
                                            ${dragIndex === index ? 'opacity-50' : ''}
                                            cursor-move
                                        `}
                                    >
                                        <GripVertical className="w-4 h-4 text-muted flex-shrink-0" />
                                        <span className="text-sm text-muted w-6">{index + 1}.</span>
                                        <div className="flex-1 min-w-0">
                                            {run ? (
                                                <div className="text-sm">
                                                    <span className="font-medium text-foreground">
                                                        {formatDate(run.createdAt)}
                                                    </span>
                                                    <span className="text-muted ml-2">
                                                        {run.imageCount} images, {formatPercent(run.accuracy)} acc
                                                    </span>
                                                </div>
                                            ) : (
                                                <span className="text-sm text-error">
                                                    Model not found: {formatShortId(id)}
                                                </span>
                                            )}
                                        </div>
                                        <button
                                            onClick={() => handleRemoveComponent(index)}
                                            className="p-1 rounded hover:bg-error/10 text-muted hover:text-error transition-colors"
                                        >
                                            <X className="w-4 h-4" />
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                {/* Add model dropdown */}
                {availableModels.length > 0 && (
                    <div className="flex gap-2">
                        <Select
                            label="Add Model"
                            value=""
                            onChange={(v) => handleAddComponent(String(v))}
                            options={[
                                { value: '', label: 'Select a model to add...' },
                                ...availableModels.map(run => ({
                                    value: run.id,
                                    label: `${formatDate(run.createdAt)} - ${run.imageCount} images, ${formatPercent(run.accuracy)} acc`,
                                })),
                            ]}
                            className="flex-1"
                        />
                    </div>
                )}

                {availableModels.length === 0 && completedRuns.length > 0 && componentIds.length > 0 && (
                    <p className="text-xs text-muted">All available models have been added to the pipeline.</p>
                )}

                {completedRuns.length === 0 && (
                    <p className="text-sm text-muted bg-surface-dim px-3 py-2 rounded-lg">
                        No completed training runs available. Train a model first.
                    </p>
                )}

                {/* Actions */}
                <div className="flex justify-end gap-2 pt-2">
                    <Button variant="secondary" onClick={handleClose}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleSave}
                        disabled={!name.trim() || componentIds.length === 0}
                        loading={isSaving}
                        leftIcon={<Plus className="w-4 h-4" />}
                    >
                        {editingPipeline ? 'Save Changes' : 'Create Pipeline'}
                    </Button>
                </div>
            </div>
        </Modal>
    );
};
