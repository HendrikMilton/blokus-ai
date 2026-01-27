import { useState } from 'react';
import type { TrainingRun, ModelPipeline } from '@blokus-ai/shared';
import { Modal } from '../../../components/ui/Modal';
import { Button } from '../../../components/ui/Button';
import { Select } from '../../../components/ui/Select';

interface Props {
    isOpen: boolean;
    onClose: () => void;
    onStart: (modelId: string, includeExistingAiLabeled: boolean) => void;
    trainingRuns: TrainingRun[];
    pipelines: ModelPipeline[];
    isLoading: boolean;
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

export const RunDetectionModal = ({ isOpen, onClose, onStart, trainingRuns, pipelines, isLoading }: Props) => {
    const [selectedModelId, setSelectedModelId] = useState<string>('');
    const [includeExistingAiLabeled, setIncludeExistingAiLabeled] = useState(false);

    // Filter to only completed training runs
    const completedRuns = trainingRuns.filter(run => run.status === 'completed');

    // Set default selection when runs are loaded
    if (completedRuns.length > 0 && !selectedModelId) {
        setSelectedModelId(completedRuns[0].id);
    }

    const handleStart = () => {
        if (selectedModelId) {
            onStart(selectedModelId, includeExistingAiLabeled);
        }
    };

    const handleClose = () => {
        setSelectedModelId('');
        setIncludeExistingAiLabeled(false);
        onClose();
    };

    // Build model options with pipelines grouped separately
    const modelOptions = [
        // Models section
        ...completedRuns.map(run => ({
            value: run.id,
            label: `${formatDate(run.createdAt)} - ${formatPercent(run.accuracy)} acc (${run.imageCount} imgs)`,
        })),
        // Pipelines section (prefixed with link emoji)
        ...pipelines.map(pipeline => ({
            value: `pipeline:${pipeline.id}`,
            label: `\u{1F517} ${pipeline.name} (${pipeline.componentIds.length} models)`,
        })),
    ];

    return (
        <Modal isOpen={isOpen} onClose={handleClose} title="Run AI Detection" size="md">
            <div className="space-y-4">
                {/* Model selection */}
                {completedRuns.length === 0 && pipelines.length === 0 ? (
                    <p className="text-sm text-muted">
                        No completed training runs available. Please train a model first.
                    </p>
                ) : (
                    <Select
                        label="Select model or pipeline"
                        value={selectedModelId}
                        onChange={(v) => setSelectedModelId(v as string)}
                        options={modelOptions}
                    />
                )}

                {/* Include existing AI detections checkbox */}
                <div className="flex items-start gap-2">
                    <input
                        type="checkbox"
                        id="includeExisting"
                        checked={includeExistingAiLabeled}
                        onChange={(e) => setIncludeExistingAiLabeled(e.target.checked)}
                        className="mt-1 h-4 w-4 rounded border-border text-primary focus:ring-primary/30"
                    />
                    <label htmlFor="includeExisting" className="text-sm text-secondary">
                        Include samples with existing AI detections (re-process them with new model)
                    </label>
                </div>

                {/* Info note */}
                <p className="text-xs text-muted bg-surface-dim px-3 py-2 rounded-lg">
                    Human-labeled samples will not be overwritten to protect curated data.
                </p>

                {/* Actions */}
                <div className="flex justify-end gap-2 pt-2">
                    <Button variant="secondary" onClick={handleClose}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleStart}
                        disabled={!selectedModelId || (completedRuns.length === 0 && pipelines.length === 0)}
                        loading={isLoading}
                    >
                        Start Detection
                    </Button>
                </div>
            </div>
        </Modal>
    );
};
