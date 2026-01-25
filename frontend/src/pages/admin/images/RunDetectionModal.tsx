import { useState } from 'react';
import type { TrainingRun } from '@blokus-ai/shared';
import { Modal } from '../../../components/ui/Modal';
import { Button } from '../../../components/ui/Button';

interface Props {
    isOpen: boolean;
    onClose: () => void;
    onStart: (modelId: string, includeExistingAiLabeled: boolean) => void;
    trainingRuns: TrainingRun[];
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

export const RunDetectionModal = ({ isOpen, onClose, onStart, trainingRuns, isLoading }: Props) => {
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

    return (
        <Modal isOpen={isOpen} onClose={handleClose} title="Run AI Detection" size="md">
            <div className="space-y-4">
                {/* Model selection */}
                <div>
                    <label className="block text-sm font-medium text-foreground mb-1.5">
                        Select trained model
                    </label>
                    {completedRuns.length === 0 ? (
                        <p className="text-sm text-muted">
                            No completed training runs available. Please train a model first.
                        </p>
                    ) : (
                        <select
                            value={selectedModelId}
                            onChange={(e) => setSelectedModelId(e.target.value)}
                            className="w-full px-3 py-2 bg-surface border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            {completedRuns.map(run => (
                                <option key={run.id} value={run.id}>
                                    {formatDate(run.createdAt)} - Accuracy: {formatPercent(run.accuracy)} ({run.imageCount} images)
                                </option>
                            ))}
                        </select>
                    )}
                </div>

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
                        disabled={!selectedModelId || completedRuns.length === 0}
                        loading={isLoading}
                    >
                        Start Detection
                    </Button>
                </div>
            </div>
        </Modal>
    );
};
