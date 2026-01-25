import type { SampleStatusFlags, SampleStatusDisplay } from '@blokus-ai/shared';

export const statusColors: Record<SampleStatusDisplay, string> = {
    unlabeled: 'bg-warning/20 text-warning',
    'ai-labeled': 'bg-primary/20 text-primary',
    'human-labeled': 'bg-info/20 text-info',
    verified: 'bg-success/20 text-success',
};

export const statusLabels: Record<SampleStatusDisplay, string> = {
    unlabeled: 'Unlabeled',
    'ai-labeled': 'AI Labeled',
    'human-labeled': 'Human Labeled',
    verified: 'Verified',
};

export const isUnlabeled = (status: SampleStatusFlags) =>
    !status.aiLabeled && !status.humanLabeled && !status.verified;
