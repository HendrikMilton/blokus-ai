import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { http, API_URL } from '../../../../api/config';
import { Button } from '../../../../components/ui/Button';
import type { BoundingBox, DetectResponseWithCorrection, SampleStatusFlags } from '@blokus-ai/shared';
import { getStatusLabels } from '@blokus-ai/shared';
import { statusColors, statusLabels } from '../../../../lib/status';

interface Sample {
    id: string;
    filename: string;
    status: SampleStatusFlags;
    created_at: string;
    imageUrl?: string;
    thumbnailUrl?: string;
}

const BackIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
    </svg>
);

const EditIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
    </svg>
);

export const ImageDetailPage = () => {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const [sample, setSample] = useState<Sample | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [detection, setDetection] = useState<DetectResponseWithCorrection | null>(null);

    // Get the current bounding box (correction overrides prediction)
    const currentDetection = detection?.correction || detection?.detection;
    const boundingBox = currentDetection?.boundingBox;

    useEffect(() => {
        const loadSample = async () => {
            try {
                const data = await http.get<Sample>(`/api/ml/samples/${id}`);
                setSample(data);
                // Try to load existing detection result
                if (id) {
                    const detectResult = await http.get<DetectResponseWithCorrection | null>(`/api/ml/detect/${id}`);
                    setDetection(detectResult);
                }
            } catch {
                // Error toast shown by http client
            } finally {
                setIsLoading(false);
            }
        };
        loadSample();
    }, [id]);

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    const formatBbox = (bbox: BoundingBox) => {
        return `x: ${(bbox.x * 100).toFixed(1)}%, y: ${(bbox.y * 100).toFixed(1)}%, w: ${(bbox.width * 100).toFixed(1)}%, h: ${(bbox.height * 100).toFixed(1)}%`;
    };

    if (isLoading) {
        return (
            <div className="space-y-6">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-surface-dim rounded-lg animate-pulse" />
                    <div className="h-6 w-48 bg-surface-dim rounded animate-pulse" />
                </div>
                <div className="aspect-video max-w-4xl bg-surface-dim rounded-lg animate-pulse" />
            </div>
        );
    }

    if (!sample) {
        return (
            <div className="space-y-6">
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => navigate(-1)}
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <BackIcon />
                    </button>
                    <h1 className="text-xl font-medium text-foreground">Image not found</h1>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between gap-4 flex-wrap">
                <div className="space-y-2">
                    <div className="flex items-center gap-2 flex-wrap">
                        <button
                            onClick={() => navigate(-1)}
                            className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                        >
                            <BackIcon />
                        </button>
                        <h1 className="text-lg sm:text-xl font-medium text-foreground font-mono">{sample.id.slice(0, 8)}</h1>
                        {getStatusLabels(sample.status).map(label => (
                            <span
                                key={label}
                                className={`px-2 py-0.5 text-xs font-medium rounded-full ${statusColors[label]}`}
                            >
                                {statusLabels[label]}
                            </span>
                        ))}
                    </div>
                    <p className="text-xs sm:text-sm text-secondary">
                        {formatDate(sample.created_at)}
                    </p>
                </div>
                <Button
                    onClick={() => navigate(`/admin/images/label/${sample.id}`)}
                    leftIcon={<EditIcon />}
                >
                    Edit in labeling mode
                </Button>
            </div>

            {/* Content container - centered with max width */}
            <div className="max-w-2xl mx-auto space-y-6">

                {/* Image with bounding box overlay (read-only) */}
                <div className="relative bg-surface rounded-lg border border-border overflow-hidden">
                    <img
                        src={sample.thumbnailUrl || sample.imageUrl || `${API_URL}/api/ml/samples/${sample.id}/image`}
                        alt={`Sample ${sample.id}`}
                        className="w-full h-auto select-none"
                        draggable={false}
                    />
                    {/* Bounding box overlay (read-only) */}
                    {boundingBox && (
                        <div
                            className={`absolute rounded-sm pointer-events-none ${
                                detection?.correction?.boundingBox
                                    ? 'border-2 border-success'
                                    : 'border-2 border-primary border-dashed'
                            }`}
                            style={{
                                left: `${boundingBox.x * 100}%`,
                                top: `${boundingBox.y * 100}%`,
                                width: `${boundingBox.width * 100}%`,
                                height: `${boundingBox.height * 100}%`,
                            }}
                        >
                            {/* Label */}
                            <div className={`absolute -top-6 left-0 text-white text-xs px-1.5 py-0.5 rounded ${
                                detection?.isHumanLabeled ? 'bg-info' : detection?.correction?.boundingBox ? 'bg-success' : 'bg-primary'
                            }`}>
                                {detection?.isHumanLabeled ? 'Human Labeled' : detection?.correction?.boundingBox ? 'Corrected' : 'AI Predicted'}
                            </div>
                        </div>
                    )}
                </div>

                {/* Detection Info - read only */}
                {detection && currentDetection && (
                    <div className="bg-surface rounded-lg border border-border p-4">
                        <div className="flex items-center justify-between mb-3">
                            <h2 className="text-sm font-medium text-foreground">
                                {detection.isHumanLabeled ? 'Human Label' : 'Detection Result'}
                            </h2>
                            {!detection.isHumanLabeled && detection.correction && (
                                <span className="text-xs text-success font-medium">Corrected</span>
                            )}
                        </div>
                        <div className="space-y-3">
                            {/* Board detected */}
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-sm text-secondary">Board detected</span>
                                <span className={`px-2 py-0.5 text-xs font-medium rounded ${
                                    currentDetection.hasBoardDetected
                                        ? 'bg-success/20 text-success'
                                        : 'bg-error/20 text-error'
                                }`}>
                                    {currentDetection.hasBoardDetected ? 'Yes' : 'No'}
                                </span>
                            </div>

                            {/* Entire board visible */}
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-sm text-secondary">Entire board visible</span>
                                <span className={`px-2 py-0.5 text-xs font-medium rounded ${
                                    currentDetection.isEntireBoardVisible
                                        ? 'bg-success/20 text-success'
                                        : 'bg-error/20 text-error'
                                }`}>
                                    {currentDetection.isEntireBoardVisible ? 'Yes' : 'No'}
                                </span>
                            </div>

                            {/* Confidence */}
                            {detection.detection?.confidence !== undefined && detection.detection.confidence > 0 && (
                                <div className="flex items-center justify-between gap-4">
                                    <span className="text-sm text-secondary">AI Confidence</span>
                                    <span className="text-sm font-mono text-foreground">
                                        {(detection.detection.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                            )}

                            {/* Bounding box coordinates */}
                            {boundingBox && (
                                <div className="pt-2 border-t border-border">
                                    <span className="text-xs text-muted">Bounding box</span>
                                    <p className="text-xs font-mono text-foreground mt-1">
                                        {formatBbox(boundingBox)}
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* No detection message */}
                {!detection && (
                    <div className="bg-surface rounded-lg border border-border p-6 text-center">
                        <p className="text-secondary">No labels yet</p>
                        <p className="text-sm text-muted mt-1">Use labeling mode to add labels to this image</p>
                    </div>
                )}

            </div>
        </div>
    );
};
