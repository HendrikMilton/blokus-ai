import { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, Pencil, Play } from 'lucide-react';
import { http, API_URL } from '../../../../api/config';
import { Button } from '../../../../components/ui/Button';
import { Select } from '../../../../components/ui/Select';
import type { BoundingBox, DetectResponseWithCorrection, SampleStatusFlags } from '@blokus-ai/shared';
import { getStatusLabels } from '@blokus-ai/shared';
import { statusColors, statusLabels } from '../../../../lib/status';

interface TrainingRun {
    id: string;
    status: string;
    imageCount: number;
    accuracy: number | null;
    createdAt: string;
    completedAt: string | null;
}

/**
 * Calculate Intersection over Union (IoU) between two bounding boxes.
 * Returns a value between 0 (no overlap) and 1 (perfect overlap).
 */
const calculateIoU = (box1: BoundingBox, box2: BoundingBox): number => {
    // Calculate intersection
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    const intersectionWidth = Math.max(0, x2 - x1);
    const intersectionHeight = Math.max(0, y2 - y1);
    const intersectionArea = intersectionWidth * intersectionHeight;

    // Calculate union
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;
    const unionArea = box1Area + box2Area - intersectionArea;

    if (unionArea === 0) {
        return 0;
    }
    return intersectionArea / unionArea;
};

interface Sample {
    id: string;
    filename: string;
    status: SampleStatusFlags;
    created_at: string;
    imageUrl?: string;
    thumbnailUrl?: string;
}

interface SampleListItem {
    id: string;
}


export const ImageDetailPage = () => {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const [sample, setSample] = useState<Sample | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [detection, setDetection] = useState<DetectResponseWithCorrection | null>(null);
    const [sampleIds, setSampleIds] = useState<string[]>([]);
    const [currentIndex, setCurrentIndex] = useState(-1);
    const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
    const [selectedModelId, setSelectedModelId] = useState<string>('');
    const [isDetecting, setIsDetecting] = useState(false);
    const [previewDetection, setPreviewDetection] = useState<DetectResponseWithCorrection['detection'] | null>(null);

    // Load sample list for navigation
    useEffect(() => {
        const loadSampleIds = async () => {
            try {
                const data = await http.get<{ samples: SampleListItem[] }>('/api/ml/samples');
                const ids = data.samples.map(s => s.id);
                setSampleIds(ids);
            } catch {
                // Ignore error
            }
        };
        loadSampleIds();
    }, []);

    // Update current index when id or sample list changes
    useEffect(() => {
        if (id && sampleIds.length > 0) {
            const index = sampleIds.indexOf(id);
            setCurrentIndex(index);
        }
    }, [id, sampleIds]);

    const goToPrevious = useCallback(() => {
        if (currentIndex > 0) {
            navigate(`/admin/images/${sampleIds[currentIndex - 1]}`);
        }
    }, [currentIndex, sampleIds, navigate]);

    const goToNext = useCallback(() => {
        if (currentIndex < sampleIds.length - 1) {
            navigate(`/admin/images/${sampleIds[currentIndex + 1]}`);
        }
    }, [currentIndex, sampleIds, navigate]);

    // Keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'ArrowLeft') {
                goToPrevious();
            } else if (e.key === 'ArrowRight') {
                goToNext();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [goToPrevious, goToNext]);

    // Load training runs for model selection
    useEffect(() => {
        const loadTrainingRuns = async () => {
            try {
                const data = await http.get<{ runs: TrainingRun[] }>('/api/ml/training-runs');
                const completedRuns = data.runs.filter(r => r.status === 'completed');
                setTrainingRuns(completedRuns);
                // Auto-select the most recent completed run
                if (completedRuns.length > 0 && !selectedModelId) {
                    setSelectedModelId(completedRuns[0].id);
                }
            } catch {
                // Ignore error
            }
        };
        loadTrainingRuns();
    }, []);

    const runDetection = useCallback(async () => {
        if (!id || !selectedModelId) {
            return;
        }

        setIsDetecting(true);
        try {
            const result = await http.get<{ detection: DetectResponseWithCorrection['detection'] }>(`/api/ml/detect/${id}/preview?modelId=${selectedModelId}`);
            setPreviewDetection(result.detection);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsDetecting(false);
        }
    }, [id, selectedModelId]);

    useEffect(() => {
        const loadSample = async () => {
            // Clear preview when navigating to a different sample
            setPreviewDetection(null);
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

    // Calculate IoU if both AI and human bboxes exist
    const iou = useMemo(() => {
        const aiBox = detection?.detection?.boundingBox;
        const humanBox = detection?.correction?.boundingBox;
        if (aiBox && humanBox && detection?.detection?.confidence > 0) {
            return calculateIoU(aiBox, humanBox);
        }
        return null;
    }, [detection]);

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
                        <ChevronLeft className="w-4 h-4" />
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
                            <ChevronLeft className="w-4 h-4" />
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
                <div className="flex items-center gap-2">
                    {/* Navigation arrows */}
                    <div className="flex items-center gap-1">
                        <button
                            onClick={goToPrevious}
                            disabled={currentIndex <= 0}
                            className="p-2 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground disabled:opacity-30 disabled:cursor-not-allowed"
                            title="Previous image (←)"
                        >
                            <ChevronLeft className="w-5 h-5" />
                        </button>
                        {sampleIds.length > 0 && (
                            <span className="text-xs text-muted min-w-[4rem] text-center">
                                {currentIndex + 1} / {sampleIds.length}
                            </span>
                        )}
                        <button
                            onClick={goToNext}
                            disabled={currentIndex >= sampleIds.length - 1}
                            className="p-2 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground disabled:opacity-30 disabled:cursor-not-allowed"
                            title="Next image (→)"
                        >
                            <ChevronRight className="w-5 h-5" />
                        </button>
                    </div>
                    <Button
                        onClick={() => navigate(`/admin/images/label/${sample.id}`)}
                        leftIcon={<Pencil className="w-4 h-4" />}
                    >
                        Edit in labeling mode
                    </Button>
                </div>
            </div>

            {/* Content container - centered with max width */}
            <div className="max-w-2xl mx-auto space-y-6">

                {/* Image with bounding box overlays (read-only) */}
                <div className="relative bg-surface rounded-lg border border-border overflow-hidden">
                    <img
                        src={sample.thumbnailUrl || sample.imageUrl || `${API_URL}/api/ml/samples/${sample.id}/image`}
                        alt={`Sample ${sample.id}`}
                        className="w-full h-auto select-none"
                        draggable={false}
                    />
                    {/* AI Detection bounding box (dashed, primary color) */}
                    {detection?.detection?.boundingBox && detection.detection.confidence > 0 && (
                        <div
                            className="absolute rounded-sm pointer-events-none border-2 border-primary border-dashed"
                            style={{
                                left: `${detection.detection.boundingBox.x * 100}%`,
                                top: `${detection.detection.boundingBox.y * 100}%`,
                                width: `${detection.detection.boundingBox.width * 100}%`,
                                height: `${detection.detection.boundingBox.height * 100}%`,
                            }}
                        >
                            <div className="absolute -top-6 left-0 text-white text-xs px-1.5 py-0.5 rounded bg-primary">
                                AI Detection
                            </div>
                        </div>
                    )}
                    {/* Human Label bounding box (solid, info color) */}
                    {detection?.correction?.boundingBox && (
                        <div
                            className="absolute rounded-sm pointer-events-none border-2 border-info"
                            style={{
                                left: `${detection.correction.boundingBox.x * 100}%`,
                                top: `${detection.correction.boundingBox.y * 100}%`,
                                width: `${detection.correction.boundingBox.width * 100}%`,
                                height: `${detection.correction.boundingBox.height * 100}%`,
                            }}
                        >
                            <div className="absolute -top-6 left-0 text-white text-xs px-1.5 py-0.5 rounded bg-info">
                                Human Labeled
                            </div>
                        </div>
                    )}
                    {/* Preview Detection bounding box (dashed, warning color) */}
                    {previewDetection?.boundingBox && (
                        <div
                            className="absolute rounded-sm pointer-events-none border-2 border-warning border-dashed"
                            style={{
                                left: `${previewDetection.boundingBox.x * 100}%`,
                                top: `${previewDetection.boundingBox.y * 100}%`,
                                width: `${previewDetection.boundingBox.width * 100}%`,
                                height: `${previewDetection.boundingBox.height * 100}%`,
                            }}
                        >
                            <div className="absolute -top-6 left-0 text-white text-xs px-1.5 py-0.5 rounded bg-warning">
                                Preview
                            </div>
                        </div>
                    )}
                </div>

                {/* Run Detection Controls */}
                {trainingRuns.length > 0 && (
                    <div className="bg-surface rounded-lg border border-border p-4">
                        <h2 className="text-sm font-medium text-foreground mb-3">Run Detection</h2>
                        <div className="flex items-center gap-3">
                            <Select
                                value={selectedModelId}
                                onChange={(value) => setSelectedModelId(String(value))}
                                placeholder="Select a model..."
                                options={trainingRuns.map((run) => ({
                                    value: run.id,
                                    label: `${run.id.slice(0, 8)} - ${run.imageCount} images${run.accuracy !== null ? ` - ${(run.accuracy * 100).toFixed(1)}% acc` : ''}`,
                                }))}
                                className="flex-1"
                            />
                            <Button
                                onClick={runDetection}
                                disabled={isDetecting || !selectedModelId}
                                leftIcon={<Play className="w-4 h-4" />}
                            >
                                {isDetecting ? 'Running...' : 'Detect'}
                            </Button>
                        </div>

                        {/* Preview Detection Result */}
                        {previewDetection && (
                            <div className="mt-4 pt-4 border-t border-border space-y-3">
                                <div className="flex items-center justify-between gap-4">
                                    <span className="text-sm text-secondary">Board detected</span>
                                    <span className={`px-2 py-0.5 text-xs font-medium rounded ${
                                        previewDetection.hasBoardDetected
                                            ? 'bg-success/20 text-success'
                                            : 'bg-error/20 text-error'
                                    }`}>
                                        {previewDetection.hasBoardDetected ? 'Yes' : 'No'}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between gap-4">
                                    <span className="text-sm text-secondary">Entire board visible</span>
                                    <span className={`px-2 py-0.5 text-xs font-medium rounded ${
                                        previewDetection.isEntireBoardVisible
                                            ? 'bg-success/20 text-success'
                                            : 'bg-error/20 text-error'
                                    }`}>
                                        {previewDetection.isEntireBoardVisible ? 'Yes' : 'No'}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between gap-4">
                                    <span className="text-sm text-secondary">Confidence</span>
                                    <span className="text-sm font-mono text-foreground">
                                        {(previewDetection.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                                {previewDetection.boundingBox && (
                                    <div className="pt-2 border-t border-border">
                                        <span className="text-xs text-muted">Bounding box</span>
                                        <p className="text-xs font-mono text-foreground mt-1">
                                            {formatBbox(previewDetection.boundingBox)}
                                        </p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}

                {/* AI Detection Info */}
                {detection && detection.detection && detection.detection.confidence > 0 && (
                    <div className="bg-surface rounded-lg border border-border p-4">
                        <h2 className="text-sm font-medium text-foreground mb-3">AI Detection</h2>
                        <div className="space-y-3">
                            {/* Board detected */}
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-sm text-secondary">Board detected</span>
                                <span className={`px-2 py-0.5 text-xs font-medium rounded ${
                                    detection.detection.hasBoardDetected
                                        ? 'bg-success/20 text-success'
                                        : 'bg-error/20 text-error'
                                }`}>
                                    {detection.detection.hasBoardDetected ? 'Yes' : 'No'}
                                </span>
                            </div>

                            {/* Entire board visible */}
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-sm text-secondary">Entire board visible</span>
                                <span className={`px-2 py-0.5 text-xs font-medium rounded ${
                                    detection.detection.isEntireBoardVisible
                                        ? 'bg-success/20 text-success'
                                        : 'bg-error/20 text-error'
                                }`}>
                                    {detection.detection.isEntireBoardVisible ? 'Yes' : 'No'}
                                </span>
                            </div>

                            {/* Confidence */}
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-sm text-secondary">Confidence</span>
                                <span className="text-sm font-mono text-foreground">
                                    {(detection.detection.confidence * 100).toFixed(0)}%
                                </span>
                            </div>

                            {/* Classifier Confidence */}
                            {detection.detection.classifierConfidence !== undefined && detection.detection.classifierConfidence !== null && (
                                <div className="flex items-center justify-between gap-4">
                                    <span className="text-sm text-secondary">Classifier confidence</span>
                                    <span className="text-sm font-mono text-foreground">
                                        {(detection.detection.classifierConfidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                            )}

                            {/* IoU vs Ground Truth */}
                            {iou !== null && (
                                <div className="flex items-center justify-between gap-4">
                                    <span className="text-sm text-secondary">IoU vs Ground Truth</span>
                                    <span className={`text-sm font-mono ${
                                        iou >= 0.75 ? 'text-success' : iou >= 0.5 ? 'text-warning' : 'text-error'
                                    }`}>
                                        {(iou * 100).toFixed(1)}%
                                    </span>
                                </div>
                            )}

                            {/* Bounding box coordinates */}
                            {detection.detection.boundingBox && (
                                <div className="pt-2 border-t border-border">
                                    <span className="text-xs text-muted">Bounding box</span>
                                    <p className="text-xs font-mono text-foreground mt-1">
                                        {formatBbox(detection.detection.boundingBox)}
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Human Label Info */}
                {detection && detection.correction && (
                    <div className="bg-surface rounded-lg border border-info/50 p-4">
                        <h2 className="text-sm font-medium text-foreground mb-3">Human Label (Ground Truth)</h2>
                        <div className="space-y-3">
                            {/* Board detected */}
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-sm text-secondary">Board detected</span>
                                <span className={`px-2 py-0.5 text-xs font-medium rounded ${
                                    detection.correction.hasBoardDetected
                                        ? 'bg-success/20 text-success'
                                        : 'bg-error/20 text-error'
                                }`}>
                                    {detection.correction.hasBoardDetected ? 'Yes' : 'No'}
                                </span>
                            </div>

                            {/* Entire board visible */}
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-sm text-secondary">Entire board visible</span>
                                <span className={`px-2 py-0.5 text-xs font-medium rounded ${
                                    detection.correction.isEntireBoardVisible
                                        ? 'bg-success/20 text-success'
                                        : 'bg-error/20 text-error'
                                }`}>
                                    {detection.correction.isEntireBoardVisible ? 'Yes' : 'No'}
                                </span>
                            </div>

                            {/* Bounding box coordinates */}
                            {detection.correction.boundingBox && (
                                <div className="pt-2 border-t border-border">
                                    <span className="text-xs text-muted">Bounding box</span>
                                    <p className="text-xs font-mono text-foreground mt-1">
                                        {formatBbox(detection.correction.boundingBox)}
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
