// Board and game types
export type PlayerColor = 'blue' | 'yellow' | 'red' | 'green';

export interface BoardCell {
    color: PlayerColor | null;
}

// Grid cell color for labeling (null = empty)
export type CellColor = PlayerColor | null;

// 20x20 grid of cell colors
export type GridLabels = CellColor[][];

// Grid label record from database
export interface GridLabel {
    sampleId: string;
    grid: GridLabels;
    labeledAt: string;
}

// Constants
export const BOARD_SIZE = 20;

export const PLAYER_COLORS: Record<PlayerColor, string> = {
    blue: '#0066CC',
    yellow: '#FFD700',
    red: '#DC143C',
    green: '#228B22',
};

// ============================================================================
// Admin types
// ============================================================================

// Sample can have multiple status flags (not mutually exclusive)
export interface SampleStatusFlags {
    aiLabeled: boolean;
    humanLabeled: boolean;
    verified: boolean;
}

// For display purposes - derived from flags (shows highest priority status)
export type SampleStatusDisplay = 'unlabeled' | 'ai-labeled' | 'human-labeled' | 'verified';

// Get the primary display status from flags (priority: verified > human-labeled > ai-labeled > unlabeled)
export const getDisplayStatus = (flags: SampleStatusFlags): SampleStatusDisplay => {
    if (flags.verified) return 'verified';
    if (flags.humanLabeled) return 'human-labeled';
    if (flags.aiLabeled) return 'ai-labeled';
    return 'unlabeled';
};

// Get all active status labels for display
export const getStatusLabels = (flags: SampleStatusFlags): SampleStatusDisplay[] => {
    const labels: SampleStatusDisplay[] = [];
    if (flags.aiLabeled) labels.push('ai-labeled');
    if (flags.humanLabeled) labels.push('human-labeled');
    if (flags.verified) labels.push('verified');
    if (labels.length === 0) labels.push('unlabeled');
    return labels;
};

// ============================================================================
// Board Detection types
// ============================================================================

export type DetectorModel = 'yolo';

export const DETECTOR_MODELS: { value: DetectorModel; label: string }[] = [
    { value: 'yolo', label: 'YOLO 26' },
];

export interface BoundingBox {
    x: number;      // left edge (0-1)
    y: number;      // top edge (0-1)
    width: number;  // (0-1)
    height: number; // (0-1)
}

export interface BoardDetection {
    hasBoardDetected: boolean;
    isEntireBoardVisible: boolean;
    boundingBox: BoundingBox | null;
    confidence: number;
    classifierConfidence?: number | null;
    corners?: BoardCorners | null;
    cornerConfidence?: number | null;
}

export interface DetectResponse {
    sample_id: string;
    detection: BoardDetection;
    processing_time_ms: number;
}

export interface DetectResponseWithCorrection extends DetectResponse {
    correction: BoardDetection | null;
    isHumanLabeled: boolean;
}

// ============================================================================
// Training Run types
// ============================================================================

export type TrainingRunStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface TrainingRun {
    id: string;
    status: TrainingRunStatus;
    imageCount: number;
    startedAt: string | null;
    completedAt: string | null;
    durationMs: number | null;
    accuracy: number | null;
    loss: number | null;
    valAccuracy: number | null;
    valLoss: number | null;
    classifierAccuracy: number | null;
    classifierLoss: number | null;
    poseAccuracy: number | null;
    poseLoss: number | null;
    poseError: number | null;
    poseImageCount: number | null;
    errorMessage: string | null;
    createdAt: string;
}

// ============================================================================
// Detection Run types
// ============================================================================

export type DetectionRunStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface DetectionRun {
    id: string;
    status: DetectionRunStatus;
    modelId: string;
    pipelineId: string | null;
    totalCount: number;
    processedCount: number;
    successCount: number | null;
    failedCount: number | null;
    disagreementCount: number | null;
    startedAt: string | null;
    completedAt: string | null;
    durationMs: number | null;
    errorMessage: string | null;
    avgIou: number | null;
    avgClassifierAccuracy: number | null;
    avgCornerError: number | null;
    cornerAccuracy: number | null;
    createdAt: string;
}

// ============================================================================
// Sample types (for admin images list)
// ============================================================================

export interface SampleAiDetection {
    hasBoardDetected: boolean;
    isEntireBoardVisible: boolean;
    confidence: number;
    classifierConfidence: number | null;
    boundingBox: BoundingBox | null;
    corners: BoardCorners | null;
    cornerError: number | null;
}

export interface BoardCorners {
    topLeft: { x: number; y: number };
    topRight: { x: number; y: number };
    bottomRight: { x: number; y: number };
    bottomLeft: { x: number; y: number };
}

export interface SampleHumanLabel {
    hasBoardDetected: boolean;
    isEntireBoardVisible: boolean;
    boundingBox: BoundingBox | null;
    corners: BoardCorners | null;
}

export interface Sample {
    id: string;
    filename: string;
    status: SampleStatusFlags;
    ai: SampleAiDetection | null;
    human: SampleHumanLabel | null;
    iou: number | null;
    created_at: string;
    imageUrl?: string;
    thumbnailUrl?: string;
    tags?: Tag[];
}

// ============================================================================
// Model Pipeline types
// ============================================================================

export interface ModelPipeline {
    id: string;
    name: string;
    componentIds: string[];
    components?: TrainingRun[];  // Populated when fetching with details
    createdAt: string;
    updatedAt: string;
}

// ============================================================================
// Tag types
// ============================================================================

export interface Tag {
    id: string;
    name: string;
    createdAt: string;
}

// ============================================================================
// Corner Detection Run types
// ============================================================================

export type CornerDetectionRunStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface CornerDetectionRun {
    id: string;
    status: CornerDetectionRunStatus;
    totalCount: number;
    processedCount: number;
    successCount: number | null;
    failedCount: number | null;
    startedAt: string | null;
    completedAt: string | null;
    durationMs: number | null;
    errorMessage: string | null;
    createdAt: string;
}

// Quality metrics for a corner detection result
export interface CornerQualityMetrics {
    aspectRatio: number | null;  // width/height, 1.0 = perfect square
    areaRatio: number | null;    // detected area / bbox area
    avgAngle: number | null;     // average corner angle in degrees, 90 = perfect
}

// Single method result with quality metrics
export interface CornerMethodResult {
    success: boolean;
    corners: BoardCorners | null;
    error: number | null;
    quality: CornerQualityMetrics | null;
    croppedUrl?: string | null;
}

// Corner detection result for a single sample (3 methods: best, hough, edge)
// "best" automatically selects between hough and edge based on quality metrics
export interface CornerDetectionResult {
    sampleId: string;
    runId: string;
    best: CornerMethodResult;
    hough: CornerMethodResult;
    edge: CornerMethodResult;
}

// Perspective correction from saved corners
export interface PerspectiveFromCornersRequest {
    best?: BoardCorners | null;
    hough?: BoardCorners | null;
    edge?: BoardCorners | null;
}

export interface PerspectiveMethodResult {
    method: string;
    image: string | null;  // base64 encoded
    success: boolean;
}

export interface PerspectiveFromCornersResponse {
    best: PerspectiveMethodResult;
    hough: PerspectiveMethodResult;
    edge: PerspectiveMethodResult;
}
