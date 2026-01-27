// Board and game types
export type PlayerColor = 'blue' | 'yellow' | 'red' | 'green';

export interface BoardCell {
    color: PlayerColor | null;
}

export interface PieceShape {
    name: string;
    cells: [number, number][];
    rotation: 0 | 90 | 180 | 270;
}

export interface AnalysisStats {
    piecesUsed: number;
    piecesRemaining: number;
    winProbability: number;
}

// API Request/Response types
export interface DetectBoardRequest {
    image: string; // base64 encoded image
}

export interface DetectBoardResponse {
    boardState: BoardCell[][];
}

export interface AnalyzeBoardRequest {
    boardState: BoardCell[][];
    selectedColor: PlayerColor;
}

export interface AnalyzeBoardResponse {
    recommendedPiece: PieceShape | null;  // null if no valid moves
    placement: { row: number; col: number } | null;  // null if no valid moves
    stats: AnalysisStats;
}

// API paths
export const mlDetectBoardPath = '/api/ml/detect-board';
export const mlAnalyzeBoardPath = '/api/ml/analyze';

// Admin API paths
export const mlAdminSamplesPath = '/api/ml/admin/samples';
export const mlAdminStatsPath = '/api/ml/admin/stats';
export const mlAdminTrainPath = '/api/ml/admin/train';

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
export type SampleSource = 'upload' | 'camera' | 'synthetic';

export interface TrainingSample {
    id: string;
    image_path: string;
    image_width: number;
    image_height: number;
    detected_state: BoardCell[][];
    labeled_state: BoardCell[][] | null;
    source: SampleSource;
    notes: string | null;
    status: SampleStatusFlags;
    labeled_at: string | null;
    features_extracted: boolean;
    created_at: string;
    updated_at: string;
}

export interface TrainingSampleCreate {
    image: string; // base64 encoded
    source?: SampleSource;
    notes?: string;
}

export interface TrainingSampleUpdate {
    labeled_state?: BoardCell[][] | null;
    notes?: string | null;
    status?: SampleStatusFlags;
}

export interface TrainingSampleListResponse {
    samples: TrainingSample[];
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
}

export interface AdminStats {
    total_samples: number;
    unlabeled_samples: number;
    ai_labeled_samples: number;
    human_labeled_samples: number;
    verified_samples: number;
    total_cells_labeled: number;
    color_distribution: Record<string, number>;
}

export interface TrainRequest {
    validation_split?: number;
    min_samples?: number;
}

export interface TrainResponse {
    success: boolean;
    message: string;
    samples_used: number;
    validation_accuracy: number | null;
    train_accuracy: number | null;
    color_accuracies: Record<string, number> | null;
    model_path: string | null;
}

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
}

export interface SampleHumanLabel {
    hasBoardDetected: boolean;
    isEntireBoardVisible: boolean;
    boundingBox: BoundingBox | null;
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
