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

export type SampleStatus = 'unlabeled' | 'labeled' | 'verified';
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
    status: SampleStatus;
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
    status?: SampleStatus;
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
    labeled_samples: number;
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
}

export interface DetectResponse {
    sample_id: string;
    detection: BoardDetection;
    processing_time_ms: number;
}
