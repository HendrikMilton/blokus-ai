import { Knex } from 'knex';
import { knex } from '../../knex';
import type { DetectResponse, DetectResponseWithCorrection, SampleStatusFlags, TrainingRun, TrainingRunStatus, BoardDetection, DetectionRun, DetectionRunStatus, SampleAiDetection, SampleHumanLabel } from '@blokus-ai/shared';

interface BoardDetectionRow {
    id: string;
    sample_id: string;
    // AI detection fields
    has_board_detected: boolean;
    is_entire_board_visible: boolean;
    confidence: number;
    classifier_confidence: number | null;
    bbox_x: number | null;
    bbox_y: number | null;
    bbox_width: number | null;
    bbox_height: number | null;
    processing_time_ms: number;
    // Human ground truth fields
    human_has_board: boolean | null;
    human_is_entire_visible: boolean | null;
    human_bbox_x: number | null;
    human_bbox_y: number | null;
    human_bbox_width: number | null;
    human_bbox_height: number | null;
    human_labeled_at: Date | null;
    // Timestamps
    created_at: Date;
    updated_at: Date;
    verified_at: Date | null;
}

const rowToDetectResponse = (row: BoardDetectionRow): DetectResponse => ({
    sample_id: row.sample_id,
    detection: {
        hasBoardDetected: row.has_board_detected,
        isEntireBoardVisible: row.is_entire_board_visible,
        confidence: row.confidence,
        classifierConfidence: row.classifier_confidence,
        boundingBox: row.bbox_x !== null ? {
            x: row.bbox_x,
            y: row.bbox_y!,
            width: row.bbox_width!,
            height: row.bbox_height!,
        } : null,
    },
    processing_time_ms: row.processing_time_ms,
});

const rowToHumanLabel = (row: BoardDetectionRow): BoardDetection | null => {
    // Return human label only if human_labeled_at is set
    if (!row.human_labeled_at) {
        return null;
    }
    return {
        hasBoardDetected: row.human_has_board ?? row.has_board_detected,
        isEntireBoardVisible: row.human_is_entire_visible ?? row.is_entire_board_visible,
        confidence: 1.0, // Human labels have 100% confidence
        boundingBox: row.human_bbox_x !== null ? {
            x: row.human_bbox_x,
            y: row.human_bbox_y!,
            width: row.human_bbox_width!,
            height: row.human_bbox_height!,
        } : null,
    };
};

const rowToDetectResponseWithCorrection = (row: BoardDetectionRow): DetectResponseWithCorrection => ({
    ...rowToDetectResponse(row),
    correction: rowToHumanLabel(row),
    isHumanLabeled: row.human_labeled_at !== null,
});

export const getBoardDetectionBySampleId = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<DetectResponse | null> => {
    const row = await trx<BoardDetectionRow>('board_detections')
        .where('sample_id', sampleId)
        .first();

    return row ? rowToDetectResponse(row) : null;
};

export const getBoardDetectionWithCorrectionBySampleId = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<DetectResponseWithCorrection | null> => {
    const row = await trx<BoardDetectionRow>('board_detections')
        .where('sample_id', sampleId)
        .first();

    return row ? rowToDetectResponseWithCorrection(row) : null;
};

export const upsertBoardDetection = async (
    data: DetectResponse,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    const { detection } = data;

    await trx('board_detections')
        .insert({
            sample_id: data.sample_id,
            has_board_detected: detection.hasBoardDetected,
            is_entire_board_visible: detection.isEntireBoardVisible,
            confidence: detection.confidence,
            classifier_confidence: detection.classifierConfidence ?? null,
            bbox_x: detection.boundingBox?.x ?? null,
            bbox_y: detection.boundingBox?.y ?? null,
            bbox_width: detection.boundingBox?.width ?? null,
            bbox_height: detection.boundingBox?.height ?? null,
            processing_time_ms: data.processing_time_ms,
            updated_at: knex.fn.now(),
        })
        .onConflict('sample_id')
        .merge();
};

export const insertBoardDetection = async (
    data: DetectResponse,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    const { detection } = data;

    await trx('board_detections').insert({
        sample_id: data.sample_id,
        has_board_detected: detection.hasBoardDetected,
        is_entire_board_visible: detection.isEntireBoardVisible,
        confidence: detection.confidence,
        classifier_confidence: detection.classifierConfidence ?? null,
        bbox_x: detection.boundingBox?.x ?? null,
        bbox_y: detection.boundingBox?.y ?? null,
        bbox_width: detection.boundingBox?.width ?? null,
        bbox_height: detection.boundingBox?.height ?? null,
        processing_time_ms: data.processing_time_ms,
    });
};

export const getAiLabeledSampleIds = async (
    trx: Knex.Transaction | Knex = knex
): Promise<Set<string>> => {
    const rows = await trx('board_detections')
        .whereNull('human_labeled_at')
        .select('sample_id');
    return new Set(rows.map(r => r.sample_id));
};

export const getHumanLabeledSampleIds = async (
    trx: Knex.Transaction | Knex = knex
): Promise<Set<string>> => {
    const rows = await trx('board_detections')
        .whereNotNull('human_labeled_at')
        .select('sample_id');
    return new Set(rows.map(r => r.sample_id));
};

export const deleteBoardDetection = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('board_detections')
        .where('sample_id', sampleId)
        .delete();
    return deleted > 0;
};

export const deleteBoardDetections = async (
    sampleIds: string[],
    trx: Knex.Transaction | Knex = knex
): Promise<number> => {
    if (sampleIds.length === 0) {
        return 0;
    }
    const deleted = await trx('board_detections')
        .whereIn('sample_id', sampleIds)
        .delete();
    return deleted;
};

export interface CorrectionData {
    hasBoardDetected: boolean;
    isEntireBoardVisible: boolean;
    boundingBox: { x: number; y: number; width: number; height: number } | null;
}

export const saveBoardDetectionCorrection = async (
    sampleId: string,
    correction: CorrectionData,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const updated = await trx('board_detections')
        .where('sample_id', sampleId)
        .update({
            human_has_board: correction.hasBoardDetected,
            human_is_entire_visible: correction.isEntireBoardVisible,
            human_bbox_x: correction.boundingBox?.x ?? null,
            human_bbox_y: correction.boundingBox?.y ?? null,
            human_bbox_width: correction.boundingBox?.width ?? null,
            human_bbox_height: correction.boundingBox?.height ?? null,
            human_labeled_at: knex.fn.now(),
            updated_at: knex.fn.now(),
        });
    return updated > 0;
};

export const getVerifiedSampleIds = async (
    trx: Knex.Transaction | Knex = knex
): Promise<Set<string>> => {
    const rows = await trx('board_detections')
        .whereNotNull('verified_at')
        .select('sample_id');
    return new Set(rows.map(r => r.sample_id));
};

export const isDetectionVerified = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const row = await trx('board_detections')
        .where('sample_id', sampleId)
        .whereNotNull('verified_at')
        .first();
    return !!row;
};

export const setVerificationStatus = async (
    sampleId: string,
    verified: boolean,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const updated = await trx('board_detections')
        .where('sample_id', sampleId)
        .update({
            verified_at: verified ? knex.fn.now() : null,
            updated_at: knex.fn.now(),
        });
    return updated > 0;
};

export interface ManualLabelData {
    hasBoardDetected: boolean;
    isEntireBoardVisible: boolean;
    boundingBox: { x: number; y: number; width: number; height: number } | null;
}

export const saveManualLabel = async (
    sampleId: string,
    label: ManualLabelData,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    // Check if record exists
    const existing = await trx('board_detections')
        .where('sample_id', sampleId)
        .first();

    if (existing) {
        // Update only the human/ground truth fields, preserve AI detection fields
        await trx('board_detections')
            .where('sample_id', sampleId)
            .update({
                human_has_board: label.hasBoardDetected,
                human_is_entire_visible: label.isEntireBoardVisible,
                human_bbox_x: label.boundingBox?.x ?? null,
                human_bbox_y: label.boundingBox?.y ?? null,
                human_bbox_width: label.boundingBox?.width ?? null,
                human_bbox_height: label.boundingBox?.height ?? null,
                human_labeled_at: knex.fn.now(),
                updated_at: knex.fn.now(),
            });
    } else {
        // No AI detection exists yet - create record with only ground truth fields
        // AI fields are null (will be set when AI runs)
        await trx('board_detections').insert({
            sample_id: sampleId,
            // No AI detection yet (null = never ran)
            has_board_detected: false,
            is_entire_board_visible: false,
            confidence: null,
            processing_time_ms: 0,
            // Human ground truth
            human_has_board: label.hasBoardDetected,
            human_is_entire_visible: label.isEntireBoardVisible,
            human_bbox_x: label.boundingBox?.x ?? null,
            human_bbox_y: label.boundingBox?.y ?? null,
            human_bbox_width: label.boundingBox?.width ?? null,
            human_bbox_height: label.boundingBox?.height ?? null,
            human_labeled_at: knex.fn.now(),
            updated_at: knex.fn.now(),
        });
    }
};

export interface SampleStatusRow {
    sample_id: string;
    confidence: number;
    human_labeled_at: Date | null;
    verified_at: Date | null;
}

export const getSampleStatusFlags = async (
    trx: Knex.Transaction | Knex = knex
): Promise<Map<string, SampleStatusFlags>> => {
    const rows = await trx<SampleStatusRow>('board_detections')
        .select('sample_id', 'confidence', 'human_labeled_at', 'verified_at');

    const result = new Map<string, SampleStatusFlags>();
    for (const row of rows) {
        result.set(row.sample_id, {
            aiLabeled: row.confidence !== null, // null = never ran, 0 = ran but no board, >0 = ran and found board
            humanLabeled: row.human_labeled_at !== null,
            verified: row.verified_at !== null,
        });
    }
    return result;
};

export const getSampleStatusFlagsById = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<SampleStatusFlags | null> => {
    const row = await trx<SampleStatusRow>('board_detections')
        .where('sample_id', sampleId)
        .select('sample_id', 'confidence', 'human_labeled_at', 'verified_at')
        .first();

    if (!row) {
        return null;
    }

    return {
        aiLabeled: row.confidence !== null, // null = never ran, 0 = ran but no board, >0 = ran and found board
        humanLabeled: row.human_labeled_at !== null,
        verified: row.verified_at !== null,
    };
};

export interface SampleDetectionInfo {
    status: SampleStatusFlags;
    ai: SampleAiDetection | null;
    human: SampleHumanLabel | null;
    iou: number | null;
}

function calculateIoU(
    aiX: number, aiY: number, aiW: number, aiH: number,
    hX: number, hY: number, hW: number, hH: number
): number {
    const interX1 = Math.max(aiX, hX);
    const interY1 = Math.max(aiY, hY);
    const interX2 = Math.min(aiX + aiW, hX + hW);
    const interY2 = Math.min(aiY + aiH, hY + hH);

    const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
    const unionArea = (aiW * aiH) + (hW * hH) - interArea;

    return unionArea === 0 ? 0 : interArea / unionArea;
}

export const getSampleDetectionInfoMap = async (
    trx: Knex.Transaction | Knex = knex
): Promise<Map<string, SampleDetectionInfo>> => {
    const rows = await trx('board_detections')
        .select(
            'sample_id',
            'has_board_detected',
            'is_entire_board_visible',
            'confidence',
            'classifier_confidence',
            'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
            'human_has_board',
            'human_is_entire_visible',
            'human_bbox_x', 'human_bbox_y', 'human_bbox_width', 'human_bbox_height',
            'human_labeled_at',
            'verified_at'
        );

    const result = new Map<string, SampleDetectionInfo>();

    for (const row of rows) {
        // AI detection: null = never ran, 0 = ran but no board, >0 = ran and found board
        const hasAiDetection = row.confidence !== null;
        const hasHumanLabel = row.human_labeled_at !== null;

        // Calculate IoU if both AI and human have bounding boxes
        let iou: number | null = null;
        if (
            hasAiDetection &&
            hasHumanLabel &&
            row.bbox_x !== null &&
            row.human_bbox_x !== null
        ) {
            iou = calculateIoU(
                row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height,
                row.human_bbox_x, row.human_bbox_y, row.human_bbox_width, row.human_bbox_height
            );
        }

        result.set(row.sample_id, {
            status: {
                aiLabeled: hasAiDetection,
                humanLabeled: hasHumanLabel,
                verified: row.verified_at !== null,
            },
            ai: hasAiDetection ? {
                hasBoardDetected: row.has_board_detected,
                isEntireBoardVisible: row.is_entire_board_visible,
                confidence: row.confidence,
                classifierConfidence: row.classifier_confidence,
                boundingBox: row.bbox_x !== null ? {
                    x: row.bbox_x,
                    y: row.bbox_y,
                    width: row.bbox_width,
                    height: row.bbox_height,
                } : null,
            } : null,
            human: hasHumanLabel ? {
                hasBoardDetected: row.human_has_board ?? false,
                isEntireBoardVisible: row.human_is_entire_visible ?? false,
                boundingBox: row.human_bbox_x !== null ? {
                    x: row.human_bbox_x,
                    y: row.human_bbox_y,
                    width: row.human_bbox_width,
                    height: row.human_bbox_height,
                } : null,
            } : null,
            iou,
        });
    }

    return result;
};

/**
 * Calculate the average IoU across all samples that have both AI and human bounding boxes.
 */
export const calculateAverageIoU = async (
    trx: Knex.Transaction | Knex = knex
): Promise<number | null> => {
    const rows = await trx('board_detections')
        .select(
            'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
            'human_bbox_x', 'human_bbox_y', 'human_bbox_width', 'human_bbox_height'
        )
        .whereNotNull('bbox_x')
        .whereNotNull('human_bbox_x');

    if (rows.length === 0) {
        return null;
    }

    let totalIoU = 0;
    for (const row of rows) {
        totalIoU += calculateIoU(
            row.bbox_x, row.bbox_y, row.bbox_width, row.bbox_height,
            row.human_bbox_x, row.human_bbox_y, row.human_bbox_width, row.human_bbox_height
        );
    }

    return totalIoU / rows.length;
};

// ============================================================================
// Training Runs
// ============================================================================

interface TrainingRunRow {
    id: string;
    status: TrainingRunStatus;
    image_count: number;
    started_at: Date | null;
    completed_at: Date | null;
    duration_ms: number | null;
    accuracy: number | null;
    loss: number | null;
    val_accuracy: number | null;
    val_loss: number | null;
    classifier_accuracy: number | null;
    classifier_loss: number | null;
    error_message: string | null;
    created_at: Date;
    updated_at: Date;
}

const rowToTrainingRun = (row: TrainingRunRow): TrainingRun => ({
    id: row.id,
    status: row.status,
    imageCount: row.image_count,
    startedAt: row.started_at?.toISOString() ?? null,
    completedAt: row.completed_at?.toISOString() ?? null,
    durationMs: row.duration_ms,
    accuracy: row.accuracy,
    loss: row.loss,
    valAccuracy: row.val_accuracy,
    valLoss: row.val_loss,
    classifierAccuracy: row.classifier_accuracy,
    classifierLoss: row.classifier_loss,
    errorMessage: row.error_message,
    createdAt: row.created_at.toISOString(),
});

export const createTrainingRun = async (
    imageCount: number,
    trx: Knex.Transaction | Knex = knex
): Promise<TrainingRun> => {
    const [row] = await trx<TrainingRunRow>('training_runs')
        .insert({ image_count: imageCount })
        .returning('*');
    return rowToTrainingRun(row);
};

export interface TrainingRunUpdate {
    status?: TrainingRunStatus;
    startedAt?: Date;
    completedAt?: Date;
    durationMs?: number;
    accuracy?: number;
    loss?: number;
    valAccuracy?: number;
    valLoss?: number;
    classifierAccuracy?: number;
    classifierLoss?: number;
    errorMessage?: string;
}

export const updateTrainingRun = async (
    id: string,
    updates: TrainingRunUpdate,
    trx: Knex.Transaction | Knex = knex
): Promise<TrainingRun | null> => {
    const dbUpdates: Partial<TrainingRunRow> = {
        updated_at: new Date(),
    };

    if (updates.status !== undefined) {
        dbUpdates.status = updates.status;
    }
    if (updates.startedAt !== undefined) {
        dbUpdates.started_at = updates.startedAt;
    }
    if (updates.completedAt !== undefined) {
        dbUpdates.completed_at = updates.completedAt;
    }
    if (updates.durationMs !== undefined) {
        dbUpdates.duration_ms = updates.durationMs;
    }
    if (updates.accuracy !== undefined) {
        dbUpdates.accuracy = updates.accuracy;
    }
    if (updates.loss !== undefined) {
        dbUpdates.loss = updates.loss;
    }
    if (updates.valAccuracy !== undefined) {
        dbUpdates.val_accuracy = updates.valAccuracy;
    }
    if (updates.valLoss !== undefined) {
        dbUpdates.val_loss = updates.valLoss;
    }
    if (updates.classifierAccuracy !== undefined) {
        dbUpdates.classifier_accuracy = updates.classifierAccuracy;
    }
    if (updates.classifierLoss !== undefined) {
        dbUpdates.classifier_loss = updates.classifierLoss;
    }
    if (updates.errorMessage !== undefined) {
        dbUpdates.error_message = updates.errorMessage;
    }

    const [row] = await trx<TrainingRunRow>('training_runs')
        .where('id', id)
        .update(dbUpdates)
        .returning('*');

    return row ? rowToTrainingRun(row) : null;
};

export const listTrainingRuns = async (
    trx: Knex.Transaction | Knex = knex
): Promise<TrainingRun[]> => {
    const rows = await trx<TrainingRunRow>('training_runs')
        .orderBy('created_at', 'desc');
    return rows.map(rowToTrainingRun);
};

export const getTrainingRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<TrainingRun | null> => {
    const row = await trx<TrainingRunRow>('training_runs')
        .where('id', id)
        .first();
    return row ? rowToTrainingRun(row) : null;
};

export const deleteTrainingRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('training_runs')
        .where('id', id)
        .delete();
    return deleted > 0;
};

export const getVerifiedImageCount = async (
    trx: Knex.Transaction | Knex = knex
): Promise<number> => {
    const result = await trx('board_detections')
        .whereNotNull('verified_at')
        .count('* as count')
        .first();
    return Number(result?.count ?? 0);
};

export const getHumanLabeledImageCount = async (
    trx: Knex.Transaction | Knex = knex
): Promise<number> => {
    const result = await trx('board_detections')
        .whereNotNull('human_labeled_at')
        .count('* as count')
        .first();
    return Number(result?.count ?? 0);
};

export interface TrainingDataSample {
    sample_id: string;
    image_url: string;
    bbox: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
}

export const getTrainingData = async (
    trx: Knex.Transaction | Knex = knex
): Promise<TrainingDataSample[]> => {
    // Get human-labeled samples with bounding boxes (ground truth for training)
    const rows = await trx<BoardDetectionRow>('board_detections')
        .whereNotNull('human_labeled_at')
        .select(
            'sample_id',
            'human_bbox_x',
            'human_bbox_y',
            'human_bbox_width',
            'human_bbox_height'
        );

    const cloudFrontDomain = 'd26azxov2byjpl.cloudfront.net';

    return rows
        .filter(row => {
            // Must have human bbox
            return row.human_bbox_x !== null;
        })
        .map(row => {
            return {
                sample_id: row.sample_id,
                image_url: `https://${cloudFrontDomain}/thumbnails/${row.sample_id}.jpg`,
                bbox: {
                    x: row.human_bbox_x!,
                    y: row.human_bbox_y!,
                    width: row.human_bbox_width!,
                    height: row.human_bbox_height!,
                },
            };
        });
};

export interface ClassifierTrainingSample {
    sample_id: string;
    image_url: string;
    bbox: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
    isEntireBoardVisible: boolean;
}

export const getClassifierTrainingData = async (
    trx: Knex.Transaction | Knex = knex
): Promise<ClassifierTrainingSample[]> => {
    // Get human-labeled samples for classifier training (ground truth)
    const rows = await trx<BoardDetectionRow>('board_detections')
        .whereNotNull('human_labeled_at')
        .select(
            'sample_id',
            'human_has_board',
            'human_is_entire_visible',
            'human_bbox_x',
            'human_bbox_y',
            'human_bbox_width',
            'human_bbox_height'
        );

    const cloudFrontDomain = 'd26azxov2byjpl.cloudfront.net';

    return rows
        .filter(row => {
            // Must have board detected and bbox in human labels
            return row.human_has_board && row.human_bbox_x !== null;
        })
        .map(row => {
            return {
                sample_id: row.sample_id,
                image_url: `https://${cloudFrontDomain}/thumbnails/${row.sample_id}.jpg`,
                bbox: {
                    x: row.human_bbox_x!,
                    y: row.human_bbox_y!,
                    width: row.human_bbox_width!,
                    height: row.human_bbox_height!,
                },
                isEntireBoardVisible: row.human_is_entire_visible ?? true,
            };
        });
};

// ============================================================================
// Detection Runs
// ============================================================================

interface DetectionRunRow {
    id: string;
    status: DetectionRunStatus;
    model_id: string;
    total_count: number;
    processed_count: number;
    success_count: number | null;
    failed_count: number | null;
    started_at: Date | null;
    completed_at: Date | null;
    duration_ms: number | null;
    error_message: string | null;
    avg_iou: number | null;
    created_at: Date;
    updated_at: Date;
}

const rowToDetectionRun = (row: DetectionRunRow): DetectionRun => ({
    id: row.id,
    status: row.status,
    modelId: row.model_id,
    totalCount: row.total_count,
    processedCount: row.processed_count,
    successCount: row.success_count,
    failedCount: row.failed_count,
    startedAt: row.started_at?.toISOString() ?? null,
    completedAt: row.completed_at?.toISOString() ?? null,
    durationMs: row.duration_ms,
    errorMessage: row.error_message,
    avgIou: row.avg_iou,
    createdAt: row.created_at.toISOString(),
});

export const createDetectionRun = async (
    modelId: string,
    totalCount: number,
    trx: Knex.Transaction | Knex = knex
): Promise<DetectionRun> => {
    const [row] = await trx<DetectionRunRow>('detection_runs')
        .insert({ model_id: modelId, total_count: totalCount })
        .returning('*');
    return rowToDetectionRun(row);
};

export interface DetectionRunUpdate {
    status?: DetectionRunStatus;
    processedCount?: number;
    successCount?: number;
    failedCount?: number;
    startedAt?: Date;
    completedAt?: Date;
    durationMs?: number;
    errorMessage?: string;
    avgIou?: number;
}

export const updateDetectionRun = async (
    id: string,
    updates: DetectionRunUpdate,
    trx: Knex.Transaction | Knex = knex
): Promise<DetectionRun | null> => {
    const dbUpdates: Partial<DetectionRunRow> = {
        updated_at: new Date(),
    };

    if (updates.status !== undefined) {
        dbUpdates.status = updates.status;
    }
    if (updates.processedCount !== undefined) {
        dbUpdates.processed_count = updates.processedCount;
    }
    if (updates.successCount !== undefined) {
        dbUpdates.success_count = updates.successCount;
    }
    if (updates.failedCount !== undefined) {
        dbUpdates.failed_count = updates.failedCount;
    }
    if (updates.startedAt !== undefined) {
        dbUpdates.started_at = updates.startedAt;
    }
    if (updates.completedAt !== undefined) {
        dbUpdates.completed_at = updates.completedAt;
    }
    if (updates.durationMs !== undefined) {
        dbUpdates.duration_ms = updates.durationMs;
    }
    if (updates.errorMessage !== undefined) {
        dbUpdates.error_message = updates.errorMessage;
    }
    if (updates.avgIou !== undefined) {
        dbUpdates.avg_iou = updates.avgIou;
    }

    const [row] = await trx<DetectionRunRow>('detection_runs')
        .where('id', id)
        .update(dbUpdates)
        .returning('*');

    return row ? rowToDetectionRun(row) : null;
};

export const listDetectionRuns = async (
    trx: Knex.Transaction | Knex = knex
): Promise<DetectionRun[]> => {
    const rows = await trx<DetectionRunRow>('detection_runs')
        .orderBy('created_at', 'desc');
    return rows.map(rowToDetectionRun);
};

export const getDetectionRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<DetectionRun | null> => {
    const row = await trx<DetectionRunRow>('detection_runs')
        .where('id', id)
        .first();
    return row ? rowToDetectionRun(row) : null;
};

export const deleteDetectionRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('detection_runs')
        .where('id', id)
        .delete();
    return deleted > 0;
};

/**
 * Clear AI labels from all board_detections.
 * Resets AI detection fields to placeholder values while preserving human labels.
 */
export const clearAllAiLabels = async (
    trx: Knex.Transaction | Knex = knex
): Promise<number> => {
    const updated = await trx('board_detections')
        .whereNotNull('confidence')
        .update({
            has_board_detected: false,
            is_entire_board_visible: false,
            confidence: null,
            classifier_confidence: null,
            bbox_x: null,
            bbox_y: null,
            bbox_width: null,
            bbox_height: null,
            processing_time_ms: 0,
            updated_at: knex.fn.now(),
        });
    return updated;
};

export interface BulkDetectionFilterResult {
    sampleIds: string[];
    existingDetectionCount: number;
}

/**
 * Get sample IDs that should be processed in bulk detection.
 * AI detection can run on ANY sample - it only updates AI fields, never touches human ground truth.
 * The checkbox controls whether to re-run on samples that already have AI detections.
 */
export const getSampleIdsForBulkDetection = async (
    allSampleIds: string[],
    includeExistingDetections: boolean,
    trx: Knex.Transaction | Knex = knex
): Promise<BulkDetectionFilterResult> => {
    // Get samples that already have AI detection (confidence is not null)
    const rows = await trx('board_detections')
        .whereNotNull('confidence')
        .select('sample_id');
    const existingDetectionIds = new Set(rows.map(r => r.sample_id));
    const existingDetectionCount = allSampleIds.filter(id => existingDetectionIds.has(id)).length;

    let eligibleSampleIds = allSampleIds;

    if (!includeExistingDetections) {
        // Exclude samples that already have AI detections
        eligibleSampleIds = allSampleIds.filter(id => !existingDetectionIds.has(id));
    }

    return {
        sampleIds: eligibleSampleIds,
        existingDetectionCount,
    };
};
