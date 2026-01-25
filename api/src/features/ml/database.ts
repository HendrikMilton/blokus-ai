import { Knex } from 'knex';
import { knex } from '../../knex';
import type { DetectResponse, DetectResponseWithCorrection, SampleStatusFlags, TrainingRun, TrainingRunStatus, BoardDetection } from '@blokus-ai/shared';

interface BoardDetectionRow {
    id: string;
    sample_id: string;
    has_board_detected: boolean;
    is_entire_board_visible: boolean;
    confidence: number;
    bbox_x: number | null;
    bbox_y: number | null;
    bbox_width: number | null;
    bbox_height: number | null;
    processing_time_ms: number;
    // Corrected fields
    corrected_has_board: boolean | null;
    corrected_is_entire_visible: boolean | null;
    corrected_bbox_x: number | null;
    corrected_bbox_y: number | null;
    corrected_bbox_width: number | null;
    corrected_bbox_height: number | null;
    corrected_at: Date | null;
    human_labeled_at: Date | null;
    created_at: Date;
    updated_at: Date;
}

const rowToDetectResponse = (row: BoardDetectionRow): DetectResponse => ({
    sample_id: row.sample_id,
    detection: {
        hasBoardDetected: row.has_board_detected,
        isEntireBoardVisible: row.is_entire_board_visible,
        confidence: row.confidence,
        boundingBox: row.bbox_x !== null ? {
            x: row.bbox_x,
            y: row.bbox_y!,
            width: row.bbox_width!,
            height: row.bbox_height!,
        } : null,
    },
    processing_time_ms: row.processing_time_ms,
});

const rowToCorrection = (row: BoardDetectionRow): BoardDetection | null => {
    // Return correction only if corrected_at is set
    if (!row.corrected_at) {
        return null;
    }
    return {
        hasBoardDetected: row.corrected_has_board ?? row.has_board_detected,
        isEntireBoardVisible: row.corrected_is_entire_visible ?? row.is_entire_board_visible,
        confidence: 1.0, // Corrections have 100% confidence
        boundingBox: row.corrected_bbox_x !== null ? {
            x: row.corrected_bbox_x,
            y: row.corrected_bbox_y!,
            width: row.corrected_bbox_width!,
            height: row.corrected_bbox_height!,
        } : null,
    };
};

const rowToDetectResponseWithCorrection = (row: BoardDetectionRow): DetectResponseWithCorrection => ({
    ...rowToDetectResponse(row),
    correction: rowToCorrection(row),
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
            corrected_has_board: correction.hasBoardDetected,
            corrected_is_entire_visible: correction.isEntireBoardVisible,
            corrected_bbox_x: correction.boundingBox?.x ?? null,
            corrected_bbox_y: correction.boundingBox?.y ?? null,
            corrected_bbox_width: correction.boundingBox?.width ?? null,
            corrected_bbox_height: correction.boundingBox?.height ?? null,
            corrected_at: knex.fn.now(),
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
    await trx('board_detections')
        .insert({
            sample_id: sampleId,
            // Set as human-labeled values
            has_board_detected: label.hasBoardDetected,
            is_entire_board_visible: label.isEntireBoardVisible,
            confidence: 1.0, // Manual = 100% confidence
            bbox_x: label.boundingBox?.x ?? null,
            bbox_y: label.boundingBox?.y ?? null,
            bbox_width: label.boundingBox?.width ?? null,
            bbox_height: label.boundingBox?.height ?? null,
            processing_time_ms: 0,
            // Also set as corrected since it's human-labeled
            corrected_has_board: label.hasBoardDetected,
            corrected_is_entire_visible: label.isEntireBoardVisible,
            corrected_bbox_x: label.boundingBox?.x ?? null,
            corrected_bbox_y: label.boundingBox?.y ?? null,
            corrected_bbox_width: label.boundingBox?.width ?? null,
            corrected_bbox_height: label.boundingBox?.height ?? null,
            corrected_at: knex.fn.now(),
            human_labeled_at: knex.fn.now(),
            updated_at: knex.fn.now(),
        })
        .onConflict('sample_id')
        .merge();
};

export interface SampleStatusRow {
    sample_id: string;
    human_labeled_at: Date | null;
    verified_at: Date | null;
}

export const getSampleStatusFlags = async (
    trx: Knex.Transaction | Knex = knex
): Promise<Map<string, SampleStatusFlags>> => {
    const rows = await trx<SampleStatusRow>('board_detections')
        .select('sample_id', 'human_labeled_at', 'verified_at');

    const result = new Map<string, SampleStatusFlags>();
    for (const row of rows) {
        result.set(row.sample_id, {
            aiLabeled: row.human_labeled_at === null, // AI labeled if no human label
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
        .select('sample_id', 'human_labeled_at', 'verified_at')
        .first();

    if (!row) {
        return null;
    }

    return {
        aiLabeled: row.human_labeled_at === null,
        humanLabeled: row.human_labeled_at !== null,
        verified: row.verified_at !== null,
    };
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
    // Get human-labeled samples with bounding boxes
    // Use corrected bbox if available, otherwise use original bbox
    const rows = await trx<BoardDetectionRow>('board_detections')
        .whereNotNull('human_labeled_at')
        .select(
            'sample_id',
            'bbox_x',
            'bbox_y',
            'bbox_width',
            'bbox_height',
            'corrected_bbox_x',
            'corrected_bbox_y',
            'corrected_bbox_width',
            'corrected_bbox_height'
        );

    const cloudFrontDomain = 'd26azxov2byjpl.cloudfront.net';

    return rows
        .filter(row => {
            // Must have bbox (either corrected or original)
            const hasCorrectedBbox = row.corrected_bbox_x !== null;
            const hasOriginalBbox = row.bbox_x !== null;
            return hasCorrectedBbox || hasOriginalBbox;
        })
        .map(row => {
            // Prefer corrected bbox over original
            const x = row.corrected_bbox_x ?? row.bbox_x!;
            const y = row.corrected_bbox_y ?? row.bbox_y!;
            const width = row.corrected_bbox_width ?? row.bbox_width!;
            const height = row.corrected_bbox_height ?? row.bbox_height!;

            return {
                sample_id: row.sample_id,
                image_url: `https://${cloudFrontDomain}/thumbnails/${row.sample_id}.jpg`,
                bbox: { x, y, width, height },
            };
        });
};
