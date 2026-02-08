import { Knex } from 'knex';
import { knex } from '../../knex';
import type { DetectResponse, DetectResponseWithCorrection, SampleStatusFlags, TrainingRun, TrainingRunStatus, BoardDetection, DetectionRun, DetectionRunStatus, SampleAiDetection, SampleHumanLabel, ModelPipeline, BoardCorners, CornerDetectionRun, CornerDetectionRunStatus, CornerDetectionResult, GridLabels, GridLabel, GridPredictionRecord } from '@blokus-ai/shared';

// Shared corner fields used by both board_detections and model_detections
interface CornerFields {
    corner_tl_x: number | null;
    corner_tl_y: number | null;
    corner_tr_x: number | null;
    corner_tr_y: number | null;
    corner_br_x: number | null;
    corner_br_y: number | null;
    corner_bl_x: number | null;
    corner_bl_y: number | null;
}

interface BoardDetectionRow extends CornerFields {
    id: string;
    sample_id: string;
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

interface ModelDetectionRow extends CornerFields {
    id: string;
    sample_id: string;
    model_id: string;
    has_board_detected: boolean;
    is_entire_board_visible: boolean;
    confidence: number | null;
    classifier_confidence: number | null;
    bbox_x: number | null;
    bbox_y: number | null;
    bbox_width: number | null;
    bbox_height: number | null;
    corner_confidence: number | null;
    processing_time_ms: number;
    created_at: Date;
    updated_at: Date;
}

const cornerFieldsToCorners = (row: CornerFields): BoardCorners | null => {
    if (row.corner_tl_x === null) {
        return null;
    }
    return {
        topLeft: { x: row.corner_tl_x, y: row.corner_tl_y! },
        topRight: { x: row.corner_tr_x!, y: row.corner_tr_y! },
        bottomRight: { x: row.corner_br_x!, y: row.corner_br_y! },
        bottomLeft: { x: row.corner_bl_x!, y: row.corner_bl_y! },
    };
};

const modelRowToDetectResponse = (row: ModelDetectionRow): DetectResponse => ({
    sample_id: row.sample_id,
    detection: {
        hasBoardDetected: row.has_board_detected,
        isEntireBoardVisible: row.is_entire_board_visible,
        confidence: row.confidence ?? 0,
        classifierConfidence: row.classifier_confidence,
        boundingBox: row.bbox_x !== null ? {
            x: row.bbox_x,
            y: row.bbox_y!,
            width: row.bbox_width!,
            height: row.bbox_height!,
        } : null,
        corners: cornerFieldsToCorners(row),
        cornerConfidence: row.corner_confidence,
    },
    processing_time_ms: row.processing_time_ms,
});

// Alias for backward compatibility
const rowToCorners = cornerFieldsToCorners;

const rowToHumanLabel = (row: BoardDetectionRow): BoardDetection | null => {
    // Return human label only if human_labeled_at is set
    if (!row.human_labeled_at) {
        return null;
    }
    return {
        hasBoardDetected: row.human_has_board ?? false,
        isEntireBoardVisible: row.human_is_entire_visible ?? false,
        confidence: 1.0, // Human labels have 100% confidence
        boundingBox: row.human_bbox_x !== null ? {
            x: row.human_bbox_x,
            y: row.human_bbox_y!,
            width: row.human_bbox_width!,
            height: row.human_bbox_height!,
        } : null,
    };
};

// ============================================================================
// Model Detections (AI detections stored per model)
// ============================================================================

/**
 * Get the latest completed training run (used as default model).
 */
export const getLatestCompletedModelId = async (
    trx: Knex.Transaction | Knex = knex
): Promise<string | null> => {
    const row = await trx('training_runs')
        .where('status', 'completed')
        .orderBy('completed_at', 'desc')
        .first();
    return row?.id ?? null;
};

/**
 * Get model detection for a specific sample and model.
 */
export const getModelDetection = async (
    sampleId: string,
    modelId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<DetectResponse | null> => {
    const row = await trx<ModelDetectionRow>('model_detections')
        .where('sample_id', sampleId)
        .where('model_id', modelId)
        .first();

    return row ? modelRowToDetectResponse(row) : null;
};

/**
 * Get detection with human correction for a sample.
 * Uses the latest model if no modelId specified.
 */
export const getBoardDetectionWithCorrectionBySampleId = async (
    sampleId: string,
    modelId?: string | null,
    trx: Knex.Transaction | Knex = knex
): Promise<DetectResponseWithCorrection | null> => {
    // Get human label from board_detections
    const humanRow = await trx<BoardDetectionRow>('board_detections')
        .where('sample_id', sampleId)
        .first();

    // Determine which model to use
    const effectiveModelId = modelId ?? await getLatestCompletedModelId(trx);

    // Get AI detection from model_detections
    let aiDetection: DetectResponse | null = null;
    if (effectiveModelId) {
        const modelRow = await trx<ModelDetectionRow>('model_detections')
            .where('sample_id', sampleId)
            .where('model_id', effectiveModelId)
            .first();
        if (modelRow) {
            aiDetection = modelRowToDetectResponse(modelRow);
        }
    }

    if (!aiDetection && !humanRow) {
        return null;
    }

    // Build response combining AI detection and human label
    const humanLabel = humanRow ? rowToHumanLabel(humanRow) : null;

    return {
        sample_id: sampleId,
        detection: aiDetection?.detection ?? {
            hasBoardDetected: false,
            isEntireBoardVisible: false,
            confidence: 0,
            classifierConfidence: null,
            boundingBox: null,
        },
        processing_time_ms: aiDetection?.processing_time_ms ?? 0,
        correction: humanLabel,
        isHumanLabeled: humanRow?.human_labeled_at !== null,
    };
};

/**
 * Upsert a model detection (insert or update on conflict).
 */
export const upsertModelDetection = async (
    data: DetectResponse,
    modelId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    const { detection } = data;

    await trx('model_detections')
        .insert({
            sample_id: data.sample_id,
            model_id: modelId,
            has_board_detected: detection.hasBoardDetected,
            is_entire_board_visible: detection.isEntireBoardVisible,
            confidence: detection.confidence,
            classifier_confidence: detection.classifierConfidence ?? null,
            bbox_x: detection.boundingBox?.x ?? null,
            bbox_y: detection.boundingBox?.y ?? null,
            bbox_width: detection.boundingBox?.width ?? null,
            bbox_height: detection.boundingBox?.height ?? null,
            corner_tl_x: detection.corners?.topLeft.x ?? null,
            corner_tl_y: detection.corners?.topLeft.y ?? null,
            corner_tr_x: detection.corners?.topRight.x ?? null,
            corner_tr_y: detection.corners?.topRight.y ?? null,
            corner_br_x: detection.corners?.bottomRight.x ?? null,
            corner_br_y: detection.corners?.bottomRight.y ?? null,
            corner_bl_x: detection.corners?.bottomLeft.x ?? null,
            corner_bl_y: detection.corners?.bottomLeft.y ?? null,
            corner_confidence: detection.cornerConfidence ?? null,
            processing_time_ms: data.processing_time_ms,
            updated_at: knex.fn.now(),
        })
        .onConflict(['sample_id', 'model_id'])
        .merge();
};

/**
 * Delete all model detections for a specific model.
 */
export const deleteModelDetections = async (
    modelId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<number> => {
    const deleted = await trx('model_detections')
        .where('model_id', modelId)
        .delete();
    return deleted;
};

/**
 * Delete model detections for specific samples (used when deleting samples).
 */
export const deleteModelDetectionsForSamples = async (
    sampleIds: string[],
    trx: Knex.Transaction | Knex = knex
): Promise<number> => {
    if (sampleIds.length === 0) {
        return 0;
    }
    const deleted = await trx('model_detections')
        .whereIn('sample_id', sampleIds)
        .delete();
    return deleted;
};

/**
 * Get sample IDs that have AI detections for a specific model.
 * If no modelId, uses the latest completed model.
 */
export const getAiLabeledSampleIds = async (
    modelId?: string | null,
    trx: Knex.Transaction | Knex = knex
): Promise<Set<string>> => {
    const effectiveModelId = modelId ?? await getLatestCompletedModelId(trx);
    if (!effectiveModelId) {
        return new Set();
    }
    const rows = await trx('model_detections')
        .where('model_id', effectiveModelId)
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
    // Delete from both tables
    await trx('model_detections')
        .where('sample_id', sampleId)
        .delete();
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
    // Delete from both tables
    await trx('model_detections')
        .whereIn('sample_id', sampleIds)
        .delete();
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
        // Update only the human/ground truth fields
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
        // Create record with ground truth fields only
        // AI detections are stored separately in model_detections
        await trx('board_detections').insert({
            sample_id: sampleId,
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

export const saveCorners = async (
    sampleId: string,
    corners: BoardCorners,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    // Check if record exists
    const existing = await trx('board_detections')
        .where('sample_id', sampleId)
        .first();

    const cornerData = {
        corner_tl_x: corners.topLeft.x,
        corner_tl_y: corners.topLeft.y,
        corner_tr_x: corners.topRight.x,
        corner_tr_y: corners.topRight.y,
        corner_br_x: corners.bottomRight.x,
        corner_br_y: corners.bottomRight.y,
        corner_bl_x: corners.bottomLeft.x,
        corner_bl_y: corners.bottomLeft.y,
        updated_at: knex.fn.now(),
    };

    if (existing) {
        await trx('board_detections')
            .where('sample_id', sampleId)
            .update(cornerData);
    } else {
        await trx('board_detections').insert({
            sample_id: sampleId,
            ...cornerData,
        });
    }
};

export const getCorners = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<BoardCorners | null> => {
    const row = await trx<BoardDetectionRow>('board_detections')
        .where('sample_id', sampleId)
        .first();

    if (!row) {
        return null;
    }

    return rowToCorners(row);
};

export const clearCorners = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const updated = await trx('board_detections')
        .where('sample_id', sampleId)
        .update({
            corner_tl_x: null,
            corner_tl_y: null,
            corner_tr_x: null,
            corner_tr_y: null,
            corner_br_x: null,
            corner_br_y: null,
            corner_bl_x: null,
            corner_bl_y: null,
            updated_at: knex.fn.now(),
        });
    return updated > 0;
};

export interface SampleStatusRow {
    sample_id: string;
    human_labeled_at: Date | null;
    verified_at: Date | null;
}

/**
 * Get status flags for all samples.
 * AI labeled status is determined by checking model_detections for the specified model.
 */
export const getSampleStatusFlags = async (
    modelId?: string | null,
    trx: Knex.Transaction | Knex = knex
): Promise<Map<string, SampleStatusFlags>> => {
    // Get human label status from board_detections
    const humanRows = await trx<SampleStatusRow>('board_detections')
        .select('sample_id', 'human_labeled_at', 'verified_at');

    // Determine which model to check for AI labels
    const effectiveModelId = modelId ?? await getLatestCompletedModelId(trx);

    // Get AI labeled sample IDs from model_detections
    const aiLabeledIds = new Set<string>();
    if (effectiveModelId) {
        const aiRows = await trx('model_detections')
            .where('model_id', effectiveModelId)
            .select('sample_id');
        for (const row of aiRows) {
            aiLabeledIds.add(row.sample_id);
        }
    }

    const result = new Map<string, SampleStatusFlags>();
    for (const row of humanRows) {
        result.set(row.sample_id, {
            aiLabeled: aiLabeledIds.has(row.sample_id),
            humanLabeled: row.human_labeled_at !== null,
            verified: row.verified_at !== null,
        });
    }
    return result;
};

export const getSampleStatusFlagsById = async (
    sampleId: string,
    modelId?: string | null,
    trx: Knex.Transaction | Knex = knex
): Promise<SampleStatusFlags | null> => {
    const row = await trx<SampleStatusRow>('board_detections')
        .where('sample_id', sampleId)
        .select('sample_id', 'human_labeled_at', 'verified_at')
        .first();

    // Determine which model to check for AI labels
    const effectiveModelId = modelId ?? await getLatestCompletedModelId(trx);

    // Check if AI detection exists for this sample
    let hasAiDetection = false;
    if (effectiveModelId) {
        const aiRow = await trx('model_detections')
            .where('sample_id', sampleId)
            .where('model_id', effectiveModelId)
            .first();
        hasAiDetection = !!aiRow;
    }

    if (!row && !hasAiDetection) {
        return null;
    }

    return {
        aiLabeled: hasAiDetection,
        humanLabeled: row?.human_labeled_at !== null,
        verified: row?.verified_at !== null,
    };
};

export interface SampleDetectionInfo {
    status: SampleStatusFlags;
    ai: SampleAiDetection | null;
    human: SampleHumanLabel | null;
    iou: number | null;
}

const calculateIoU = (
    aiX: number, aiY: number, aiW: number, aiH: number,
    hX: number, hY: number, hW: number, hH: number
): number => {
    const interX1 = Math.max(aiX, hX);
    const interY1 = Math.max(aiY, hY);
    const interX2 = Math.min(aiX + aiW, hX + hW);
    const interY2 = Math.min(aiY + aiH, hY + hH);

    const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
    const unionArea = (aiW * aiH) + (hW * hH) - interArea;

    return unionArea === 0 ? 0 : interArea / unionArea;
};

/**
 * Get detection info for all samples, combining human labels from board_detections
 * with AI detections from model_detections for a specific model.
 */
export const getSampleDetectionInfoMap = async (
    modelId?: string | null,
    trx: Knex.Transaction | Knex = knex
): Promise<Map<string, SampleDetectionInfo>> => {
    // Get human labels from board_detections
    const humanRows = await trx('board_detections')
        .select(
            'sample_id',
            'human_has_board',
            'human_is_entire_visible',
            'human_bbox_x', 'human_bbox_y', 'human_bbox_width', 'human_bbox_height',
            'corner_tl_x', 'corner_tl_y', 'corner_tr_x', 'corner_tr_y',
            'corner_br_x', 'corner_br_y', 'corner_bl_x', 'corner_bl_y',
            'human_labeled_at',
            'verified_at'
        );

    // Determine which model to use for AI detections
    const effectiveModelId = modelId ?? await getLatestCompletedModelId(trx);

    // Get AI detections from model_detections for the selected model
    type PartialModelDetection = Pick<ModelDetectionRow,
        'sample_id' | 'has_board_detected' | 'is_entire_board_visible' |
        'confidence' | 'classifier_confidence' |
        'bbox_x' | 'bbox_y' | 'bbox_width' | 'bbox_height' |
        'corner_tl_x' | 'corner_tl_y' | 'corner_tr_x' | 'corner_tr_y' |
        'corner_br_x' | 'corner_br_y' | 'corner_bl_x' | 'corner_bl_y' |
        'corner_confidence'
    >;
    const aiDetectionMap = new Map<string, PartialModelDetection>();
    if (effectiveModelId) {
        const aiRows = await trx<PartialModelDetection>('model_detections')
            .where('model_id', effectiveModelId)
            .select(
                'sample_id',
                'has_board_detected',
                'is_entire_board_visible',
                'confidence',
                'classifier_confidence',
                'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
                'corner_tl_x', 'corner_tl_y', 'corner_tr_x', 'corner_tr_y',
                'corner_br_x', 'corner_br_y', 'corner_bl_x', 'corner_bl_y',
                'corner_confidence'
            );
        for (const row of aiRows) {
            aiDetectionMap.set(row.sample_id, row);
        }
    }

    const result = new Map<string, SampleDetectionInfo>();

    for (const row of humanRows) {
        const aiRow = aiDetectionMap.get(row.sample_id);
        const hasAiDetection = !!aiRow;
        const hasHumanLabel = row.human_labeled_at !== null;

        // Calculate IoU if both AI and human have bounding boxes
        let iou: number | null = null;
        if (
            hasAiDetection &&
            hasHumanLabel &&
            aiRow?.bbox_x !== null &&
            row.human_bbox_x !== null
        ) {
            iou = calculateIoU(
                aiRow.bbox_x!, aiRow.bbox_y!, aiRow.bbox_width!, aiRow.bbox_height!,
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
                hasBoardDetected: aiRow!.has_board_detected,
                isEntireBoardVisible: aiRow!.is_entire_board_visible,
                confidence: aiRow!.confidence ?? 0,
                classifierConfidence: aiRow!.classifier_confidence,
                boundingBox: aiRow!.bbox_x !== null ? {
                    x: aiRow!.bbox_x,
                    y: aiRow!.bbox_y!,
                    width: aiRow!.bbox_width!,
                    height: aiRow!.bbox_height!,
                } : null,
                corners: aiRow!.corner_tl_x !== null ? {
                    topLeft: { x: aiRow!.corner_tl_x, y: aiRow!.corner_tl_y! },
                    topRight: { x: aiRow!.corner_tr_x!, y: aiRow!.corner_tr_y! },
                    bottomRight: { x: aiRow!.corner_br_x!, y: aiRow!.corner_br_y! },
                    bottomLeft: { x: aiRow!.corner_bl_x!, y: aiRow!.corner_bl_y! },
                } : null,
                cornerError: null, // Corner error not stored in model_detections
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
                corners: row.corner_tl_x !== null ? {
                    topLeft: { x: row.corner_tl_x, y: row.corner_tl_y },
                    topRight: { x: row.corner_tr_x, y: row.corner_tr_y },
                    bottomRight: { x: row.corner_br_x, y: row.corner_br_y },
                    bottomLeft: { x: row.corner_bl_x, y: row.corner_bl_y },
                } : null,
            } : null,
            iou,
        });
        // Remove from aiDetectionMap since we've processed this sample
        aiDetectionMap.delete(row.sample_id);
    }

    // Add samples that have AI detections but no human labels
    for (const [sampleId, aiRow] of aiDetectionMap) {
        result.set(sampleId, {
            status: {
                aiLabeled: true,
                humanLabeled: false,
                verified: false,
            },
            ai: {
                hasBoardDetected: aiRow.has_board_detected,
                isEntireBoardVisible: aiRow.is_entire_board_visible,
                confidence: aiRow.confidence ?? 0,
                classifierConfidence: aiRow.classifier_confidence,
                boundingBox: aiRow.bbox_x !== null ? {
                    x: aiRow.bbox_x,
                    y: aiRow.bbox_y!,
                    width: aiRow.bbox_width!,
                    height: aiRow.bbox_height!,
                } : null,
                corners: aiRow.corner_tl_x !== null ? {
                    topLeft: { x: aiRow.corner_tl_x, y: aiRow.corner_tl_y! },
                    topRight: { x: aiRow.corner_tr_x!, y: aiRow.corner_tr_y! },
                    bottomRight: { x: aiRow.corner_br_x!, y: aiRow.corner_br_y! },
                    bottomLeft: { x: aiRow.corner_bl_x!, y: aiRow.corner_bl_y! },
                } : null,
                cornerError: null, // Corner error not stored in model_detections
            },
            human: null,
            iou: null,
        });
    }

    return result;
};

/**
 * Calculate the average IoU across all samples that have both AI and human bounding boxes
 * for a specific model.
 */
export const calculateAverageIoU = async (
    modelId?: string | null,
    trx: Knex.Transaction | Knex = knex
): Promise<number | null> => {
    const effectiveModelId = modelId ?? await getLatestCompletedModelId(trx);
    if (!effectiveModelId) {
        return null;
    }

    // Join model_detections with board_detections to compare AI and human bboxes
    const rows = await trx('model_detections as md')
        .join('board_detections as bd', 'md.sample_id', 'bd.sample_id')
        .where('md.model_id', effectiveModelId)
        .whereNotNull('md.bbox_x')
        .whereNotNull('bd.human_bbox_x')
        .select(
            'md.bbox_x', 'md.bbox_y', 'md.bbox_width', 'md.bbox_height',
            'bd.human_bbox_x', 'bd.human_bbox_y', 'bd.human_bbox_width', 'bd.human_bbox_height'
        );

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
    pose_accuracy: number | null;
    pose_loss: number | null;
    pose_error: number | null;
    pose_image_count: number | null;
    error_message: string | null;
    model_s3_url: string | null;
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
    poseAccuracy: row.pose_accuracy,
    poseLoss: row.pose_loss,
    poseError: row.pose_error,
    poseImageCount: row.pose_image_count,
    errorMessage: row.error_message,
    modelS3Url: row.model_s3_url,
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
    poseAccuracy?: number;
    poseLoss?: number;
    poseError?: number;
    poseImageCount?: number;
    errorMessage?: string;
    modelS3Url?: string;
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
    if (updates.poseAccuracy !== undefined) {
        dbUpdates.pose_accuracy = updates.poseAccuracy;
    }
    if (updates.poseLoss !== undefined) {
        dbUpdates.pose_loss = updates.poseLoss;
    }
    if (updates.poseError !== undefined) {
        dbUpdates.pose_error = updates.poseError;
    }
    if (updates.poseImageCount !== undefined) {
        dbUpdates.pose_image_count = updates.poseImageCount;
    }
    if (updates.errorMessage !== undefined) {
        dbUpdates.error_message = updates.errorMessage;
    }
    if (updates.modelS3Url !== undefined) {
        dbUpdates.model_s3_url = updates.modelS3Url;
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
    sampleIds?: string[],
    trx: Knex.Transaction | Knex = knex
): Promise<number> => {
    let query = trx('board_detections')
        .whereNotNull('human_labeled_at');

    // Filter by specific sample IDs if provided
    if (sampleIds && sampleIds.length > 0) {
        query = query.whereIn('sample_id', sampleIds);
    }

    const result = await query.count('* as count').first();
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
    sampleIds?: string[],
    trx: Knex.Transaction | Knex = knex
): Promise<TrainingDataSample[]> => {
    // Get human-labeled samples with bounding boxes (ground truth for training)
    let query = trx<BoardDetectionRow>('board_detections')
        .whereNotNull('human_labeled_at')
        .select(
            'sample_id',
            'human_bbox_x',
            'human_bbox_y',
            'human_bbox_width',
            'human_bbox_height'
        );

    // Filter by specific sample IDs if provided
    if (sampleIds && sampleIds.length > 0) {
        query = query.whereIn('sample_id', sampleIds);
    }

    const rows = await query;

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

export interface PoseTrainingSample {
    sample_id: string;
    image_url: string;
    bbox: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
    corners: {
        topLeft: { x: number; y: number };
        topRight: { x: number; y: number };
        bottomRight: { x: number; y: number };
        bottomLeft: { x: number; y: number };
    };
}

export const getPoseTrainingData = async (
    trx: Knex.Transaction | Knex = knex
): Promise<PoseTrainingSample[]> => {
    // Get human-labeled samples with corner labels for pose training
    const rows = await trx<BoardDetectionRow>('board_detections')
        .whereNotNull('human_labeled_at')
        .whereNotNull('corner_tl_x') // Must have corners labeled
        .select(
            'sample_id',
            'human_bbox_x',
            'human_bbox_y',
            'human_bbox_width',
            'human_bbox_height',
            'corner_tl_x', 'corner_tl_y',
            'corner_tr_x', 'corner_tr_y',
            'corner_br_x', 'corner_br_y',
            'corner_bl_x', 'corner_bl_y'
        );

    const cloudFrontDomain = 'd26azxov2byjpl.cloudfront.net';

    return rows
        .filter(row => {
            // Must have human bbox and all corners
            return row.human_bbox_x !== null && row.corner_tl_x !== null;
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
                corners: {
                    topLeft: { x: row.corner_tl_x!, y: row.corner_tl_y! },
                    topRight: { x: row.corner_tr_x!, y: row.corner_tr_y! },
                    bottomRight: { x: row.corner_br_x!, y: row.corner_br_y! },
                    bottomLeft: { x: row.corner_bl_x!, y: row.corner_bl_y! },
                },
            };
        });
};

export const getPoseTrainingImageCount = async (
    trx: Knex.Transaction | Knex = knex
): Promise<number> => {
    const result = await trx('board_detections')
        .whereNotNull('human_labeled_at')
        .whereNotNull('corner_tl_x')
        .count('* as count')
        .first();
    return Number(result?.count ?? 0);
};

// ============================================================================
// Detection Runs
// ============================================================================

interface DetectionRunRow {
    id: string;
    status: DetectionRunStatus;
    model_id: string;
    pipeline_id: string | null;
    total_count: number;
    processed_count: number;
    success_count: number | null;
    failed_count: number | null;
    started_at: Date | null;
    completed_at: Date | null;
    duration_ms: number | null;
    error_message: string | null;
    avg_iou: number | null;
    avg_corner_error: number | null;
    corner_accuracy: number | null;
    created_at: Date;
    updated_at: Date;
    // Computed fields from join
    disagreement_count?: number | null;
    avg_classifier_accuracy?: number | null;
}

const rowToDetectionRun = (row: DetectionRunRow): DetectionRun => ({
    id: row.id,
    status: row.status,
    modelId: row.model_id,
    pipelineId: row.pipeline_id,
    totalCount: row.total_count,
    processedCount: row.processed_count,
    successCount: row.success_count,
    failedCount: row.failed_count,
    disagreementCount: row.disagreement_count ?? null,
    startedAt: row.started_at?.toISOString() ?? null,
    completedAt: row.completed_at?.toISOString() ?? null,
    durationMs: row.duration_ms,
    errorMessage: row.error_message,
    avgIou: row.avg_iou,
    avgClassifierAccuracy: row.avg_classifier_accuracy ?? null,
    avgCornerError: row.avg_corner_error,
    cornerAccuracy: row.corner_accuracy,
    createdAt: row.created_at.toISOString(),
});

export const createDetectionRun = async (
    modelId: string,
    totalCount: number,
    pipelineId?: string | null,
    trx: Knex.Transaction | Knex = knex
): Promise<DetectionRun> => {
    const [row] = await trx<DetectionRunRow>('detection_runs')
        .insert({
            model_id: modelId,
            total_count: totalCount,
            pipeline_id: pipelineId || null,
        })
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

    await trx<DetectionRunRow>('detection_runs')
        .where('id', id)
        .update(dbUpdates);

    // Return full object with computed fields
    return getDetectionRun(id, trx);
};

export const listDetectionRuns = async (
    trx: Knex.Transaction | Knex = knex
): Promise<DetectionRun[]> => {
    // Get detection runs with computed stats via subqueries
    // Only compute metrics for completed runs
    const rows = await trx<DetectionRunRow>('detection_runs as dr')
        .select('dr.*')
        .select(
            trx.raw(`(
                CASE WHEN dr.status = 'completed' THEN (
                    SELECT COUNT(*)
                    FROM model_detections md
                    INNER JOIN board_detections bd ON bd.sample_id = md.sample_id
                    WHERE md.model_id = dr.model_id
                    AND bd.human_has_board IS NOT NULL
                    AND md.has_board_detected != bd.human_has_board
                ) ELSE NULL END
            ) as disagreement_count`)
        )
        .select(
            trx.raw(`(
                CASE WHEN dr.status = 'completed' THEN (
                    SELECT AVG(
                        CASE WHEN md.is_entire_board_visible = bd.human_is_entire_visible
                        THEN 1.0 ELSE 0.0 END
                    )
                    FROM model_detections md
                    INNER JOIN board_detections bd ON bd.sample_id = md.sample_id
                    WHERE md.model_id = dr.model_id
                    AND bd.human_is_entire_visible IS NOT NULL
                    AND md.classifier_confidence IS NOT NULL
                ) ELSE NULL END
            ) as avg_classifier_accuracy`)
        )
        .select(
            // Average corner error in pixels (using euclidean distance)
            trx.raw(`(
                CASE WHEN dr.status = 'completed' THEN (
                    SELECT AVG(
                        SQRT(
                            POWER(md.corner_tl_x - bd.corner_tl_x, 2) + POWER(md.corner_tl_y - bd.corner_tl_y, 2) +
                            POWER(md.corner_tr_x - bd.corner_tr_x, 2) + POWER(md.corner_tr_y - bd.corner_tr_y, 2) +
                            POWER(md.corner_br_x - bd.corner_br_x, 2) + POWER(md.corner_br_y - bd.corner_br_y, 2) +
                            POWER(md.corner_bl_x - bd.corner_bl_x, 2) + POWER(md.corner_bl_y - bd.corner_bl_y, 2)
                        ) / 4.0
                    )
                    FROM model_detections md
                    INNER JOIN board_detections bd ON bd.sample_id = md.sample_id
                    WHERE md.model_id = dr.model_id
                    AND md.corner_tl_x IS NOT NULL
                    AND bd.corner_tl_x IS NOT NULL
                ) ELSE NULL END
            ) as avg_corner_error`)
        )
        .select(
            // Corner accuracy: percentage of corners within threshold (0.05 = 5% of image)
            trx.raw(`(
                CASE WHEN dr.status = 'completed' THEN (
                    SELECT AVG(
                        CASE WHEN
                            SQRT(POWER(md.corner_tl_x - bd.corner_tl_x, 2) + POWER(md.corner_tl_y - bd.corner_tl_y, 2)) < 0.05 AND
                            SQRT(POWER(md.corner_tr_x - bd.corner_tr_x, 2) + POWER(md.corner_tr_y - bd.corner_tr_y, 2)) < 0.05 AND
                            SQRT(POWER(md.corner_br_x - bd.corner_br_x, 2) + POWER(md.corner_br_y - bd.corner_br_y, 2)) < 0.05 AND
                            SQRT(POWER(md.corner_bl_x - bd.corner_bl_x, 2) + POWER(md.corner_bl_y - bd.corner_bl_y, 2)) < 0.05
                        THEN 1.0 ELSE 0.0 END
                    )
                    FROM model_detections md
                    INNER JOIN board_detections bd ON bd.sample_id = md.sample_id
                    WHERE md.model_id = dr.model_id
                    AND md.corner_tl_x IS NOT NULL
                    AND bd.corner_tl_x IS NOT NULL
                ) ELSE NULL END
            ) as corner_accuracy`)
        )
        .orderBy('dr.created_at', 'desc');
    return rows.map(rowToDetectionRun);
};

export const getDetectionRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<DetectionRun | null> => {
    // Only compute metrics for completed runs
    const rows = await trx<DetectionRunRow>('detection_runs as dr')
        .select('dr.*')
        .select(
            trx.raw(`(
                CASE WHEN dr.status = 'completed' THEN (
                    SELECT COUNT(*)
                    FROM model_detections md
                    INNER JOIN board_detections bd ON bd.sample_id = md.sample_id
                    WHERE md.model_id = dr.model_id
                    AND bd.human_has_board IS NOT NULL
                    AND md.has_board_detected != bd.human_has_board
                ) ELSE NULL END
            ) as disagreement_count`)
        )
        .select(
            trx.raw(`(
                CASE WHEN dr.status = 'completed' THEN (
                    SELECT AVG(
                        CASE WHEN md.is_entire_board_visible = bd.human_is_entire_visible
                        THEN 1.0 ELSE 0.0 END
                    )
                    FROM model_detections md
                    INNER JOIN board_detections bd ON bd.sample_id = md.sample_id
                    WHERE md.model_id = dr.model_id
                    AND bd.human_is_entire_visible IS NOT NULL
                    AND md.classifier_confidence IS NOT NULL
                ) ELSE NULL END
            ) as avg_classifier_accuracy`)
        )
        .select(
            // Average corner error in pixels (using euclidean distance)
            trx.raw(`(
                CASE WHEN dr.status = 'completed' THEN (
                    SELECT AVG(
                        SQRT(
                            POWER(md.corner_tl_x - bd.corner_tl_x, 2) + POWER(md.corner_tl_y - bd.corner_tl_y, 2) +
                            POWER(md.corner_tr_x - bd.corner_tr_x, 2) + POWER(md.corner_tr_y - bd.corner_tr_y, 2) +
                            POWER(md.corner_br_x - bd.corner_br_x, 2) + POWER(md.corner_br_y - bd.corner_br_y, 2) +
                            POWER(md.corner_bl_x - bd.corner_bl_x, 2) + POWER(md.corner_bl_y - bd.corner_bl_y, 2)
                        ) / 4.0
                    )
                    FROM model_detections md
                    INNER JOIN board_detections bd ON bd.sample_id = md.sample_id
                    WHERE md.model_id = dr.model_id
                    AND md.corner_tl_x IS NOT NULL
                    AND bd.corner_tl_x IS NOT NULL
                ) ELSE NULL END
            ) as avg_corner_error`)
        )
        .select(
            // Corner accuracy: percentage of corners within threshold (0.05 = 5% of image)
            trx.raw(`(
                CASE WHEN dr.status = 'completed' THEN (
                    SELECT AVG(
                        CASE WHEN
                            SQRT(POWER(md.corner_tl_x - bd.corner_tl_x, 2) + POWER(md.corner_tl_y - bd.corner_tl_y, 2)) < 0.05 AND
                            SQRT(POWER(md.corner_tr_x - bd.corner_tr_x, 2) + POWER(md.corner_tr_y - bd.corner_tr_y, 2)) < 0.05 AND
                            SQRT(POWER(md.corner_br_x - bd.corner_br_x, 2) + POWER(md.corner_br_y - bd.corner_br_y, 2)) < 0.05 AND
                            SQRT(POWER(md.corner_bl_x - bd.corner_bl_x, 2) + POWER(md.corner_bl_y - bd.corner_bl_y, 2)) < 0.05
                        THEN 1.0 ELSE 0.0 END
                    )
                    FROM model_detections md
                    INNER JOIN board_detections bd ON bd.sample_id = md.sample_id
                    WHERE md.model_id = dr.model_id
                    AND md.corner_tl_x IS NOT NULL
                    AND bd.corner_tl_x IS NOT NULL
                ) ELSE NULL END
            ) as corner_accuracy`)
        )
        .where('dr.id', id);
    return rows.length > 0 ? rowToDetectionRun(rows[0]) : null;
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
 * Clear AI detections for a specific model.
 * (Model detections are cascade-deleted when training run is deleted,
 * but this can be used to explicitly clear them.)
 */
export const clearModelDetections = async (
    modelId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<number> => {
    const deleted = await trx('model_detections')
        .where('model_id', modelId)
        .delete();
    return deleted;
};

export interface BulkDetectionFilterResult {
    sampleIds: string[];
    existingDetectionCount: number;
}

/**
 * Get sample IDs that should be processed in bulk detection for a specific model.
 * The checkbox controls whether to re-run on samples that already have AI detections for this model.
 */
export const getSampleIdsForBulkDetection = async (
    allSampleIds: string[],
    modelId: string,
    includeExistingDetections: boolean,
    trx: Knex.Transaction | Knex = knex
): Promise<BulkDetectionFilterResult> => {
    // Get samples that already have AI detection for this model
    const rows = await trx('model_detections')
        .where('model_id', modelId)
        .select('sample_id');
    const existingDetectionIds = new Set(rows.map(r => r.sample_id));
    const existingDetectionCount = allSampleIds.filter(id => existingDetectionIds.has(id)).length;

    let eligibleSampleIds = allSampleIds;

    if (!includeExistingDetections) {
        // Exclude samples that already have AI detections for this model
        eligibleSampleIds = allSampleIds.filter(id => !existingDetectionIds.has(id));
    }

    return {
        sampleIds: eligibleSampleIds,
        existingDetectionCount,
    };
};

// ============================================================================
// Model Pipelines
// ============================================================================

interface ModelPipelineRow {
    id: string;
    name: string;
    component_ids: string[];
    created_at: Date;
    updated_at: Date;
}

const rowToModelPipeline = (row: ModelPipelineRow): ModelPipeline => ({
    id: row.id,
    name: row.name,
    componentIds: row.component_ids,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
});

export const createPipeline = async (
    name: string,
    componentIds: string[],
    trx: Knex.Transaction | Knex = knex
): Promise<ModelPipeline> => {
    const [row] = await trx<ModelPipelineRow>('model_pipelines')
        .insert({
            name,
            component_ids: JSON.stringify(componentIds) as unknown as string[],
        })
        .returning('*');
    return rowToModelPipeline(row);
};

export const listPipelines = async (
    trx: Knex.Transaction | Knex = knex
): Promise<ModelPipeline[]> => {
    const rows = await trx<ModelPipelineRow>('model_pipelines')
        .orderBy('created_at', 'desc');
    return rows.map(rowToModelPipeline);
};

export const getPipeline = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<ModelPipeline | null> => {
    const row = await trx<ModelPipelineRow>('model_pipelines')
        .where('id', id)
        .first();
    return row ? rowToModelPipeline(row) : null;
};

export interface PipelineUpdate {
    name?: string;
    componentIds?: string[];
}

export const updatePipeline = async (
    id: string,
    updates: PipelineUpdate,
    trx: Knex.Transaction | Knex = knex
): Promise<ModelPipeline | null> => {
    const dbUpdates: Partial<ModelPipelineRow> = {
        updated_at: new Date(),
    };

    if (updates.name !== undefined) {
        dbUpdates.name = updates.name;
    }
    if (updates.componentIds !== undefined) {
        dbUpdates.component_ids = JSON.stringify(updates.componentIds) as unknown as string[];
    }

    const [row] = await trx<ModelPipelineRow>('model_pipelines')
        .where('id', id)
        .update(dbUpdates)
        .returning('*');

    return row ? rowToModelPipeline(row) : null;
};

export const deletePipeline = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('model_pipelines')
        .where('id', id)
        .delete();
    return deleted > 0;
};

/**
 * Get a pipeline with its component training runs populated.
 */
export const getPipelineWithComponents = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<ModelPipeline | null> => {
    const pipeline = await getPipeline(id, trx);
    if (!pipeline) {
        return null;
    }

    // Fetch all component training runs
    const componentRows = await trx<TrainingRunRow>('training_runs')
        .whereIn('id', pipeline.componentIds);

    // Map to TrainingRun and preserve order from componentIds
    const componentsMap = new Map(componentRows.map(r => [r.id, rowToTrainingRun(r)]));
    const components = pipeline.componentIds
        .map(id => componentsMap.get(id))
        .filter((c): c is TrainingRun => c !== undefined);

    return {
        ...pipeline,
        components,
    };
};

// ============================================================================
// Corner Detection Runs
// ============================================================================

interface CornerDetectionRunRow {
    id: string;
    status: CornerDetectionRunStatus;
    total_count: number;
    processed_count: number;
    success_count: number | null;
    failed_count: number | null;
    started_at: Date | null;
    completed_at: Date | null;
    duration_ms: number | null;
    error_message: string | null;
    created_at: Date;
    updated_at: Date;
}

const rowToCornerDetectionRun = (row: CornerDetectionRunRow): CornerDetectionRun => ({
    id: row.id,
    status: row.status,
    totalCount: row.total_count,
    processedCount: row.processed_count,
    successCount: row.success_count,
    failedCount: row.failed_count,
    startedAt: row.started_at?.toISOString() ?? null,
    completedAt: row.completed_at?.toISOString() ?? null,
    durationMs: row.duration_ms,
    errorMessage: row.error_message,
    createdAt: row.created_at.toISOString(),
});

export const createCornerDetectionRun = async (
    totalCount: number,
    trx: Knex.Transaction | Knex = knex
): Promise<CornerDetectionRun> => {
    const [row] = await trx<CornerDetectionRunRow>('corner_detection_runs')
        .insert({ total_count: totalCount })
        .returning('*');
    return rowToCornerDetectionRun(row);
};

export interface CornerDetectionRunUpdate {
    status?: CornerDetectionRunStatus;
    processedCount?: number;
    successCount?: number;
    failedCount?: number;
    startedAt?: Date;
    completedAt?: Date;
    durationMs?: number;
    errorMessage?: string;
}

export const updateCornerDetectionRun = async (
    id: string,
    updates: CornerDetectionRunUpdate,
    trx: Knex.Transaction | Knex = knex
): Promise<CornerDetectionRun | null> => {
    const dbUpdates: Partial<CornerDetectionRunRow> = {
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

    const [row] = await trx<CornerDetectionRunRow>('corner_detection_runs')
        .where('id', id)
        .update(dbUpdates)
        .returning('*');

    return row ? rowToCornerDetectionRun(row) : null;
};

export const listCornerDetectionRuns = async (
    trx: Knex.Transaction | Knex = knex
): Promise<CornerDetectionRun[]> => {
    const rows = await trx<CornerDetectionRunRow>('corner_detection_runs')
        .orderBy('created_at', 'desc');
    return rows.map(rowToCornerDetectionRun);
};

export const getCornerDetectionRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<CornerDetectionRun | null> => {
    const row = await trx<CornerDetectionRunRow>('corner_detection_runs')
        .where('id', id)
        .first();
    return row ? rowToCornerDetectionRun(row) : null;
};

export const deleteCornerDetectionRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('corner_detection_runs')
        .where('id', id)
        .delete();
    return deleted > 0;
};

// ============================================================================
// Corner Detections (results per sample)
// ============================================================================

interface CornerDetectionRow {
    id: string;
    sample_id: string;
    run_id: string;
    // "best" method - automatically selected between hough and edge
    best_success: boolean | null;
    best_tl_x: number | null;
    best_tl_y: number | null;
    best_tr_x: number | null;
    best_tr_y: number | null;
    best_br_x: number | null;
    best_br_y: number | null;
    best_bl_x: number | null;
    best_bl_y: number | null;
    best_error: number | null;
    best_aspect_ratio: number | null;
    best_area_ratio: number | null;
    best_avg_angle: number | null;
    hough_success: boolean | null;
    hough_tl_x: number | null;
    hough_tl_y: number | null;
    hough_tr_x: number | null;
    hough_tr_y: number | null;
    hough_br_x: number | null;
    hough_br_y: number | null;
    hough_bl_x: number | null;
    hough_bl_y: number | null;
    hough_error: number | null;
    hough_aspect_ratio: number | null;
    hough_area_ratio: number | null;
    hough_avg_angle: number | null;
    edge_success: boolean | null;
    edge_tl_x: number | null;
    edge_tl_y: number | null;
    edge_tr_x: number | null;
    edge_tr_y: number | null;
    edge_br_x: number | null;
    edge_br_y: number | null;
    edge_bl_x: number | null;
    edge_bl_y: number | null;
    edge_error: number | null;
    edge_aspect_ratio: number | null;
    edge_area_ratio: number | null;
    edge_avg_angle: number | null;
    best_cropped_url: string | null;
    hough_cropped_url: string | null;
    edge_cropped_url: string | null;
    created_at: Date;
}

const rowToCornerDetectionResult = (row: CornerDetectionRow): CornerDetectionResult => ({
    sampleId: row.sample_id,
    runId: row.run_id,
    best: {
        success: row.best_success ?? false,
        corners: row.best_tl_x !== null ? {
            topLeft: { x: row.best_tl_x, y: row.best_tl_y! },
            topRight: { x: row.best_tr_x!, y: row.best_tr_y! },
            bottomRight: { x: row.best_br_x!, y: row.best_br_y! },
            bottomLeft: { x: row.best_bl_x!, y: row.best_bl_y! },
        } : null,
        error: row.best_error,
        quality: row.best_aspect_ratio !== null ? {
            aspectRatio: row.best_aspect_ratio,
            areaRatio: row.best_area_ratio,
            avgAngle: row.best_avg_angle,
        } : null,
        croppedUrl: row.best_cropped_url,
    },
    hough: {
        success: row.hough_success ?? false,
        corners: row.hough_tl_x !== null ? {
            topLeft: { x: row.hough_tl_x, y: row.hough_tl_y! },
            topRight: { x: row.hough_tr_x!, y: row.hough_tr_y! },
            bottomRight: { x: row.hough_br_x!, y: row.hough_br_y! },
            bottomLeft: { x: row.hough_bl_x!, y: row.hough_bl_y! },
        } : null,
        error: row.hough_error,
        quality: row.hough_aspect_ratio !== null ? {
            aspectRatio: row.hough_aspect_ratio,
            areaRatio: row.hough_area_ratio,
            avgAngle: row.hough_avg_angle,
        } : null,
        croppedUrl: row.hough_cropped_url,
    },
    edge: {
        success: row.edge_success ?? false,
        corners: row.edge_tl_x !== null ? {
            topLeft: { x: row.edge_tl_x, y: row.edge_tl_y! },
            topRight: { x: row.edge_tr_x!, y: row.edge_tr_y! },
            bottomRight: { x: row.edge_br_x!, y: row.edge_br_y! },
            bottomLeft: { x: row.edge_bl_x!, y: row.edge_bl_y! },
        } : null,
        error: row.edge_error,
        quality: row.edge_aspect_ratio !== null ? {
            aspectRatio: row.edge_aspect_ratio,
            areaRatio: row.edge_area_ratio,
            avgAngle: row.edge_avg_angle,
        } : null,
        croppedUrl: row.edge_cropped_url,
    },
});

export interface CornerQualityData {
    aspectRatio: number | null;
    areaRatio: number | null;
    avgAngle: number | null;
}

export interface CornerDetectionData {
    sampleId: string;
    runId: string;
    best: {
        success: boolean;
        corners: BoardCorners | null;
        error: number | null;
        quality?: CornerQualityData | null;
        croppedUrl?: string | null;
    };
    hough: {
        success: boolean;
        corners: BoardCorners | null;
        error: number | null;
        quality?: CornerQualityData | null;
        croppedUrl?: string | null;
    };
    edge: {
        success: boolean;
        corners: BoardCorners | null;
        error: number | null;
        quality?: CornerQualityData | null;
        croppedUrl?: string | null;
    };
}

export const upsertCornerDetection = async (
    data: CornerDetectionData,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    await trx('corner_detections')
        .insert({
            sample_id: data.sampleId,
            run_id: data.runId,
            best_success: data.best.success,
            best_tl_x: data.best.corners?.topLeft.x ?? null,
            best_tl_y: data.best.corners?.topLeft.y ?? null,
            best_tr_x: data.best.corners?.topRight.x ?? null,
            best_tr_y: data.best.corners?.topRight.y ?? null,
            best_br_x: data.best.corners?.bottomRight.x ?? null,
            best_br_y: data.best.corners?.bottomRight.y ?? null,
            best_bl_x: data.best.corners?.bottomLeft.x ?? null,
            best_bl_y: data.best.corners?.bottomLeft.y ?? null,
            best_error: data.best.error,
            best_aspect_ratio: data.best.quality?.aspectRatio ?? null,
            best_area_ratio: data.best.quality?.areaRatio ?? null,
            best_avg_angle: data.best.quality?.avgAngle ?? null,
            hough_success: data.hough.success,
            hough_tl_x: data.hough.corners?.topLeft.x ?? null,
            hough_tl_y: data.hough.corners?.topLeft.y ?? null,
            hough_tr_x: data.hough.corners?.topRight.x ?? null,
            hough_tr_y: data.hough.corners?.topRight.y ?? null,
            hough_br_x: data.hough.corners?.bottomRight.x ?? null,
            hough_br_y: data.hough.corners?.bottomRight.y ?? null,
            hough_bl_x: data.hough.corners?.bottomLeft.x ?? null,
            hough_bl_y: data.hough.corners?.bottomLeft.y ?? null,
            hough_error: data.hough.error,
            hough_aspect_ratio: data.hough.quality?.aspectRatio ?? null,
            hough_area_ratio: data.hough.quality?.areaRatio ?? null,
            hough_avg_angle: data.hough.quality?.avgAngle ?? null,
            edge_success: data.edge.success,
            edge_tl_x: data.edge.corners?.topLeft.x ?? null,
            edge_tl_y: data.edge.corners?.topLeft.y ?? null,
            edge_tr_x: data.edge.corners?.topRight.x ?? null,
            edge_tr_y: data.edge.corners?.topRight.y ?? null,
            edge_br_x: data.edge.corners?.bottomRight.x ?? null,
            edge_br_y: data.edge.corners?.bottomRight.y ?? null,
            edge_bl_x: data.edge.corners?.bottomLeft.x ?? null,
            edge_bl_y: data.edge.corners?.bottomLeft.y ?? null,
            edge_error: data.edge.error,
            edge_aspect_ratio: data.edge.quality?.aspectRatio ?? null,
            edge_area_ratio: data.edge.quality?.areaRatio ?? null,
            edge_avg_angle: data.edge.quality?.avgAngle ?? null,
            best_cropped_url: data.best.croppedUrl ?? null,
            hough_cropped_url: data.hough.croppedUrl ?? null,
            edge_cropped_url: data.edge.croppedUrl ?? null,
        })
        .onConflict(['sample_id', 'run_id'])
        .merge();
};

export const getCornerDetectionsByRunId = async (
    runId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<CornerDetectionResult[]> => {
    const rows = await trx<CornerDetectionRow>('corner_detections')
        .where('run_id', runId);
    return rows.map(rowToCornerDetectionResult);
};

export const getLatestCornerDetectionForSample = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<CornerDetectionResult | null> => {
    const row = await trx<CornerDetectionRow>('corner_detections')
        .where('sample_id', sampleId)
        .orderBy('created_at', 'desc')
        .first();
    return row ? rowToCornerDetectionResult(row) : null;
};

/**
 * Get corner detections for multiple samples at once (batch query).
 * Returns the latest result for each sample.
 */
export const getCornerDetectionsForSamples = async (
    sampleIds: string[],
    trx: Knex.Transaction | Knex = knex
): Promise<Map<string, CornerDetectionResult>> => {
    if (sampleIds.length === 0) {
        return new Map();
    }

    // Use a subquery to get the latest detection for each sample
    const rows = await trx<CornerDetectionRow>('corner_detections as cd')
        .whereIn('cd.sample_id', sampleIds)
        .whereRaw(`cd.created_at = (
            SELECT MAX(cd2.created_at)
            FROM corner_detections cd2
            WHERE cd2.sample_id = cd.sample_id
        )`)
        .select('cd.*');

    const result = new Map<string, CornerDetectionResult>();
    for (const row of rows) {
        result.set(row.sample_id, rowToCornerDetectionResult(row));
    }
    return result;
};

// ============================================================================
// Grid Labels
// ============================================================================

interface GridLabelRow {
    sample_id: string;
    grid: GridLabels;
    labeled_at: Date;
    created_at: Date;
    updated_at: Date;
}

const rowToGridLabel = (row: GridLabelRow): GridLabel => ({
    sampleId: row.sample_id,
    grid: row.grid,
    labeledAt: row.labeled_at.toISOString(),
});

export const getGridLabel = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<GridLabel | null> => {
    const row = await trx<GridLabelRow>('grid_labels')
        .where('sample_id', sampleId)
        .first();
    return row ? rowToGridLabel(row) : null;
};

export const saveGridLabel = async (
    sampleId: string,
    grid: GridLabels,
    trx: Knex.Transaction | Knex = knex
): Promise<GridLabel> => {
    const now = new Date();

    // Upsert the grid label
    await trx('grid_labels')
        .insert({
            sample_id: sampleId,
            grid: JSON.stringify(grid),
            labeled_at: now,
            created_at: now,
            updated_at: now,
        })
        .onConflict('sample_id')
        .merge({
            grid: JSON.stringify(grid),
            labeled_at: now,
            updated_at: now,
        });

    return {
        sampleId,
        grid,
        labeledAt: now.toISOString(),
    };
};

export const deleteGridLabel = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    await trx('grid_labels').where('sample_id', sampleId).delete();
};

export const getGridLabelsForSamples = async (
    sampleIds: string[],
    trx: Knex.Transaction | Knex = knex
): Promise<Map<string, GridLabel>> => {
    if (sampleIds.length === 0) {
        return new Map();
    }
    const rows = await trx<GridLabelRow>('grid_labels')
        .whereIn('sample_id', sampleIds);
    const result = new Map<string, GridLabel>();
    for (const row of rows) {
        result.set(row.sample_id, rowToGridLabel(row));
    }
    return result;
};

// ============================================================================
// Grid Predictions (AI predictions, separate from human labels)
// ============================================================================

interface GridPredictionRow {
    sample_id: string;
    grid: GridLabels;
    confidence: number;
    processing_time_ms: number;
    created_at: Date;
    updated_at: Date;
}

const rowToGridPrediction = (row: GridPredictionRow): GridPredictionRecord => ({
    sampleId: row.sample_id,
    grid: row.grid,
    confidence: row.confidence,
    processingTimeMs: row.processing_time_ms,
    createdAt: row.created_at.toISOString(),
});

export const getGridPrediction = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<GridPredictionRecord | null> => {
    const row = await trx<GridPredictionRow>('grid_predictions')
        .where('sample_id', sampleId)
        .first();
    return row ? rowToGridPrediction(row) : null;
};

export const getGridPredictionsForSamples = async (
    sampleIds: string[],
    trx: Knex.Transaction | Knex = knex
): Promise<Map<string, GridPredictionRecord>> => {
    if (sampleIds.length === 0) {
        return new Map();
    }

    const rows = await trx<GridPredictionRow>('grid_predictions')
        .whereIn('sample_id', sampleIds);

    const result = new Map<string, GridPredictionRecord>();
    for (const row of rows) {
        result.set(row.sample_id, rowToGridPrediction(row));
    }
    return result;
};

export const saveGridPrediction = async (
    sampleId: string,
    grid: GridLabels,
    confidence: number,
    processingTimeMs: number,
    trx: Knex.Transaction | Knex = knex
): Promise<GridPredictionRecord> => {
    const now = new Date();

    await trx('grid_predictions')
        .insert({
            sample_id: sampleId,
            grid: JSON.stringify(grid),
            confidence,
            processing_time_ms: processingTimeMs,
            created_at: now,
            updated_at: now,
        })
        .onConflict('sample_id')
        .merge({
            grid: JSON.stringify(grid),
            confidence,
            processing_time_ms: processingTimeMs,
            updated_at: now,
        });

    return {
        sampleId,
        grid,
        confidence,
        processingTimeMs,
        createdAt: now.toISOString(),
    };
};

export const deleteGridPrediction = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    await trx('grid_predictions').where('sample_id', sampleId).delete();
};

/**
 * Get sample IDs that have grid labels (human labels).
 */
export const getGridLabeledSampleIds = async (
    trx: Knex.Transaction | Knex = knex
): Promise<Set<string>> => {
    const rows = await trx('grid_labels').select('sample_id');
    return new Set(rows.map(r => r.sample_id));
};
