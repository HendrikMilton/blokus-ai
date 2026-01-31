import type { Knex } from 'knex';
import { knex } from '../../knex';
import type { ModelPipeline, TrainingRun } from '@blokus-ai/shared';

interface ModelPipelineRow {
    id: string;
    name: string;
    component_ids: string[];
    created_at: Date;
    updated_at: Date;
}

interface TrainingRunRow {
    id: string;
    status: string;
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
    created_at: Date;
}

const rowToModelPipeline = (row: ModelPipelineRow): ModelPipeline => ({
    id: row.id,
    name: row.name,
    componentIds: row.component_ids,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
});

const rowToTrainingRun = (row: TrainingRunRow): TrainingRun => ({
    id: row.id,
    status: row.status as TrainingRun['status'],
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
    createdAt: row.created_at.toISOString(),
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
        .map(cid => componentsMap.get(cid))
        .filter((c): c is TrainingRun => c !== undefined);

    return {
        ...pipeline,
        components,
    };
};

// Get training run (needed for validation)
export const getTrainingRun = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<TrainingRun | null> => {
    const row = await trx<TrainingRunRow>('training_runs')
        .where('id', id)
        .first();
    return row ? rowToTrainingRun(row) : null;
};
