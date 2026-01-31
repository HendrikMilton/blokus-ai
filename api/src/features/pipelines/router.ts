import { Router } from 'express';
import { asyncHandler } from '../../lib/async-handler';
import { BadRequestError, NotFoundError } from '../../error';
import { log } from '../../lib/log';
import {
    createPipeline,
    listPipelines,
    getPipeline,
    getPipelineWithComponents,
    updatePipeline,
    deletePipeline,
    getTrainingRun,
} from './database';

export const pipelinesRouter = Router();

// POST /api/pipelines - Create a new pipeline
pipelinesRouter.post('/', asyncHandler(async (req, res) => {
    const { name, componentIds } = req.body as { name: string; componentIds: string[] };

    if (!name || typeof name !== 'string' || name.trim().length === 0) {
        throw new BadRequestError('name is required');
    }
    if (!componentIds || !Array.isArray(componentIds) || componentIds.length === 0) {
        throw new BadRequestError('componentIds must be a non-empty array');
    }

    // Validate that all component IDs are valid completed training runs
    for (const id of componentIds) {
        const run = await getTrainingRun(id);
        if (!run) {
            throw new BadRequestError(`Training run not found: ${id}`);
        }
        if (run.status !== 'completed') {
            throw new BadRequestError(`Training run is not completed: ${id}`);
        }
    }

    const pipeline = await createPipeline(name.trim(), componentIds);
    res.json(pipeline);
}));

// GET /api/pipelines - List all pipelines
pipelinesRouter.get('/', asyncHandler(async (_req, res) => {
    const pipelines = await listPipelines();
    res.json({ pipelines });
}));

// GET /api/pipelines/:id - Get single pipeline with component details
pipelinesRouter.get('/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const pipeline = await getPipelineWithComponents(id);
    if (!pipeline) {
        throw new NotFoundError('Pipeline not found');
    }

    res.json(pipeline);
}));

// PUT /api/pipelines/:id - Update a pipeline
pipelinesRouter.put('/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { name, componentIds } = req.body as { name?: string; componentIds?: string[] };

    const existing = await getPipeline(id);
    if (!existing) {
        throw new NotFoundError('Pipeline not found');
    }

    // Validate component IDs if provided
    if (componentIds) {
        if (!Array.isArray(componentIds) || componentIds.length === 0) {
            throw new BadRequestError('componentIds must be a non-empty array');
        }
        for (const compId of componentIds) {
            const run = await getTrainingRun(compId);
            if (!run) {
                throw new BadRequestError(`Training run not found: ${compId}`);
            }
            if (run.status !== 'completed') {
                throw new BadRequestError(`Training run is not completed: ${compId}`);
            }
        }
    }

    const updated = await updatePipeline(id, {
        name: name?.trim(),
        componentIds,
    });

    res.json(updated);
}));

// DELETE /api/pipelines/:id - Delete a pipeline
pipelinesRouter.delete('/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const existing = await getPipeline(id);
    if (!existing) {
        throw new NotFoundError('Pipeline not found');
    }

    await deletePipeline(id);
    log.info(`Deleted pipeline ${id}`);
    res.json({ success: true });
}));

// Re-export for use by other features
export { getPipeline } from './database';
