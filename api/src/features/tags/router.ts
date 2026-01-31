import { Router } from 'express';
import { asyncHandler } from '../../lib/async-handler';
import { BadRequestError, NotFoundError } from '../../error';
import {
    getAllTags,
    createTag,
    deleteTag,
    getTagsForSample,
    addTagToSample,
    removeTagFromSample,
} from './database';

export const tagsRouter = Router();

// GET /api/tags - List all tags
tagsRouter.get('/', asyncHandler(async (_req, res) => {
    const tags = await getAllTags();
    res.json({ tags });
}));

// POST /api/tags - Create a new tag
tagsRouter.post('/', asyncHandler(async (req, res) => {
    const { name } = req.body as { name: string };

    if (!name || typeof name !== 'string' || name.trim().length === 0) {
        throw new BadRequestError('name is required');
    }

    const tag = await createTag(name.trim());
    res.json(tag);
}));

// DELETE /api/tags/:id - Delete a tag
tagsRouter.delete('/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;

    const deleted = await deleteTag(id);
    if (!deleted) {
        throw new NotFoundError('Tag not found');
    }

    res.json({ success: true });
}));

// GET /api/tags/samples/:id - Get tags for a sample
tagsRouter.get('/samples/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const tags = await getTagsForSample(id);
    res.json({ tags });
}));

// POST /api/tags/samples/:id - Add tag to sample
tagsRouter.post('/samples/:id', asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { tagId } = req.body as { tagId: string };

    if (!tagId) {
        throw new BadRequestError('tagId is required');
    }

    await addTagToSample(id, tagId);
    res.json({ success: true });
}));

// DELETE /api/tags/samples/:id/:tagId - Remove tag from sample
tagsRouter.delete('/samples/:id/:tagId', asyncHandler(async (req, res) => {
    const { id, tagId } = req.params;

    const removed = await removeTagFromSample(id, tagId);
    if (!removed) {
        throw new NotFoundError('Tag assignment not found');
    }

    res.json({ success: true });
}));
