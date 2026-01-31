import type { Knex } from 'knex';
import { knex } from '../../knex';
import type { Tag } from '@blokus-ai/shared';

interface TagRow {
    id: string;
    name: string;
    created_at: Date;
}

const rowToTag = (row: TagRow): Tag => ({
    id: row.id,
    name: row.name,
    createdAt: row.created_at.toISOString(),
});

export const getAllTags = async (
    trx: Knex.Transaction | Knex = knex
): Promise<Tag[]> => {
    const rows = await trx<TagRow>('tags')
        .orderBy('name', 'asc');
    return rows.map(rowToTag);
};

export const createTag = async (
    name: string,
    trx: Knex.Transaction | Knex = knex
): Promise<Tag> => {
    const [row] = await trx<TagRow>('tags')
        .insert({ name })
        .returning('*');
    return rowToTag(row);
};

export const deleteTag = async (
    id: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('tags')
        .where('id', id)
        .delete();
    return deleted > 0;
};

export const getTagsForSample = async (
    sampleId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<Tag[]> => {
    const rows = await trx<TagRow>('tags as t')
        .join('sample_tags as st', 't.id', 'st.tag_id')
        .where('st.sample_id', sampleId)
        .select('t.*')
        .orderBy('t.name', 'asc');
    return rows.map(rowToTag);
};

export const addTagToSample = async (
    sampleId: string,
    tagId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<void> => {
    await trx('sample_tags')
        .insert({ sample_id: sampleId, tag_id: tagId })
        .onConflict(['sample_id', 'tag_id'])
        .ignore();
};

export const removeTagFromSample = async (
    sampleId: string,
    tagId: string,
    trx: Knex.Transaction | Knex = knex
): Promise<boolean> => {
    const deleted = await trx('sample_tags')
        .where('sample_id', sampleId)
        .where('tag_id', tagId)
        .delete();
    return deleted > 0;
};

/**
 * Get tags for multiple samples at once (batch query).
 * Returns a map of sampleId -> Tag[]
 */
export const getTagsForSamples = async (
    sampleIds: string[],
    trx: Knex.Transaction | Knex = knex
): Promise<Map<string, Tag[]>> => {
    if (sampleIds.length === 0) {
        return new Map();
    }

    const rows = await trx('sample_tags as st')
        .join('tags as t', 't.id', 'st.tag_id')
        .whereIn('st.sample_id', sampleIds)
        .select('st.sample_id', 't.id', 't.name', 't.created_at')
        .orderBy('t.name', 'asc');

    const result = new Map<string, Tag[]>();
    for (const row of rows) {
        const tag: Tag = {
            id: row.id,
            name: row.name,
            createdAt: row.created_at.toISOString(),
        };
        const existing = result.get(row.sample_id) || [];
        existing.push(tag);
        result.set(row.sample_id, existing);
    }
    return result;
};
