import { useState, useEffect } from 'react';
import { X, Tag as TagIcon } from 'lucide-react';
import type { Tag } from '@blokus-ai/shared';
import { ML_API } from '../../../../api/ml';
import { Select } from '../../../../components/ui/Select';

interface Props {
    sampleId: string;
    allTags: Tag[];
}

export const SampleTagsSection = ({ sampleId, allTags }: Props) => {
    const [sampleTags, setSampleTags] = useState<Tag[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isAdding, setIsAdding] = useState(false);
    const [removingTagId, setRemovingTagId] = useState<string | null>(null);

    useEffect(() => {
        const loadTags = async () => {
            setIsLoading(true);
            try {
                const tags = await ML_API.getTagsForSample(sampleId);
                setSampleTags(tags);
            } catch {
                // Error toast shown by http client
            } finally {
                setIsLoading(false);
            }
        };
        loadTags();
    }, [sampleId]);

    const handleAddTag = async (tagId: string) => {
        if (!tagId) {
            return;
        }
        setIsAdding(true);
        try {
            await ML_API.addTagToSample(sampleId, tagId);
            const tag = allTags.find(t => t.id === tagId);
            if (tag) {
                setSampleTags(prev => [...prev, tag].sort((a, b) => a.name.localeCompare(b.name)));
            }
        } catch {
            // Error toast shown by http client
        } finally {
            setIsAdding(false);
        }
    };

    const handleRemoveTag = async (tagId: string) => {
        setRemovingTagId(tagId);
        try {
            await ML_API.removeTagFromSample(sampleId, tagId);
            setSampleTags(prev => prev.filter(t => t.id !== tagId));
        } catch {
            // Error toast shown by http client
        } finally {
            setRemovingTagId(null);
        }
    };

    const availableTags = allTags.filter(t => !sampleTags.some(st => st.id === t.id));

    return (
        <div className="bg-surface rounded-lg border border-border p-4">
            <h2 className="text-sm font-medium text-foreground mb-3">Tags</h2>

            {isLoading ? (
                <div className="animate-pulse">
                    <div className="h-8 bg-surface-dim rounded w-1/2" />
                </div>
            ) : (
                <div className="space-y-3">
                    {/* Current tags */}
                    {sampleTags.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                            {sampleTags.map(tag => (
                                <span
                                    key={tag.id}
                                    className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-primary/10 text-primary rounded-full text-sm"
                                >
                                    <TagIcon className="w-3.5 h-3.5" />
                                    {tag.name}
                                    <button
                                        onClick={() => handleRemoveTag(tag.id)}
                                        disabled={removingTagId === tag.id}
                                        className="p-0.5 hover:bg-primary/20 rounded-full transition-colors disabled:opacity-50"
                                        title="Remove tag"
                                    >
                                        <X className="w-3.5 h-3.5" />
                                    </button>
                                </span>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm text-muted">No tags assigned</p>
                    )}

                    {/* Add tag dropdown */}
                    {availableTags.length > 0 && (
                        <div className="flex items-center gap-2">
                            <Select
                                value=""
                                onChange={(v) => handleAddTag(String(v))}
                                options={[
                                    { value: '', label: 'Add a tag...' },
                                    ...availableTags.map(tag => ({
                                        value: tag.id,
                                        label: tag.name,
                                    })),
                                ]}
                                className="w-48"
                                disabled={isAdding}
                            />
                            {isAdding && (
                                <span className="text-xs text-muted">Adding...</span>
                            )}
                        </div>
                    )}

                    {allTags.length === 0 && (
                        <p className="text-xs text-muted">
                            No tags available. Create tags in the Configurations tab.
                        </p>
                    )}
                </div>
            )}
        </div>
    );
};
