/**
 * Format a date string to short format (e.g., "Jan 29")
 */
export const formatShortDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
};

/**
 * Format a UUID to short format (first 8 characters)
 */
export const formatShortId = (id: string): string => {
    return id.slice(0, 8);
};
