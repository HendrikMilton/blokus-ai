import { ReactNode, useState, useEffect, useMemo } from 'react';
import { ChevronUp, ChevronDown, Search, X, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { cn } from '../../utils/cn';

export interface Column<T> {
    key: string;
    header: string;
    sortable?: boolean;
    render?: (item: T) => ReactNode;
    className?: string;
    headerClassName?: string;
    width?: string;
    noTruncate?: boolean;
}

export type SortDirection = 'asc' | 'desc' | null;

export interface SortState {
    column: string | null;
    direction: SortDirection;
}

interface TableProps<T> {
    columns: Column<T>[];
    data: T[];
    onRowClick?: (item: T) => void;
    sortState?: SortState;
    onSort?: (column: string) => void;
    emptyMessage?: string | ReactNode;
    className?: string;
    getRowKey?: (item: T) => string | number;
    getRowClassName?: (item: T) => string;
    loading?: boolean;
    // Search props
    searchable?: boolean;
    searchValue?: string;
    onSearchChange?: (value: string) => void;
    searchPlaceholder?: string;
    // Header content (rendered between search and table)
    headerContent?: ReactNode;
    // Pagination props
    paginated?: boolean;
    currentPage?: number;
    pageSize?: number;
    totalItems?: number;
    onPageChange?: (page: number) => void;
    // Selection props (Google Drive style - click to select, double-click for action)
    selectable?: boolean;
    selectedKeys?: Set<string | number>;
    onSelectionChange?: (key: string | number, shiftKey: boolean, ctrlKey: boolean) => void;
    onRowDoubleClick?: (item: T) => void;
}

export function Table<T>({
    columns,
    data,
    onRowClick,
    sortState,
    onSort,
    emptyMessage = 'No data available',
    className,
    getRowKey = (item: unknown) => (item as { id: string | number }).id,
    getRowClassName,
    loading = false,
    searchable = false,
    searchValue = '',
    onSearchChange,
    searchPlaceholder = 'Search...',
    headerContent,
    paginated = false,
    currentPage: externalCurrentPage,
    pageSize = 10,
    totalItems: externalTotalItems,
    onPageChange: externalOnPageChange,
    selectable = false,
    selectedKeys,
    onSelectionChange,
    onRowDoubleClick
}: TableProps<T>) {
    // Internal pagination state (used when not controlled)
    const [internalCurrentPage, setInternalCurrentPage] = useState(1);

    // Use external or internal state
    const currentPage = externalCurrentPage ?? internalCurrentPage;
    const onPageChange = externalOnPageChange ?? setInternalCurrentPage;

    // Reset to page 1 when data changes (filters/search applied)
    useEffect(() => {
        if (paginated && !externalCurrentPage) {
            setInternalCurrentPage(1);
        }
    }, [data.length, paginated, externalCurrentPage]);

    // Paginate data if enabled
    const displayData = useMemo(() => {
        if (!paginated) {
            return data;
        }
        const startIndex = (currentPage - 1) * pageSize;
        const endIndex = startIndex + pageSize;
        return data.slice(startIndex, endIndex);
    }, [data, paginated, currentPage, pageSize]);

    const totalItems = externalTotalItems ?? data.length;

    const handleSort = (column: Column<T>) => {
        if (column.sortable && onSort) {
            onSort(column.key);
        }
    };

    return (
        <div className={cn('w-full rounded-md border border-border overflow-hidden', className)}>
            {/* Search Bar */}
            {searchable && (
                <div className="p-4 border-b border-border bg-surface">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-secondary" />
                        <input
                            type="text"
                            placeholder={searchPlaceholder}
                            value={searchValue}
                            onChange={(e) => onSearchChange?.(e.target.value)}
                            className="w-full pl-10 pr-10 py-2 bg-background border border-border rounded-lg text-sm text-foreground placeholder:text-secondary focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                        />
                        {searchValue && (
                            <button
                                onClick={() => onSearchChange?.('')}
                                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-secondary hover:text-foreground"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </div>
            )}

            {/* Header Content */}
            {headerContent}

            {/* Table */}
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead className="bg-surface-dim">
                        <tr>
                            {columns.map((column) => (
                                <th
                                    key={column.key}
                                    className={cn(
                                        'text-left py-3 px-4 font-medium text-foreground whitespace-nowrap',
                                        column.sortable && 'cursor-pointer select-none transition-colors hover:bg-border/30',
                                        column.headerClassName
                                    )}
                                    style={column.width ? { width: column.width, minWidth: column.width } : undefined}
                                    onClick={() => column.sortable && handleSort(column)}
                                >
                                    <div className={cn(
                                        'flex items-center gap-1.5',
                                        column.headerClassName?.includes('text-center') && 'justify-center'
                                    )}>
                                        <span>{column.header}</span>
                                        {column.sortable && (
                                            <div className="flex flex-col items-center justify-center flex-shrink-0">
                                                {sortState?.column === column.key && sortState.direction ? (
                                                    sortState.direction === 'asc' ? (
                                                        <ChevronUp className="w-4 h-4" />
                                                    ) : (
                                                        <ChevronDown className="w-4 h-4" />
                                                    )
                                                ) : (
                                                    <div className="opacity-30 flex flex-col items-center justify-center -space-y-1.5">
                                                        <ChevronUp className="w-3 h-3" />
                                                        <ChevronDown className="w-3 h-3" />
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                        {loading ? (
                            <tr>
                                <td colSpan={columns.length} className="px-4 py-8 text-center">
                                    <div className="flex items-center justify-center gap-2">
                                        <Loader2 className="animate-spin h-5 w-5 text-primary" />
                                        <span>Loading...</span>
                                    </div>
                                </td>
                            </tr>
                        ) : displayData.length === 0 ? (
                            <tr>
                                <td colSpan={columns.length} className="px-4 py-8 text-center text-muted">
                                    {typeof emptyMessage === 'string' ? (
                                        <p className="text-sm text-secondary">{emptyMessage}</p>
                                    ) : (
                                        emptyMessage
                                    )}
                                </td>
                            </tr>
                        ) : (
                            displayData.map((item) => {
                                const rowKey = getRowKey(item);
                                const isSelected = selectable && selectedKeys?.has(rowKey);
                                return (
                                    <tr
                                        key={rowKey}
                                        onClick={(e) => {
                                            if (selectable && onSelectionChange) {
                                                onSelectionChange(rowKey, e.shiftKey, e.ctrlKey || e.metaKey);
                                            } else if (onRowClick) {
                                                onRowClick(item);
                                            }
                                        }}
                                        onDoubleClick={() => onRowDoubleClick?.(item)}
                                        className={cn(
                                            'bg-surface transition-colors',
                                            (onRowClick || selectable || onRowDoubleClick) && 'cursor-pointer select-none',
                                            isSelected
                                                ? 'bg-primary/10 hover:bg-primary/15'
                                                : 'hover:bg-surface-dim',
                                            getRowClassName?.(item)
                                        )}
                                    >
                                        {columns.map((column) => {
                                            const content = column.render
                                                ? column.render(item)
                                                : (item as Record<string, unknown>)[column.key];

                                            const displayContent = content === null || content === undefined || content === ''
                                                ? '-'
                                                : content;

                                            return (
                                                <td
                                                    key={column.key}
                                                    className={cn(
                                                        'py-3 px-4 text-foreground',
                                                        column.className
                                                    )}
                                                    style={column.width ? { width: column.width, minWidth: column.width } : undefined}
                                                >
                                                    <div className={column.noTruncate ? '' : 'truncate'}>
                                                        {displayContent as ReactNode}
                                                    </div>
                                                </td>
                                            );
                                        })}
                                    </tr>
                                );
                            })
                        )}
                    </tbody>
                </table>
            </div>

            {/* Pagination */}
            {paginated && totalItems > pageSize && (
                <div className="flex items-center justify-between px-4 py-3 border-t border-border bg-surface">
                    <div className="text-xs text-secondary">
                        {displayData.length === 0 ? 0 : ((currentPage - 1) * pageSize) + 1}–{Math.min(currentPage * pageSize, totalItems)} of {totalItems}
                    </div>
                    <div className="flex items-center gap-1">
                        <button
                            onClick={() => onPageChange?.(currentPage - 1)}
                            disabled={currentPage === 1}
                            className={cn(
                                'p-1.5 rounded transition-colors text-secondary',
                                currentPage === 1
                                    ? 'opacity-30 cursor-not-allowed'
                                    : 'hover:bg-surface-dim hover:text-foreground'
                            )}
                        >
                            <ChevronLeft className="w-4 h-4" />
                        </button>
                        <div className="flex items-center gap-0.5">
                            {(() => {
                                const totalPages = Math.ceil(totalItems / pageSize);
                                const pages = [];
                                const maxVisible = 5;

                                let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
                                const endPage = Math.min(totalPages, startPage + maxVisible - 1);

                                if (endPage - startPage < maxVisible - 1) {
                                    startPage = Math.max(1, endPage - maxVisible + 1);
                                }

                                for (let i = startPage; i <= endPage; i++) {
                                    pages.push(
                                        <button
                                            key={i}
                                            onClick={() => onPageChange?.(i)}
                                            className={cn(
                                                'min-w-[28px] h-7 px-2 rounded text-xs font-medium transition-colors',
                                                i === currentPage
                                                    ? 'bg-primary text-white'
                                                    : 'text-secondary hover:bg-surface-dim hover:text-foreground'
                                            )}
                                        >
                                            {i}
                                        </button>
                                    );
                                }
                                return pages;
                            })()}
                        </div>
                        <button
                            onClick={() => onPageChange?.(currentPage + 1)}
                            disabled={currentPage >= Math.ceil(totalItems / pageSize)}
                            className={cn(
                                'p-1.5 rounded transition-colors text-secondary',
                                currentPage >= Math.ceil(totalItems / pageSize)
                                    ? 'opacity-30 cursor-not-allowed'
                                    : 'hover:bg-surface-dim hover:text-foreground'
                            )}
                        >
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
