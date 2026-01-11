import { ReactNode } from 'react';

export interface Column<T> {
    key: keyof T | string;
    header: string;
    render?: (row: T) => ReactNode;
    className?: string;
}

interface TableProps<T> {
    columns: Column<T>[];
    data: T[];
    keyField: keyof T;
    loading?: boolean;
    emptyMessage?: string;
    onRowClick?: (row: T) => void;
}

export function Table<T>({
    columns,
    data,
    keyField,
    loading = false,
    emptyMessage = 'No data available',
    onRowClick,
}: TableProps<T>) {
    const getCellValue = (row: T, column: Column<T>): ReactNode => {
        if (column.render) {
            return column.render(row);
        }
        const value = row[column.key as keyof T];
        if (value === null || value === undefined) {
            return '-';
        }
        if (value instanceof Date) {
            return value.toLocaleDateString();
        }
        return String(value);
    };

    return (
        <div className="w-full overflow-x-auto rounded-md border border-input">
            <table className="w-full text-sm">
                <thead className="bg-muted">
                    <tr>
                        {columns.map((column) => (
                            <th
                                key={String(column.key)}
                                className={`px-4 py-3 text-left font-medium text-foreground ${column.className || ''}`}
                            >
                                {column.header}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody className="divide-y divide-input">
                    {loading ? (
                        <tr>
                            <td colSpan={columns.length} className="px-4 py-8 text-center">
                                <div className="flex items-center justify-center gap-2">
                                    <svg
                                        className="animate-spin h-5 w-5 text-primary"
                                        xmlns="http://www.w3.org/2000/svg"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                    >
                                        <circle
                                            className="opacity-25"
                                            cx="12"
                                            cy="12"
                                            r="10"
                                            stroke="currentColor"
                                            strokeWidth="4"
                                        />
                                        <path
                                            className="opacity-75"
                                            fill="currentColor"
                                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                        />
                                    </svg>
                                    <span>Loading...</span>
                                </div>
                            </td>
                        </tr>
                    ) : data.length === 0 ? (
                        <tr>
                            <td colSpan={columns.length} className="px-4 py-8 text-center text-muted-foreground">
                                {emptyMessage}
                            </td>
                        </tr>
                    ) : (
                        data.map((row) => (
                            <tr
                                key={String(row[keyField])}
                                onClick={() => onRowClick?.(row)}
                                className={`
                                    bg-background hover:bg-muted/50 transition-colors
                                    ${onRowClick ? 'cursor-pointer' : ''}
                                `}
                            >
                                {columns.map((column) => (
                                    <td
                                        key={String(column.key)}
                                        className={`px-4 py-3 ${column.className || ''}`}
                                    >
                                        {getCellValue(row, column)}
                                    </td>
                                ))}
                            </tr>
                        ))
                    )}
                </tbody>
            </table>
        </div>
    );
}
