import { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';

export interface SelectOption {
    value: string | number;
    label: string;
}

interface SelectProps {
    label?: string;
    value: string | number | undefined;
    options: SelectOption[];
    onChange: (value: string | number) => void;
    placeholder?: string;
    searchable?: boolean;
    error?: string;
    touched?: boolean;
    disabled?: boolean;
    className?: string;
}

export const Select = ({
    label,
    value,
    options,
    onChange,
    placeholder = 'Select...',
    searchable = false,
    error,
    touched,
    disabled = false,
    className = '',
}: SelectProps) => {
    const [isOpen, setIsOpen] = useState(false);
    const [search, setSearch] = useState('');
    const containerRef = useRef<HTMLDivElement>(null);
    const showError = touched && error;

    const selectedOption = options.find((opt) => opt.value === value);

    const filteredOptions = searchable
        ? options.filter((opt) =>
            opt.label.toLowerCase().includes(search.toLowerCase())
        )
        : options;

    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
                setIsOpen(false);
                setSearch('');
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className={className || 'w-full'} ref={containerRef}>
            {label && (
                <label className="block text-sm font-medium text-foreground mb-1">
                    {label}
                </label>
            )}
            <div className="relative">
                <button
                    type="button"
                    onClick={() => !disabled && setIsOpen(!isOpen)}
                    disabled={disabled}
                    className={`
                        w-full flex items-center justify-between rounded-md border bg-surface
                        px-3 py-2 text-sm text-left
                        focus:outline-none focus:ring-2 focus:ring-primary/30
                        disabled:cursor-not-allowed disabled:opacity-50
                        ${showError ? 'border-error' : 'border-border'}
                    `}
                >
                    <span className={selectedOption ? 'text-foreground' : 'text-muted'}>
                        {selectedOption?.label || placeholder}
                    </span>
                    <ChevronDown className={`h-4 w-4 text-secondary transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                </button>

                {isOpen && (
                    <div className="absolute z-50 w-full mt-1 bg-surface border border-border rounded-md shadow-gcp-lg max-h-60 overflow-auto">
                        {searchable && (
                            <div className="p-2 border-b border-border">
                                <input
                                    type="text"
                                    value={search}
                                    onChange={(e) => setSearch(e.target.value)}
                                    placeholder="Search..."
                                    className="w-full px-2 py-1 text-sm border border-border rounded focus:outline-none focus:ring-1 focus:ring-primary/30"
                                    autoFocus
                                />
                            </div>
                        )}
                        {filteredOptions.length === 0 ? (
                            <div className="px-3 py-2 text-sm text-muted">
                                No options found
                            </div>
                        ) : (
                            filteredOptions.map((option) => (
                                <button
                                    key={option.value}
                                    type="button"
                                    onClick={() => {
                                        onChange(option.value);
                                        setIsOpen(false);
                                        setSearch('');
                                    }}
                                    className={`
                                        w-full px-3 py-2 text-sm text-left text-foreground hover:bg-surface-dim
                                        ${option.value === value ? 'bg-surface-dim text-primary' : ''}
                                    `}
                                >
                                    {option.label}
                                </button>
                            ))
                        )}
                    </div>
                )}
            </div>
            {showError && (
                <p className="mt-1 text-xs text-error">{error}</p>
            )}
        </div>
    );
};
