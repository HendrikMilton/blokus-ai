import { useState, useRef, useEffect } from 'react';
import { Check, ChevronDown, Search } from 'lucide-react';

export interface MultiSelectOption {
    value: string;
    label: string;
}

interface MultiSelectProps {
    label?: string;
    options: MultiSelectOption[];
    selected: string[];
    onChange: (selected: string[]) => void;
    placeholder?: string;
    allLabel?: string;
    searchable?: boolean;
    disabled?: boolean;
    className?: string;
}

export const MultiSelect = ({
    label,
    options,
    selected,
    onChange,
    placeholder = 'Select...',
    allLabel = 'All',
    searchable = false,
    disabled = false,
    className,
}: MultiSelectProps) => {
    const [isOpen, setIsOpen] = useState(false);
    const [search, setSearch] = useState('');
    const containerRef = useRef<HTMLDivElement>(null);
    const searchInputRef = useRef<HTMLInputElement>(null);

    const filteredOptions = searchable
        ? options.filter(opt => opt.label.toLowerCase().includes(search.toLowerCase()))
        : options;

    const allSelected = selected.length === options.length && options.length > 0;
    const noneSelected = selected.length === 0;

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

    useEffect(() => {
        if (isOpen && searchable && searchInputRef.current) {
            searchInputRef.current.focus();
        }
    }, [isOpen, searchable]);

    const toggleOption = (value: string) => {
        if (selected.includes(value)) {
            onChange(selected.filter(s => s !== value));
        } else {
            onChange([...selected, value]);
        }
    };

    const toggleAll = () => {
        if (allSelected) {
            onChange([]);
        } else {
            onChange(options.map(o => o.value));
        }
    };

    const getDisplayText = () => {
        if (noneSelected) {
            return placeholder;
        }
        if (allSelected) {
            return allLabel;
        }
        const selectedLabels = options.filter(o => selected.includes(o.value)).map(o => o.label);
        if (selectedLabels.length <= 2) {
            return selectedLabels.join(', ');
        }
        return `${selectedLabels.length} selected`;
    };

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
                        px-3 py-2 text-sm text-left border-border
                        focus:outline-none focus:ring-2 focus:ring-primary/30
                        disabled:cursor-not-allowed disabled:opacity-50
                    `}
                >
                    <span className={noneSelected ? 'text-muted' : 'text-foreground'}>
                        {getDisplayText()}
                    </span>
                    <ChevronDown className={`h-4 w-4 text-secondary transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                </button>

                {isOpen && (
                    <div className="absolute z-50 w-full mt-1 bg-surface border border-border rounded-md shadow-gcp-lg max-h-60 overflow-hidden flex flex-col">
                        {/* Search Input */}
                        {searchable && (
                            <div className="p-2 border-b border-border">
                                <div className="relative">
                                    <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
                                    <input
                                        ref={searchInputRef}
                                        type="text"
                                        value={search}
                                        onChange={(e) => setSearch(e.target.value)}
                                        placeholder="Search..."
                                        className="w-full pl-8 pr-3 py-1.5 text-sm border border-border rounded bg-surface text-foreground focus:outline-none focus:ring-1 focus:ring-primary/30"
                                        onClick={(e) => e.stopPropagation()}
                                    />
                                </div>
                            </div>
                        )}

                        <div className="overflow-y-auto max-h-52">
                            {/* Select All Option */}
                            {!search && options.length > 1 && (
                                <button
                                    type="button"
                                    onClick={toggleAll}
                                    className="w-full px-3 py-2 text-sm text-left hover:bg-surface-dim flex items-center justify-between border-b border-border"
                                >
                                    <span className="font-medium text-foreground">{allLabel}</span>
                                    <div className={`w-4 h-4 rounded border-2 flex items-center justify-center ${
                                        allSelected ? 'bg-primary border-primary' : 'border-border'
                                    }`}>
                                        {allSelected && <Check className="w-3 h-3 text-white" />}
                                    </div>
                                </button>
                            )}

                            {/* Individual Options */}
                            {filteredOptions.length > 0 ? (
                                filteredOptions.map(option => (
                                    <button
                                        key={option.value}
                                        type="button"
                                        onClick={() => toggleOption(option.value)}
                                        className="w-full px-3 py-2 text-sm text-left hover:bg-surface-dim flex items-center justify-between"
                                    >
                                        <span className="text-foreground">{option.label}</span>
                                        <div className={`w-4 h-4 rounded border-2 flex items-center justify-center ${
                                            selected.includes(option.value) ? 'bg-primary border-primary' : 'border-border'
                                        }`}>
                                            {selected.includes(option.value) && <Check className="w-3 h-3 text-white" />}
                                        </div>
                                    </button>
                                ))
                            ) : (
                                <div className="px-3 py-2 text-sm text-muted text-center">
                                    No options found
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
