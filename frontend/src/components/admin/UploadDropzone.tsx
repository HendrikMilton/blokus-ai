/**
 * UploadDropzone - Drag-and-drop file upload component
 */

import { useCallback, useState, useRef } from 'react';

interface UploadDropzoneProps {
    onUpload: (file: File) => void | Promise<void>;
    accept?: string;
    maxSize?: number; // in bytes
    disabled?: boolean;
}

export const UploadDropzone = ({
    onUpload,
    accept = 'image/*',
    maxSize = 10 * 1024 * 1024, // 10MB default
    disabled = false,
}: UploadDropzoneProps) => {
    const [isDragOver, setIsDragOver] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFile = useCallback(async (file: File) => {
        setError(null);

        // Validate file type
        if (accept && !file.type.match(accept.replace('*', '.*'))) {
            setError('Invalid file type. Please upload an image.');
            return;
        }

        // Validate file size
        if (file.size > maxSize) {
            const maxSizeMB = (maxSize / (1024 * 1024)).toFixed(1);
            setError(`File too large. Maximum size is ${maxSizeMB}MB.`);
            return;
        }

        setIsUploading(true);
        try {
            await onUpload(file);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Upload failed');
        } finally {
            setIsUploading(false);
        }
    }, [onUpload, accept, maxSize]);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (!disabled) {
            setIsDragOver(true);
        }
    }, [disabled]);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragOver(false);

        if (disabled) {
            return;
        }

        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }, [disabled, handleFile]);

    const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
        // Reset input so the same file can be selected again
        e.target.value = '';
    }, [handleFile]);

    const handleClick = useCallback(() => {
        if (!disabled && inputRef.current) {
            inputRef.current.click();
        }
    }, [disabled]);

    return (
        <div
            onClick={handleClick}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`
                relative flex flex-col items-center justify-center
                p-8 border-2 border-dashed rounded-xl
                transition-all duration-200 cursor-pointer
                ${disabled
            ? 'bg-surface/50 border-border/50 cursor-not-allowed'
            : isDragOver
                ? 'bg-primary/10 border-primary'
                : 'bg-surface hover:bg-surface/80 border-border hover:border-primary/50'
        }
            `}
        >
            <input
                ref={inputRef}
                type="file"
                accept={accept}
                onChange={handleInputChange}
                className="hidden"
                disabled={disabled}
            />

            {isUploading ? (
                <div className="flex flex-col items-center gap-3">
                    <div className="w-10 h-10 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                    <p className="text-sm text-muted-foreground">Uploading...</p>
                </div>
            ) : (
                <>
                    <div className={`
                        w-12 h-12 mb-3 rounded-full flex items-center justify-center
                        ${isDragOver ? 'bg-primary/20' : 'bg-surface'}
                    `}>
                        <svg
                            className={`w-6 h-6 ${isDragOver ? 'text-primary' : 'text-muted-foreground'}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={1.5}
                                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                            />
                        </svg>
                    </div>

                    <p className="text-sm font-medium text-foreground mb-1">
                        {isDragOver ? 'Drop to upload' : 'Drop image here'}
                    </p>
                    <p className="text-xs text-muted-foreground">
                        or click to browse
                    </p>

                    {/* Mobile camera option */}
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            if (inputRef.current) {
                                inputRef.current.setAttribute('capture', 'environment');
                                inputRef.current.click();
                                inputRef.current.removeAttribute('capture');
                            }
                        }}
                        className="mt-3 px-3 py-1.5 text-xs bg-primary/10 text-primary rounded-md
                            hover:bg-primary/20 transition-colors md:hidden"
                    >
                        Take Photo
                    </button>
                </>
            )}

            {error && (
                <div className="absolute bottom-2 left-2 right-2 p-2 bg-red-500/10 border border-red-500/30 rounded-md">
                    <p className="text-xs text-red-400 text-center">{error}</p>
                </div>
            )}
        </div>
    );
};
