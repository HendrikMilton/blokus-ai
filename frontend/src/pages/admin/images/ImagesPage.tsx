import { useRef, useState, useEffect, useCallback } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { http, API_URL } from '../../../api/config';
import { Button } from '../../../components/ui/Button';

type SampleStatus = 'unlabeled' | 'labeled' | 'verified';
type FilterStatus = SampleStatus | 'all';

const validFilters: FilterStatus[] = ['all', 'unlabeled', 'labeled', 'verified'];

interface Sample {
    id: string;
    filename: string;
    status: SampleStatus;
    created_at: string;
}

interface SamplesResponse {
    samples: Sample[];
}

interface UploadResponse {
    id: string;
    filename: string;
}

const statusColors: Record<SampleStatus, string> = {
    unlabeled: 'bg-warning/20 text-warning',
    labeled: 'bg-primary/20 text-primary',
    verified: 'bg-success/20 text-success',
};

const PlusIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
);

const BackIcon = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
    </svg>
);

export const ImagesPage = () => {
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [samples, setSamples] = useState<Sample[]>([]);

    // Get filter from URL or default to 'all'
    const urlFilter = searchParams.get('status') as FilterStatus | null;
    const filter: FilterStatus = urlFilter && validFilters.includes(urlFilter) ? urlFilter : 'all';

    const setFilter = (newFilter: FilterStatus) => {
        if (newFilter === 'all') {
            searchParams.delete('status');
        } else {
            searchParams.set('status', newFilter);
        }
        setSearchParams(searchParams);
    };

    const loadSamples = useCallback(async () => {
        try {
            const data = await http.get<SamplesResponse>('/api/ml/samples');
            setSamples(data.samples || []);
        } catch {
            // Error toast shown by http client
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadSamples();
    }, [loadSamples]);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) {
            return;
        }

        setIsUploading(true);

        try {
            const base64 = await new Promise<string>((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const result = reader.result as string;
                    const base64Data = result.split(',')[1];
                    resolve(base64Data);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });

            await http.post<UploadResponse>('/api/ml/upload', { image: base64 });
            toast.success('Image uploaded');
            loadSamples();
        } catch {
            // Error toast shown by http client
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    const filteredSamples = filter === 'all'
        ? samples
        : samples.filter(s => s.status === filter);

    const counts = {
        all: samples.length,
        unlabeled: samples.filter(s => s.status === 'unlabeled').length,
        labeled: samples.filter(s => s.status === 'labeled').length,
        verified: samples.filter(s => s.status === 'verified').length,
    };

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleDateString(undefined, {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Link
                        to="/admin"
                        className="p-1.5 -ml-1.5 rounded-lg hover:bg-surface-dim transition-colors text-secondary hover:text-foreground"
                    >
                        <BackIcon />
                    </Link>
                    <h1 className="text-xl font-medium text-foreground">Training Images</h1>
                    <span className="px-2 py-0.5 text-xs font-medium bg-surface-dim text-secondary rounded-full">
                        {samples.length}
                    </span>
                </div>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                />
                <Button
                    onClick={handleUploadClick}
                    loading={isUploading}
                    leftIcon={<PlusIcon />}
                >
                    Upload
                </Button>
            </div>

            {/* Filter Tabs */}
            <div className="flex gap-1">
                {(['all', 'unlabeled', 'labeled', 'verified'] as const).map((status) => (
                    <button
                        key={status}
                        onClick={() => setFilter(status)}
                        className={`
                            px-2.5 py-1 text-xs font-medium rounded-full transition-colors capitalize
                            ${filter === status
                        ? 'bg-primary text-white'
                        : 'bg-surface-dim text-secondary hover:text-foreground'
                    }
                        `}
                    >
                        {status} {counts[status] > 0 && `(${counts[status]})`}
                    </button>
                ))}
            </div>

            {/* Grid */}
            {isLoading ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                    {[...Array(10)].map((_, i) => (
                        <div key={i} className="bg-surface rounded-lg border border-border overflow-hidden">
                            <div className="aspect-square bg-surface-dim animate-pulse" />
                            <div className="p-2 space-y-1.5">
                                <div className="h-3 bg-surface-dim rounded animate-pulse w-2/3" />
                                <div className="h-2.5 bg-surface-dim rounded animate-pulse w-1/3" />
                            </div>
                        </div>
                    ))}
                </div>
            ) : filteredSamples.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                    <div className="w-16 h-16 bg-surface-dim rounded-full flex items-center justify-center mb-4">
                        <svg className="w-8 h-8 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                    </div>
                    <p className="text-secondary font-medium">No images found</p>
                    <p className="text-sm text-muted mt-1">Upload board images to get started</p>
                </div>
            ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                    {filteredSamples.map((sample) => (
                        <button
                            key={sample.id}
                            onClick={() => navigate(`/admin/images/${sample.id}`)}
                            className="bg-surface rounded-lg border border-border overflow-hidden text-left transition-shadow hover:shadow-gcp-hover focus:outline-none focus:ring-2 focus:ring-primary/30"
                        >
                            <div className="aspect-square bg-surface-dim">
                                <img
                                    src={`${API_URL}/api/ml/samples/${sample.id}/image`}
                                    alt={`Sample ${sample.id}`}
                                    className="w-full h-full object-cover"
                                    loading="lazy"
                                />
                            </div>
                            <div className="p-2">
                                <div className="flex items-center justify-between gap-2">
                                    <span className="text-xs text-muted truncate font-mono">
                                        {sample.id.slice(0, 8)}
                                    </span>
                                    <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded-full capitalize ${statusColors[sample.status]}`}>
                                        {sample.status}
                                    </span>
                                </div>
                                <p className="text-[11px] text-secondary mt-0.5">
                                    {formatDate(sample.created_at)}
                                </p>
                            </div>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
};
