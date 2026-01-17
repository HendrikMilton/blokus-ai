/**
 * StatsPanel - Display labeling progress statistics
 */

import { AdminStats, PLAYER_COLORS } from '@blokus-ai/shared';

interface StatsPanelProps {
    stats: AdminStats | null;
    loading?: boolean;
}

function StatCard({
    label,
    value,
    color,
}: {
    label: string;
    value: number;
    color?: string;
}) {
    return (
        <div className="flex flex-col p-3 bg-surface rounded-lg border border-border">
            <span className="text-xs text-muted-foreground mb-1">{label}</span>
            <span
                className="text-2xl font-bold"
                style={{ color: color || 'inherit' }}
            >
                {value}
            </span>
        </div>
    );
}

function ProgressBar({
    labeled,
    verified,
    total,
}: {
    labeled: number;
    verified: number;
    total: number;
}) {
    if (total === 0) {
        return null;
    }

    const labeledPercent = ((labeled + verified) / total) * 100;
    const verifiedPercent = (verified / total) * 100;

    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
                <span>Labeling Progress</span>
                <span>{Math.round(labeledPercent)}%</span>
            </div>
            <div className="h-2 bg-surface rounded-full overflow-hidden">
                <div
                    className="h-full bg-blue-500 transition-all duration-300"
                    style={{ width: `${labeledPercent}%` }}
                >
                    <div
                        className="h-full bg-green-500 transition-all duration-300"
                        style={{ width: `${total > 0 ? (verifiedPercent / labeledPercent) * 100 : 0}%` }}
                    />
                </div>
            </div>
            <div className="flex gap-4 text-xs">
                <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-green-500" />
                    Verified
                </span>
                <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-blue-500" />
                    Labeled
                </span>
                <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-surface border border-border" />
                    Pending
                </span>
            </div>
        </div>
    );
}

function ColorDistribution({ distribution }: { distribution: Record<string, number> }) {
    const colors = ['blue', 'yellow', 'red', 'green'] as const;
    const total = colors.reduce((sum, c) => sum + (distribution[c] || 0), 0);

    if (total === 0) {
        return null;
    }

    return (
        <div className="space-y-2">
            <span className="text-xs text-muted-foreground">Color Distribution</span>
            <div className="flex gap-1 h-4 rounded overflow-hidden">
                {colors.map((color) => {
                    const count = distribution[color] || 0;
                    const percent = (count / total) * 100;
                    if (percent === 0) {
                        return null;
                    }

                    return (
                        <div
                            key={color}
                            className="transition-all duration-300"
                            style={{
                                width: `${percent}%`,
                                backgroundColor: PLAYER_COLORS[color],
                            }}
                            title={`${color}: ${count} cells (${Math.round(percent)}%)`}
                        />
                    );
                })}
            </div>
            <div className="grid grid-cols-4 gap-2 text-xs">
                {colors.map((color) => (
                    <div key={color} className="flex items-center gap-1">
                        <span
                            className="w-2 h-2 rounded-sm"
                            style={{ backgroundColor: PLAYER_COLORS[color] }}
                        />
                        <span className="capitalize">{distribution[color] || 0}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function LoadingSkeleton() {
    return (
        <div className="space-y-4 animate-pulse">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="p-3 bg-surface rounded-lg border border-border">
                        <div className="h-3 w-16 bg-muted rounded mb-2" />
                        <div className="h-8 w-12 bg-muted rounded" />
                    </div>
                ))}
            </div>
            <div className="h-2 bg-surface rounded-full" />
        </div>
    );
}

export function StatsPanel({ stats, loading }: StatsPanelProps) {
    if (loading || !stats) {
        return <LoadingSkeleton />;
    }

    return (
        <div className="space-y-4">
            {/* Main stats */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <StatCard label="Total Samples" value={stats.total_samples} />
                <StatCard
                    label="Pending"
                    value={stats.pending_samples}
                    color="#eab308"
                />
                <StatCard
                    label="Labeled"
                    value={stats.labeled_samples}
                    color="#3b82f6"
                />
                <StatCard
                    label="Verified"
                    value={stats.verified_samples}
                    color="#22c55e"
                />
            </div>

            {/* Progress bar */}
            <ProgressBar
                labeled={stats.labeled_samples}
                verified={stats.verified_samples}
                total={stats.total_samples}
            />

            {/* Color distribution */}
            {stats.total_cells_labeled > 0 && (
                <ColorDistribution distribution={stats.color_distribution} />
            )}
        </div>
    );
}
