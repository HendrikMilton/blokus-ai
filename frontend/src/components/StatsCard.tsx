import { AnalysisResult, PlayerColor, PLAYER_COLORS } from '../types/game';

interface StatsCardProps {
    stats: AnalysisResult['stats'];
    color: PlayerColor;
}

export const StatsCard = ({ stats, color }: StatsCardProps) => <div className="bg-surface rounded-xl p-4 border border-border">
    <div className="flex items-center gap-2 mb-4">
        <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: PLAYER_COLORS[color] }}
        />
        <span className="font-semibold text-foreground capitalize">{color} Stats</span>
    </div>

    <div className="space-y-3">
        <div className="flex justify-between">
            <span className="text-muted-foreground">Pieces used</span>
            <span className="font-medium text-foreground">{stats.piecesUsed}/21</span>
        </div>

        <div className="flex justify-between">
            <span className="text-muted-foreground">Pieces remaining</span>
            <span className="font-medium text-foreground">{stats.piecesRemaining}/21</span>
        </div>

        <div className="flex justify-between">
            <span className="text-muted-foreground">Win probability</span>
            <span className="font-medium text-foreground">{stats.winProbability}%</span>
        </div>

        <div className="mt-2 bg-background rounded-full h-2 overflow-hidden">
            <div
                className="h-full transition-all duration-500"
                style={{
                    width: `${stats.winProbability}%`,
                    backgroundColor: PLAYER_COLORS[color],
                }}
            />
        </div>
    </div>
</div>;
