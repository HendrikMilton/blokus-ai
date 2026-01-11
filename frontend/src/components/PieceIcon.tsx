interface PieceIconProps {
    className?: string;
}

export function PieceIcon({ className = '' }: PieceIconProps) {
    return (
        <svg
            viewBox="0 0 40 40"
            className={className}
            fill="currentColor"
            opacity={0.3}
        >
            {/* L-piece shape */}
            <rect x="0" y="0" width="10" height="10" rx="1" />
            <rect x="0" y="10" width="10" height="10" rx="1" />
            <rect x="0" y="20" width="10" height="10" rx="1" />
            <rect x="10" y="20" width="10" height="10" rx="1" />
        </svg>
    );
}
