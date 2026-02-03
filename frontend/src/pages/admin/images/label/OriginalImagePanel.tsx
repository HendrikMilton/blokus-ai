import { useRef, useCallback } from 'react';
import type { BoundingBox, BoardCorners } from '@blokus-ai/shared';
import { API_URL } from '../../../../api/config';
import { CORNER_CONFIG, type Sample, type LabelData, type PartialCorners, type DragHandle } from './types';

interface OriginalImagePanelProps {
    currentSample: Sample;
    labelData: LabelData;
    boundingBox: BoundingBox | null;
    hideBox: boolean;
    isDrawingBox: boolean;
    drawStart: { x: number; y: number } | null;
    dragHandle: DragHandle;
    isCornerMode: boolean;
    cornerCount: number;
    pendingCorners: PartialCorners;
    savedCorners: BoardCorners | null;
    onPointerDown: (e: React.PointerEvent, containerRef: React.RefObject<HTMLDivElement>) => void;
    onPointerMove: (e: React.PointerEvent, containerRef: React.RefObject<HTMLDivElement>) => void;
    onPointerUp: (e: React.PointerEvent) => void;
}

export const OriginalImagePanel = ({
    currentSample,
    labelData,
    boundingBox,
    hideBox,
    isDrawingBox,
    drawStart,
    dragHandle,
    isCornerMode,
    cornerCount,
    pendingCorners,
    savedCorners,
    onPointerDown,
    onPointerMove,
    onPointerUp,
}: OriginalImagePanelProps) => {
    const imageContainerRef = useRef<HTMLDivElement>(null);

    const isInteracting = isDrawingBox || dragHandle !== null || drawStart !== null;
    const isEditable = boundingBox && !drawStart;

    const getCursor = useCallback(() => {
        if (isCornerMode) {
            return 'crosshair';
        }
        if (isDrawingBox) {
            return 'crosshair';
        }
        if (dragHandle === 'move') {
            return 'move';
        }
        if (dragHandle === 'nw' || dragHandle === 'se') {
            return 'nwse-resize';
        }
        if (dragHandle === 'ne' || dragHandle === 'sw') {
            return 'nesw-resize';
        }
        return 'default';
    }, [isCornerMode, isDrawingBox, dragHandle]);

    const handlePointerDown = (e: React.PointerEvent) => {
        onPointerDown(e, imageContainerRef);
    };

    const handlePointerMove = (e: React.PointerEvent) => {
        onPointerMove(e, imageContainerRef);
    };

    return (
        <div
            ref={imageContainerRef}
            className="relative bg-surface rounded-lg border border-border overflow-hidden"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            style={{
                touchAction: isInteracting || isDrawingBox || boundingBox ? 'none' : 'auto',
                cursor: getCursor(),
            }}
        >
            <img
                src={currentSample.thumbnailUrl || currentSample.imageUrl || `${API_URL}/api/samples/${currentSample.id}/image`}
                alt={`Sample ${currentSample.id}`}
                className="w-full h-auto select-none"
                draggable={false}
            />

            {/* Bounding box overlay */}
            {boundingBox && !hideBox && (
                <div
                    className={`absolute rounded-sm ${
                        labelData.correction?.boundingBox
                            ? 'border-2 border-success'
                            : 'border-2 border-primary border-dashed'
                    } ${isEditable ? 'cursor-move' : ''}`}
                    style={{
                        left: `${boundingBox.x * 100}%`,
                        top: `${boundingBox.y * 100}%`,
                        width: `${boundingBox.width * 100}%`,
                        height: `${boundingBox.height * 100}%`,
                        pointerEvents: drawStart ? 'none' : 'auto',
                    }}
                >
                    {isEditable && (
                        <>
                            <div className="absolute -left-2 -top-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nwse-resize shadow-sm" />
                            <div className="absolute -right-2 -top-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nesw-resize shadow-sm" />
                            <div className="absolute -left-2 -bottom-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nesw-resize shadow-sm" />
                            <div className="absolute -right-2 -bottom-2 w-4 h-4 bg-white border-2 border-success rounded-full cursor-nwse-resize shadow-sm" />
                        </>
                    )}
                </div>
            )}

            {isDrawingBox && (
                <div className="absolute inset-0 bg-black/10 pointer-events-none" />
            )}

            {/* Saved corners overlay */}
            {savedCorners && !isCornerMode && (
                <>
                    <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ overflow: 'visible' }}>
                        <polygon
                            points={`${savedCorners.topLeft.x * 100}%,${savedCorners.topLeft.y * 100}% ${savedCorners.topRight.x * 100}%,${savedCorners.topRight.y * 100}% ${savedCorners.bottomRight.x * 100}%,${savedCorners.bottomRight.y * 100}% ${savedCorners.bottomLeft.x * 100}%,${savedCorners.bottomLeft.y * 100}%`}
                            fill="none"
                            stroke="cyan"
                            strokeWidth="2"
                            strokeDasharray="6,4"
                        />
                    </svg>
                    {CORNER_CONFIG.map(({ key, label, color }) => {
                        const corner = savedCorners[key];
                        return (
                            <div
                                key={key}
                                className="absolute pointer-events-none"
                                style={{
                                    left: `${corner.x * 100}%`,
                                    top: `${corner.y * 100}%`,
                                    transform: 'translate(-50%, -50%)',
                                }}
                            >
                                <div
                                    className="w-4 h-4 rounded-full border-2 border-white"
                                    style={{ backgroundColor: color }}
                                />
                                <div
                                    className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold px-1 rounded"
                                    style={{ backgroundColor: color, color: 'white' }}
                                >
                                    {label}
                                </div>
                            </div>
                        );
                    })}
                </>
            )}

            {/* Pending corners overlay (while labeling) */}
            {isCornerMode && Object.keys(pendingCorners).length > 0 && (
                <>
                    {cornerCount >= 2 && (
                        <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ overflow: 'visible' }}>
                            {cornerCount >= 2 && pendingCorners.topLeft && pendingCorners.topRight && (
                                <line
                                    x1={`${pendingCorners.topLeft.x * 100}%`}
                                    y1={`${pendingCorners.topLeft.y * 100}%`}
                                    x2={`${pendingCorners.topRight.x * 100}%`}
                                    y2={`${pendingCorners.topRight.y * 100}%`}
                                    stroke="cyan"
                                    strokeWidth="2"
                                    strokeDasharray="6,4"
                                />
                            )}
                            {cornerCount >= 3 && pendingCorners.topRight && pendingCorners.bottomRight && (
                                <line
                                    x1={`${pendingCorners.topRight.x * 100}%`}
                                    y1={`${pendingCorners.topRight.y * 100}%`}
                                    x2={`${pendingCorners.bottomRight.x * 100}%`}
                                    y2={`${pendingCorners.bottomRight.y * 100}%`}
                                    stroke="cyan"
                                    strokeWidth="2"
                                    strokeDasharray="6,4"
                                />
                            )}
                            {cornerCount >= 4 && pendingCorners.bottomRight && pendingCorners.bottomLeft && (
                                <line
                                    x1={`${pendingCorners.bottomRight.x * 100}%`}
                                    y1={`${pendingCorners.bottomRight.y * 100}%`}
                                    x2={`${pendingCorners.bottomLeft.x * 100}%`}
                                    y2={`${pendingCorners.bottomLeft.y * 100}%`}
                                    stroke="cyan"
                                    strokeWidth="2"
                                    strokeDasharray="6,4"
                                />
                            )}
                            {cornerCount >= 4 && pendingCorners.bottomLeft && pendingCorners.topLeft && (
                                <line
                                    x1={`${pendingCorners.bottomLeft.x * 100}%`}
                                    y1={`${pendingCorners.bottomLeft.y * 100}%`}
                                    x2={`${pendingCorners.topLeft.x * 100}%`}
                                    y2={`${pendingCorners.topLeft.y * 100}%`}
                                    stroke="cyan"
                                    strokeWidth="2"
                                    strokeDasharray="6,4"
                                />
                            )}
                        </svg>
                    )}
                    {CORNER_CONFIG.slice(0, cornerCount).map(({ key, label, color }) => {
                        const corner = pendingCorners[key];
                        if (!corner) {
                            return null;
                        }
                        return (
                            <div
                                key={key}
                                className="absolute pointer-events-none"
                                style={{
                                    left: `${corner.x * 100}%`,
                                    top: `${corner.y * 100}%`,
                                    transform: 'translate(-50%, -50%)',
                                }}
                            >
                                <div
                                    className="w-4 h-4 rounded-full border-2 border-white"
                                    style={{ backgroundColor: color }}
                                />
                                <div
                                    className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold px-1 rounded"
                                    style={{ backgroundColor: color, color: 'white' }}
                                >
                                    {label}
                                </div>
                            </div>
                        );
                    })}
                </>
            )}

            {/* Corner mode overlay */}
            {isCornerMode && (
                <div className="absolute inset-0 bg-warning/10 pointer-events-none" />
            )}
        </div>
    );
};
