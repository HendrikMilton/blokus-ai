import { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, RefreshCw, Loader2 } from 'lucide-react';
import { Button } from '../../../components/ui/Button';
import { Select } from '../../../components/ui/Select';
import { ML_API, TestDetectResponse, SuggestMovesResponse, SuggestedMove } from '../../../api/ml';
import { PIECES_API } from '../../../api/pieces';
import { formatShortDate, formatShortId } from '../../../utils/format';
import { PLAYER_COLORS } from '@blokus-ai/shared';
import type { TrainingRun, BoundingBox, ModelPipeline, GridLabels, GridPrediction, GridAnalysisResult, PlayerColor } from '@blokus-ai/shared';
import { PiecesPanel } from './label/PiecesPanel';

interface Props {
    trainingRuns: TrainingRun[];
    pipelines: ModelPipeline[];
}

export const TestModelTab = ({ trainingRuns, pipelines }: Props) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');
    const [selectedModelId, setSelectedModelId] = useState<string>('');
    const [selectedDevice, setSelectedDevice] = useState<string>('');
    const [capturedImage, setCapturedImage] = useState<string | null>(null);
    const [detection, setDetection] = useState<TestDetectResponse | null>(null);
    const [isDetecting, setIsDetecting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [gridPrediction, setGridPrediction] = useState<GridPrediction | null>(null);
    const [isGridPredicting, setIsGridPredicting] = useState(false);
    const [selectedColor, setSelectedColor] = useState<PlayerColor | null>(null);
    const [moveSuggestions, setMoveSuggestions] = useState<SuggestMovesResponse | null>(null);
    const [isSuggestingMoves, setIsSuggestingMoves] = useState(false);
    const [gridAnalysis, setGridAnalysis] = useState<GridAnalysisResult | null>(null);

    // Get completed training runs for model selection
    const completedRuns = trainingRuns.filter(r => r.status === 'completed');

    // Track if user has made a selection (to allow selecting "Default")
    const [hasUserSelected, setHasUserSelected] = useState(false);

    // Set default model to most recent completed run (only on initial load)
    useEffect(() => {
        if (!hasUserSelected && !selectedModelId && completedRuns.length > 0) {
            setSelectedModelId(completedRuns[0].id);
        }
    }, [completedRuns, selectedModelId, hasUserSelected]);

    // Auto-run grid prediction when correctedBoardImage becomes available
    useEffect(() => {
        if (detection?.correctedBoardImage && !gridPrediction && !isGridPredicting) {
            setIsGridPredicting(true);
            ML_API.testGridPredict(detection.correctedBoardImage)
                .then(setGridPrediction)
                .catch((err) => {
                    console.error('Grid prediction failed:', err);
                })
                .finally(() => setIsGridPredicting(false));
        }
    }, [detection?.correctedBoardImage, gridPrediction, isGridPredicting]);

    // Analyze grid to get used pieces per color
    useEffect(() => {
        if (!gridPrediction) {
            setGridAnalysis(null);
            return;
        }
        const hasData = gridPrediction.grid.some(row => row.some(cell => cell !== null));
        if (!hasData) {
            setGridAnalysis(null);
            return;
        }
        const gridLabels = gridPrediction.grid.map(row =>
            row.map(cell => cell as GridLabels[number][number])
        );
        PIECES_API.analyzeGrid(gridLabels)
            .then(setGridAnalysis)
            .catch(() => {});
    }, [gridPrediction]);

    // Fetch move suggestions when color is selected and grid analysis is available
    useEffect(() => {
        if (!selectedColor || !gridPrediction || !gridAnalysis) {
            setMoveSuggestions(null);
            return;
        }

        const gridLabels = gridPrediction.grid.map(row =>
            row.map(cell => cell as GridLabels[number][number])
        );

        // Build usedPieces from grid analysis
        const usedPieces: Record<string, string[]> = {};
        for (const c of ['blue', 'yellow', 'red', 'green'] as PlayerColor[]) {
            usedPieces[c] = gridAnalysis[c]?.onBoard ?? [];
        }

        setIsSuggestingMoves(true);
        setMoveSuggestions(null);
        ML_API.suggestMoves(gridLabels, selectedColor, usedPieces)
            .then(setMoveSuggestions)
            .catch((err) => {
                console.error('Move suggestion failed:', err);
            })
            .finally(() => setIsSuggestingMoves(false));
    }, [selectedColor, gridPrediction, gridAnalysis]);

    const startCamera = useCallback(async () => {
        try {
            setError(null);
            // Stop existing stream
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }

            const newStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
                audio: false,
            });

            setStream(newStream);
            if (videoRef.current) {
                videoRef.current.srcObject = newStream;
                // Explicitly play - autoPlay doesn't always work on mobile
                try {
                    await videoRef.current.play();
                } catch (playErr) {
                    console.warn('Video autoplay failed:', playErr);
                }
            }
        } catch (err) {
            console.error('Failed to start camera:', err);
            setError('Failed to access camera. Please ensure camera permissions are granted.');
        }
    }, [facingMode, stream]);

    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
    }, [stream]);

    const switchCamera = useCallback(() => {
        setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
    }, []);

    // Restart camera when facing mode changes
    useEffect(() => {
        if (stream) {
            startCamera();
        }

    }, [facingMode]);

    // Connect stream to video element when both are available
    useEffect(() => {
        if (stream && videoRef.current && videoRef.current.srcObject !== stream) {
            videoRef.current.srcObject = stream;
            videoRef.current.play().catch(err => {
                console.warn('Video play failed:', err);
            });
        }
    }, [stream]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, [stream]);

    const captureAndDetect = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current) {
            return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;

        // Set canvas size to video size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame to canvas
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }
        ctx.drawImage(video, 0, 0);

        // Get base64 image
        const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
        setCapturedImage(imageDataUrl);
        setDetection(null);
        setGridPrediction(null);
        setSelectedColor(null);
        setMoveSuggestions(null);
        setGridAnalysis(null);

        // Run detection
        setIsDetecting(true);
        setError(null);
        try {
            const result = await ML_API.testDetect(
                imageDataUrl,
                selectedModelId || undefined,
                selectedDevice || undefined
            );
            setDetection(result);
        } catch (err) {
            console.error('Detection failed:', err);
            setError('Detection failed. Please try again.');
        } finally {
            setIsDetecting(false);
        }
    }, [selectedModelId, selectedDevice]);

    const clearCapture = useCallback(() => {
        setCapturedImage(null);
        setDetection(null);
        setGridPrediction(null);
        setSelectedColor(null);
        setMoveSuggestions(null);
        setGridAnalysis(null);
        // Reconnect stream to video after React re-renders with video element
        requestAnimationFrame(() => {
            if (stream && videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play().catch(err => {
                    console.warn('Video play failed:', err);
                });
            }
        });
    }, [stream]);

    const renderBoundingBox = (bbox: BoundingBox) => {
        return (
            <div
                className="absolute border-2 border-success rounded"
                style={{
                    left: `${bbox.x * 100}%`,
                    top: `${bbox.y * 100}%`,
                    width: `${bbox.width * 100}%`,
                    height: `${bbox.height * 100}%`,
                }}
            >
                <div className="absolute -top-6 left-0 bg-success text-white text-xs px-1.5 py-0.5 rounded">
                    Board {detection?.detection.confidence ? `${Math.round(detection.detection.confidence * 100)}%` : ''}
                </div>
            </div>
        );
    };

    const renderGridVisualization = (prediction: GridPrediction) => {
        const colorMap: Record<string, string> = {
            blue: PLAYER_COLORS.blue,
            yellow: PLAYER_COLORS.yellow,
            red: PLAYER_COLORS.red,
            green: PLAYER_COLORS.green,
        };

        return (
            <div className="bg-surface-dim rounded-lg p-3 border border-border">
                <h4 className="text-xs font-medium text-secondary mb-2">Grid State</h4>
                <div
                    className="grid gap-[1px] mx-auto"
                    style={{
                        gridTemplateColumns: 'repeat(20, 1fr)',
                        maxWidth: '400px',
                    }}
                >
                    {prediction.grid.flatMap((row, rowIdx) =>
                        row.map((cell, colIdx) => (
                            <div
                                key={`${rowIdx}-${colIdx}`}
                                className="aspect-square rounded-[1px]"
                                style={{
                                    backgroundColor: cell ? colorMap[cell] || '#1a1a2e' : '#1a1a2e',
                                }}
                            />
                        ))
                    )}
                </div>
                <div className="mt-2 flex items-center gap-4 text-xs text-muted">
                    <span>Confidence: {Math.round(prediction.confidence * 100)}%</span>
                    <span>Time: {prediction.processingTimeMs.toFixed(0)}ms</span>
                </div>
            </div>
        );
    };

    const renderMiniBoard = (grid: GridPrediction['grid'], highlightCells: number[][], highlightColor: string) => {
        const colorMap: Record<string, string> = {
            blue: PLAYER_COLORS.blue,
            yellow: PLAYER_COLORS.yellow,
            red: PLAYER_COLORS.red,
            green: PLAYER_COLORS.green,
        };

        const highlightSet = new Set(highlightCells.map(([r, c]) => `${r},${c}`));

        return (
            <div
                className="grid gap-0 mx-auto"
                style={{
                    gridTemplateColumns: 'repeat(20, 1fr)',
                    width: '100%',
                    maxWidth: '200px',
                    aspectRatio: '1',
                }}
            >
                {grid.flatMap((row, rowIdx) =>
                    row.map((cell, colIdx) => {
                        const isHighlighted = highlightSet.has(`${rowIdx},${colIdx}`);
                        const bgColor = isHighlighted
                            ? highlightColor
                            : cell ? colorMap[cell] || '#1a1a2e' : '#1a1a2e';
                        return (
                            <div
                                key={`${rowIdx}-${colIdx}`}
                                style={{
                                    backgroundColor: bgColor,
                                    opacity: isHighlighted ? 1 : cell ? 0.35 : 1,
                                    boxShadow: isHighlighted ? `0 0 3px ${highlightColor}` : undefined,
                                }}
                            />
                        );
                    })
                )}
            </div>
        );
    };

    const renderMoveSuggestionCard = (move: SuggestedMove, rank: number) => {
        const medalColors = ['#FFD700', '#C0C0C0', '#CD7F32'];
        const medalColor = medalColors[rank] || medalColors[2];
        const highlightColor = selectedColor ? PLAYER_COLORS[selectedColor] : '#fff';

        return (
            <div
                key={`${move.pieceId}-${move.row}-${move.col}`}
                className="bg-surface rounded-lg border border-border p-3 space-y-2"
            >
                {/* Header with rank and piece info */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span
                            className="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold"
                            style={{ backgroundColor: medalColor, color: rank === 0 ? '#1a1a2e' : '#fff' }}
                        >
                            {rank + 1}
                        </span>
                        <span className="text-sm font-medium text-foreground">{move.pieceId}</span>
                        <span className="text-xs text-muted">{move.pieceSize} cells</span>
                    </div>
                    <span className="text-xs text-muted">
                        {move.score.toFixed(1)} pts
                    </span>
                </div>

                {/* Mini board */}
                {gridPrediction && renderMiniBoard(gridPrediction.grid, move.cells, highlightColor)}
            </div>
        );
    };

    // Convert grid prediction to GridLabels format for PiecesPanel
    const gridLabelsFromPrediction: GridLabels | null = gridPrediction
        ? gridPrediction.grid.map(row =>
            row.map(cell => cell as GridLabels[number][number])
        )
        : null;

    return (
        <div className="space-y-4">
            {/* Model & Device Selection */}
            <div className="flex flex-wrap items-end gap-3">
                <Select
                    label="Model"
                    value={selectedModelId}
                    onChange={(val) => {
                        setHasUserSelected(true);
                        setSelectedModelId(String(val));
                    }}
                    options={[
                        { value: '', label: 'Default (YOLO)' },
                        ...completedRuns.map(run => ({
                            value: run.id,
                            label: `${formatShortId(run.id)} - ${formatShortDate(run.createdAt)} - ${run.imageCount} imgs - ${Math.round((run.accuracy || 0) * 100)}% acc`,
                        })),
                        ...pipelines.map(pipeline => ({
                            value: `pipeline:${pipeline.id}`,
                            label: `\u{1F517} ${pipeline.name} (${pipeline.componentIds.length} models)`,
                        })),
                    ]}
                    className="w-64"
                />
                <Select
                    label="Device"
                    value={selectedDevice}
                    onChange={(val) => setSelectedDevice(String(val))}
                    options={[
                        { value: '', label: 'Auto' },
                        { value: 'cpu', label: 'CPU' },
                        { value: 'cuda', label: 'GPU' },
                    ]}
                    className="w-28"
                />
                {detection && (
                    <span className="text-xs text-muted pb-2">
                        Detection: {detection.processing_time_ms.toFixed(0)}ms
                    </span>
                )}
            </div>

            {/* Camera / Image Display */}
            <div className="space-y-4">
                {/* Original image + Corrected board side by side */}
                <div className="flex flex-wrap gap-4">
                    {/* Original image with detection */}
                    <div className="relative bg-surface-dim rounded-lg overflow-hidden flex-1 min-w-[300px]" style={{ maxWidth: '640px' }}>
                        {!stream && !capturedImage ? (
                            // Start camera prompt
                            <div className="aspect-video flex flex-col items-center justify-center gap-4 p-8">
                                <div className="w-16 h-16 bg-surface rounded-full flex items-center justify-center">
                                    <Camera className="w-5 h-5" />
                                </div>
                                <p className="text-secondary text-center">
                                    Open camera to test board detection
                                </p>
                                <Button onClick={startCamera} leftIcon={<Camera className="w-5 h-5" />}>
                                    Open Camera
                                </Button>
                            </div>
                        ) : capturedImage ? (
                            // Show captured image with detection overlay
                            <div className="relative">
                                <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded z-10">
                                    Original + Detection
                                </div>
                                <img
                                    src={capturedImage}
                                    alt="Captured"
                                    className="w-full h-auto"
                                />
                                {detection?.detection.boundingBox && renderBoundingBox(detection.detection.boundingBox)}
                                {detection && !detection.detection.hasBoardDetected && (
                                    <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                                        <div className="bg-error text-white px-4 py-2 rounded-lg">
                                            No board detected
                                        </div>
                                    </div>
                                )}
                                {detection && detection.detection.hasBoardDetected && !detection.detection.isEntireBoardVisible && (
                                    <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                                        <div className="bg-warning text-white px-4 py-2 rounded-lg">
                                            Entire board not visible
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            // Live camera feed
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                className="w-full h-auto"
                            />
                        )}

                        {/* Hidden canvas for capture */}
                        <canvas ref={canvasRef} className="hidden" />
                    </div>

                    {/* Corrected Board Image */}
                    {detection?.correctedBoardImage && (
                        <div className="relative bg-surface-dim rounded-lg overflow-hidden" style={{ width: '400px', maxWidth: '100%' }}>
                            <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded z-10">
                                Corrected Board
                            </div>
                            <img
                                src={`data:image/jpeg;base64,${detection.correctedBoardImage}`}
                                alt="Corrected Board"
                                className="w-full aspect-square object-contain"
                            />
                        </div>
                    )}
                </div>

                {/* Detection Results */}
                {detection && (
                    <div className="p-4 bg-surface rounded-lg border border-border space-y-2">
                        <h3 className="font-medium text-foreground">Detection Results</h3>
                        <div className="grid grid-cols-2 gap-2 text-sm" style={{ maxWidth: '400px' }}>
                            <div className="text-secondary">Board Detected:</div>
                            <div className={detection.detection.hasBoardDetected ? 'text-success font-medium' : 'text-error font-medium'}>
                                {detection.detection.hasBoardDetected ? 'Yes' : 'No'}
                            </div>
                            <div className="text-secondary">Confidence:</div>
                            <div className="text-foreground">{Math.round(detection.detection.confidence * 100)}%</div>
                            {detection.detection.hasBoardDetected && (
                                <>
                                    <div className="text-secondary">Entire Board Visible:</div>
                                    <div className={detection.detection.isEntireBoardVisible ? 'text-success font-medium' : 'text-warning font-medium'}>
                                        {detection.detection.isEntireBoardVisible ? 'Yes' : 'No'}
                                    </div>
                                    {detection.detection.classifierConfidence !== null && detection.detection.classifierConfidence !== undefined && (
                                        <>
                                            <div className="text-secondary">Classifier Confidence:</div>
                                            <div className="text-foreground">{Math.round(detection.detection.classifierConfidence * 100)}%</div>
                                        </>
                                    )}
                                </>
                            )}
                            <div className="text-secondary">Processing Time:</div>
                            <div className="text-foreground">{detection.processing_time_ms.toFixed(0)}ms</div>
                        </div>
                    </div>
                )}

                {/* Grid Prediction + Pieces Analysis */}
                {(gridPrediction || isGridPredicting) && (
                    <div className="flex flex-wrap gap-4">
                        {/* Grid Visualization */}
                        <div className="flex-1 min-w-[300px]" style={{ maxWidth: '440px' }}>
                            {isGridPredicting ? (
                                <div className="bg-surface-dim rounded-lg p-8 border border-border flex items-center justify-center">
                                    <span className="text-sm text-muted">Predicting grid...</span>
                                </div>
                            ) : gridPrediction ? (
                                renderGridVisualization(gridPrediction)
                            ) : null}
                        </div>

                        {/* Pieces Panel */}
                        {gridPrediction && (
                            <div className="flex-1 min-w-[250px]" style={{ maxWidth: '400px' }}>
                                <PiecesPanel
                                    gridLabels={null}
                                    aiPrediction={gridLabelsFromPrediction}
                                />
                            </div>
                        )}
                    </div>
                )}

                {/* Move Suggestions Section */}
                {gridPrediction && gridAnalysis && (
                    <div className="space-y-3">
                        {/* Color Selector */}
                        <div className="flex items-center gap-3 flex-wrap">
                            <span className="text-sm text-secondary">Suggest moves for:</span>
                            <div className="flex gap-2">
                                {(['blue', 'yellow', 'red', 'green'] as PlayerColor[]).map(c => (
                                    <button
                                        key={c}
                                        onClick={() => setSelectedColor(selectedColor === c ? null : c)}
                                        className="w-8 h-8 rounded-full transition-all"
                                        style={{
                                            backgroundColor: PLAYER_COLORS[c],
                                            outline: selectedColor === c ? '3px solid white' : '2px solid transparent',
                                            outlineOffset: '1px',
                                            opacity: selectedColor && selectedColor !== c ? 0.4 : 1,
                                        }}
                                        title={c}
                                    />
                                ))}
                            </div>
                            {isSuggestingMoves && (
                                <Loader2 className="w-4 h-4 animate-spin text-muted" />
                            )}
                            {moveSuggestions && (
                                <span className="text-xs text-muted">
                                    {moveSuggestions.processingTimeMs.toFixed(0)}ms
                                </span>
                            )}
                        </div>

                        {/* Move Suggestion Cards */}
                        {moveSuggestions && moveSuggestions.moves.length > 0 && (
                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                {moveSuggestions.moves.map((move, idx) =>
                                    renderMoveSuggestionCard(move, idx)
                                )}
                            </div>
                        )}

                        {/* No moves available */}
                        {moveSuggestions && moveSuggestions.moves.length === 0 && selectedColor && (
                            <div className="p-4 bg-surface-dim rounded-lg border border-border text-center">
                                <p className="text-sm text-muted">No legal moves available for {selectedColor}</p>
                            </div>
                        )}
                    </div>
                )}

            </div>

            {/* Error display */}
            {error && (
                <div className="p-3 bg-error/10 border border-error/20 rounded-lg text-error text-sm">
                    {error}
                </div>
            )}

            {/* Controls */}
            <div className="flex flex-wrap gap-2">
                {stream && !capturedImage && (
                    <>
                        <Button
                            onClick={captureAndDetect}
                            loading={isDetecting}
                            leftIcon={<Camera className="w-5 h-5" />}
                        >
                            Capture & Detect
                        </Button>
                        <Button
                            onClick={switchCamera}
                            variant="secondary"
                            leftIcon={<RefreshCw className="w-5 h-5" />}
                        >
                            Switch Camera
                        </Button>
                        <Button
                            onClick={stopCamera}
                            variant="secondary"
                        >
                            Close Camera
                        </Button>
                    </>
                )}
                {capturedImage && (
                    <>
                        <Button
                            onClick={clearCapture}
                            variant="secondary"
                        >
                            Back to Camera
                        </Button>
                        <Button
                            onClick={captureAndDetect}
                            loading={isDetecting}
                        >
                            Re-detect
                        </Button>
                    </>
                )}
            </div>
        </div>
    );
};
