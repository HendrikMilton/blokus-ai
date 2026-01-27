import { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, RefreshCw } from 'lucide-react';
import { Button } from '../../../components/ui/Button';
import { Select } from '../../../components/ui/Select';
import { ML_API, TestDetectResponse } from '../../../api/ml';
import type { TrainingRun, BoundingBox } from '@blokus-ai/shared';

interface Props {
    trainingRuns: TrainingRun[];
}

export const TestModelTab = ({ trainingRuns }: Props) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');
    const [selectedModelId, setSelectedModelId] = useState<string>('');
    const [capturedImage, setCapturedImage] = useState<string | null>(null);
    const [detection, setDetection] = useState<TestDetectResponse | null>(null);
    const [isDetecting, setIsDetecting] = useState(false);
    const [error, setError] = useState<string | null>(null);

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

        // Run detection
        setIsDetecting(true);
        setError(null);
        try {
            const result = await ML_API.testDetect(
                imageDataUrl,
                selectedModelId || undefined
            );
            setDetection(result);
        } catch (err) {
            console.error('Detection failed:', err);
            setError('Detection failed. Please try again.');
        } finally {
            setIsDetecting(false);
        }
    }, [selectedModelId]);

    const clearCapture = useCallback(() => {
        setCapturedImage(null);
        setDetection(null);
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

    return (
        <div className="space-y-4">
            {/* Model Selection */}
            <div className="flex flex-wrap items-end gap-3">
                <Select
                    label="Model"
                    value={selectedModelId}
                    onChange={(val) => {
                        setHasUserSelected(true);
                        setSelectedModelId(val);
                    }}
                    options={[
                        { value: '', label: 'Default (YOLO)' },
                        ...completedRuns.map(run => ({
                            value: run.id,
                            label: `${run.id.slice(0, 8)} - ${run.imageCount} images, ${Math.round((run.accuracy || 0) * 100)}% acc`,
                        })),
                    ]}
                    className="w-64"
                />
                {detection && (
                    <span className="text-xs text-muted pb-2">
                        Detection time: {detection.processing_time_ms.toFixed(0)}ms
                    </span>
                )}
            </div>

            {/* Camera / Image Display */}
            <div className="relative bg-surface-dim rounded-lg overflow-hidden" style={{ maxWidth: '640px' }}>
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

            {/* Detection Results */}
            {detection && (
                <div className="p-4 bg-surface rounded-lg border border-border space-y-2">
                    <h3 className="font-medium text-foreground">Detection Results</h3>
                    <div className="grid grid-cols-2 gap-2 text-sm">
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
        </div>
    );
};
