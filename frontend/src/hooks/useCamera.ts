import { useRef, useState, useCallback, useEffect } from 'react';
import { debug } from '../api/debug';

interface UseCameraOptions {
    facingMode?: 'user' | 'environment';
}

export function useCamera(options: UseCameraOptions = {}) {
    const { facingMode = 'environment' } = options;
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const [isReady, setIsReady] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const startCamera = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode },
                audio: false,
            });

            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
                setIsReady(true);
            }
        } catch (err) {
            const error = err instanceof Error
                ? { name: err.name, message: err.message }
                : String(err);
            setError('Camera access denied or not available');
            debug.error('Camera error', { error, facingMode });
        }
    }, [facingMode]);

    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        setIsReady(false);
    }, []);

    const capture = useCallback((): string | null => {
        if (!videoRef.current || !canvasRef.current) {
            return null;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');

        if (!context) {
            return null;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);

        return canvas.toDataURL('image/jpeg', 0.8);
    }, []);

    useEffect(() => {
        return () => {
            stopCamera();
        };
    }, [stopCamera]);

    return {
        videoRef,
        canvasRef,
        isReady,
        error,
        startCamera,
        stopCamera,
        capture,
    };
}
