import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from app.lib.s3 import download_sample_image

from .detector_yolo import detect_board_yolo, detect_board_with_trained_model
from .schemas import DetectResponse, BoardDetection

# API URL for callbacks
API_URL = os.environ.get("API_URL", "http://localhost:3009")

router = APIRouter()

DetectorModel = Literal["yolo"]


# ============================================================================
# Bulk Detection - MUST be defined before /{sample_id} routes
# ============================================================================


class BulkDetectRequest(BaseModel):
    run_id: str
    model_id: str
    sample_ids: list[str]


class BulkDetectResponse(BaseModel):
    success: bool
    message: str


def store_detection_result(sample_id: str, response: DetectResponse) -> bool:
    """Store a single detection result by calling the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            result = client.post(
                f"{API_URL}/api/ml/detect/{sample_id}/result",
                json={
                    "sample_id": response.sample_id,
                    "detection": {
                        "hasBoardDetected": response.detection.hasBoardDetected,
                        "isEntireBoardVisible": response.detection.isEntireBoardVisible,
                        "boundingBox": {
                            "x": response.detection.boundingBox.x,
                            "y": response.detection.boundingBox.y,
                            "width": response.detection.boundingBox.width,
                            "height": response.detection.boundingBox.height,
                        } if response.detection.boundingBox else None,
                        "confidence": response.detection.confidence,
                        "classifierConfidence": response.detection.classifierConfidence,
                    },
                    "processing_time_ms": response.processing_time_ms,
                },
            )
            return result.status_code == 200
    except Exception as e:
        print(f"[bulk_detect] Failed to store result for {sample_id}: {e}")
        return False


def report_bulk_progress(
    run_id: str,
    processed_count: int,
    total_count: int,
    success_count: int,
    failed_count: int,
) -> None:
    """Report progress to the API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{API_URL}/api/ml/detection-runs/{run_id}/progress",
                json={
                    "processedCount": processed_count,
                    "totalCount": total_count,
                    "successCount": success_count,
                    "failedCount": failed_count,
                },
            )
    except Exception as e:
        print(f"[bulk_detect] Failed to report progress: {e}")


def report_bulk_complete(
    run_id: str,
    success_count: int,
    failed_count: int,
    duration_ms: int,
) -> None:
    """Report completion to the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                f"{API_URL}/api/ml/detection-runs/{run_id}/complete",
                json={
                    "successCount": success_count,
                    "failedCount": failed_count,
                    "durationMs": duration_ms,
                },
            )
    except Exception as e:
        print(f"[bulk_detect] Failed to report completion: {e}")


def run_bulk_detection(run_id: str, model_id: str, sample_ids: list[str]) -> None:
    """Run bulk detection in the background."""
    print(f"[bulk_detect] Starting bulk detection: run_id={run_id}, model_id={model_id}, samples={len(sample_ids)}")

    start_time = time.perf_counter()
    success_count = 0
    failed_count = 0
    processed_count = 0
    total_count = len(sample_ids)

    # Pre-load the model once
    from .detector_yolo import get_trained_model
    try:
        model = get_trained_model(model_id)
        print(f"[bulk_detect] Model loaded: {model_id}")
    except Exception as e:
        print(f"[bulk_detect] Failed to load model: {e}")
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        report_bulk_complete(run_id, 0, total_count, duration_ms)
        return

    for i, sample_id in enumerate(sample_ids):
        try:
            # Download image
            filepath = download_sample_image(sample_id)

            # Run detection
            sample_start = time.perf_counter()
            detection = detect_board_with_trained_model(filepath, model_id)
            sample_elapsed_ms = (time.perf_counter() - sample_start) * 1000

            # Create response
            response = DetectResponse(
                sample_id=sample_id,
                detection=detection,
                processing_time_ms=round(sample_elapsed_ms, 2),
            )

            # Store result
            if store_detection_result(sample_id, response):
                success_count += 1
            else:
                failed_count += 1

        except Exception as e:
            print(f"[bulk_detect] Failed to process {sample_id}: {e}")
            failed_count += 1

        processed_count += 1

        # Report progress every 5 samples or on the last sample
        if processed_count % 5 == 0 or processed_count == total_count:
            report_bulk_progress(run_id, processed_count, total_count, success_count, failed_count)
            print(f"[bulk_detect] Progress: {processed_count}/{total_count} (success={success_count}, failed={failed_count})")

    # Report completion
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    report_bulk_complete(run_id, success_count, failed_count, duration_ms)
    print(f"[bulk_detect] Completed: success={success_count}, failed={failed_count}, duration={duration_ms}ms")


@router.post("/bulk", response_model=BulkDetectResponse)
async def bulk_detect(request: BulkDetectRequest, background_tasks: BackgroundTasks):
    """
    Start bulk detection on multiple samples.
    Returns immediately and processes in the background.
    """
    print(f"[bulk_detect] Received request: run_id={request.run_id}, model_id={request.model_id}, samples={len(request.sample_ids)}")

    # Add background task
    background_tasks.add_task(
        run_bulk_detection,
        request.run_id,
        request.model_id,
        request.sample_ids,
    )

    return BulkDetectResponse(
        success=True,
        message=f"Started bulk detection for {len(request.sample_ids)} samples",
    )


# ============================================================================
# Test Detection - accepts base64 image directly
# MUST be defined before /{sample_id} routes to avoid path conflicts
# ============================================================================


class TestDetectRequest(BaseModel):
    image: str  # base64 encoded image
    model_id: Optional[str] = None


class TestDetectResponse(BaseModel):
    detection: BoardDetection
    processing_time_ms: float


@router.post("/test", response_model=TestDetectResponse)
async def test_detect(request: TestDetectRequest):
    """
    Run detection on a base64 encoded image.
    Used for testing models with camera captures.
    """
    # Decode base64 image and save to temp file
    try:
        # Handle data URL format (e.g., "data:image/jpeg;base64,...")
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        temp_path = Path(f.name)

    try:
        start_time = time.perf_counter()

        if request.model_id:
            detection = detect_board_with_trained_model(temp_path, request.model_id)
        else:
            detection = detect_board_yolo(temp_path)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TestDetectResponse(
            detection=detection,
            processing_time_ms=round(elapsed_ms, 2),
        )
    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


# ============================================================================
# Single Sample Detection
# ============================================================================


@router.post("/{sample_id}", response_model=DetectResponse)
async def run_detection(
    sample_id: str,
    model: DetectorModel = Query(default="yolo"),
    model_id: Optional[str] = Query(default=None, description="Trained model ID to use"),
):
    """Run board detection on a sample image."""
    try:
        filepath = download_sample_image(sample_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sample not found: {e}")

    start_time = time.perf_counter()

    if model_id:
        # Use trained model
        detection = detect_board_with_trained_model(filepath, model_id)
    else:
        # Use default YOLO model
        detection = detect_board_yolo(filepath)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return DetectResponse(
        sample_id=sample_id,
        detection=detection,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.get("/{sample_id}", response_model=DetectResponse)
async def get_detection(
    sample_id: str,
    model: DetectorModel = Query(default="yolo"),
    model_id: Optional[str] = Query(default=None, description="Trained model ID to use"),
):
    """
    Get detection result for a sample.
    Currently runs detection on-demand. Future: return cached result.
    """
    try:
        filepath = download_sample_image(sample_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sample not found: {e}")

    start_time = time.perf_counter()

    if model_id:
        # Use trained model
        detection = detect_board_with_trained_model(filepath, model_id)
    else:
        # Use default YOLO model
        detection = detect_board_yolo(filepath)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return DetectResponse(
        sample_id=sample_id,
        detection=detection,
        processing_time_ms=round(elapsed_ms, 2),
    )
