import time
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from app.lib.s3 import download_sample_image

from .detector_yolo import detect_board_yolo, detect_board_with_trained_model
from .schemas import DetectResponse

router = APIRouter()

DetectorModel = Literal["yolo"]


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
