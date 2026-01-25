import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.lib.s3 import download_sample_image

from .detector_yolo import detect_board_yolo
from .schemas import DetectResponse

router = APIRouter()

DetectorModel = Literal["yolo"]


@router.post("/{sample_id}", response_model=DetectResponse)
async def run_detection(
    sample_id: str,
    model: DetectorModel = Query(default="yolo"),
):
    """Run board detection on a sample image."""
    try:
        filepath = download_sample_image(sample_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sample not found: {e}")

    start_time = time.perf_counter()
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
    detection = detect_board_yolo(filepath)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return DetectResponse(
        sample_id=sample_id,
        detection=detection,
        processing_time_ms=round(elapsed_ms, 2),
    )
