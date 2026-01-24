import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .detector import detect_board
from .schemas import DetectResponse

router = APIRouter()

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "images"


@router.post("/{sample_id}", response_model=DetectResponse)
async def run_detection(sample_id: str):
    """Run board detection on a sample image."""
    filepath = DATA_DIR / f"{sample_id}.png"

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Sample not found")

    start_time = time.perf_counter()
    detection = detect_board(filepath)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return DetectResponse(
        sample_id=sample_id,
        detection=detection,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.get("/{sample_id}", response_model=DetectResponse)
async def get_detection(sample_id: str):
    """
    Get detection result for a sample.
    Currently runs detection on-demand. Future: return cached result.
    """
    filepath = DATA_DIR / f"{sample_id}.png"

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Sample not found")

    start_time = time.perf_counter()
    detection = detect_board(filepath)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return DetectResponse(
        sample_id=sample_id,
        detection=detection,
        processing_time_ms=round(elapsed_ms, 2),
    )
