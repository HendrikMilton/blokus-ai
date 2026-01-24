from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: float       # left edge (0-1)
    y: float       # top edge (0-1)
    width: float   # (0-1)
    height: float  # (0-1)


class BoardDetection(BaseModel):
    hasBoardDetected: bool
    isEntireBoardVisible: bool
    boundingBox: BoundingBox | None
    confidence: float = 0.0


class DetectResponse(BaseModel):
    sample_id: str
    detection: BoardDetection
    processing_time_ms: float
