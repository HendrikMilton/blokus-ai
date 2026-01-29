from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: float       # left edge (0-1)
    y: float       # top edge (0-1)
    width: float   # (0-1)
    height: float  # (0-1)


class Corner(BaseModel):
    x: float  # (0-1)
    y: float  # (0-1)


class BoardCorners(BaseModel):
    topLeft: Corner
    topRight: Corner
    bottomRight: Corner
    bottomLeft: Corner


class BoardDetection(BaseModel):
    hasBoardDetected: bool
    isEntireBoardVisible: bool
    boundingBox: BoundingBox | None
    confidence: float = 0.0
    classifierConfidence: float | None = None
    corners: BoardCorners | None = None
    cornerConfidence: float | None = None


class DetectResponse(BaseModel):
    sample_id: str
    detection: BoardDetection
    processing_time_ms: float
