from fastapi import APIRouter, Request

from app.schemas import DetectBoardRequest, DetectBoardResponse

from .service import DetectionService

router = APIRouter()
service = DetectionService()


@router.post("/", response_model=DetectBoardResponse)
async def detect_board(request: Request, body: DetectBoardRequest):
    """Detect board state from camera image."""
    cid = request.state.correlation_id
    return service.detect(body.image, cid)
