from fastapi import APIRouter, Request

from app.schemas import DetectBoardRequest, DetectBoardResponse

from .service import DetectionService
from .claude_detector import get_claude_detector
from .preprocessing import Preprocessor

router = APIRouter()
service = DetectionService()
preprocessor = Preprocessor()


@router.post("/", response_model=DetectBoardResponse)
async def detect_board(request: Request, body: DetectBoardRequest):
    """Detect board state from camera image."""
    cid = request.state.correlation_id
    return service.detect(body.image, cid)


@router.post("/claude", response_model=DetectBoardResponse)
async def detect_board_claude(request: Request, body: DetectBoardRequest):
    """Detect board state using Claude vision (more accurate, slower)."""
    cid = request.state.correlation_id

    # Preprocess image
    img = preprocessor.process(body.image)

    # Use Claude for detection
    detector = get_claude_detector()
    board_state = detector.detect(img, cid)

    return DetectBoardResponse(boardState=board_state)
