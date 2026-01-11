from fastapi import APIRouter, Request

from app.schemas import AnalyzeBoardRequest, AnalyzeBoardResponse

from .service import AnalysisService

router = APIRouter()
service = AnalysisService()


@router.post("/", response_model=AnalyzeBoardResponse)
async def analyze_board(request: Request, body: AnalyzeBoardRequest):
    """Analyze board and recommend best move."""
    cid = request.state.correlation_id
    return service.analyze(body.boardState, body.selectedColor, cid)
