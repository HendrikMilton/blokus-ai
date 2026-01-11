from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

BOARD_SIZE = 20


class PlayerColor(str, Enum):
    BLUE = "blue"
    YELLOW = "yellow"
    RED = "red"
    GREEN = "green"


class BoardCell(BaseModel):
    color: PlayerColor | None = None


class PieceShape(BaseModel):
    name: str
    cells: list[tuple[int, int]]
    rotation: Literal[0, 90, 180, 270]


class AnalysisStats(BaseModel):
    piecesUsed: int = Field(ge=0, le=21)
    piecesRemaining: int = Field(ge=0, le=21)
    winProbability: float = Field(ge=0, le=100)


class DetectBoardRequest(BaseModel):
    image: str  # base64 encoded


class DetectBoardResponse(BaseModel):
    boardState: list[list[BoardCell]]  # 20x20 grid


class AnalyzeBoardRequest(BaseModel):
    boardState: list[list[BoardCell]]
    selectedColor: PlayerColor


class AnalyzeBoardResponse(BaseModel):
    recommendedPiece: PieceShape | None = None  # None if no valid moves
    placement: dict | None = None  # {"row": int, "col": int}, None if no valid moves
    stats: AnalysisStats
