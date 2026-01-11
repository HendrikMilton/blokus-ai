import time

from app.lib import FeatureMetrics, get_logger
from app.schemas import BOARD_SIZE, BoardCell, DetectBoardResponse

from .cell_classifier import CellClassifier
from .grid_finder import GridFinder
from .preprocessing import Preprocessor

metrics = FeatureMetrics()


class DetectionService:
    """Board detection from camera images using OpenCV."""

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.grid_finder = GridFinder()
        self.classifier = CellClassifier()

    def detect(self, image_b64: str, cid: str) -> DetectBoardResponse:
        log = get_logger(cid)
        start = time.perf_counter()

        try:
            # Preprocess image
            img = self.preprocessor.process(image_b64)

            # Find grid and extract cells
            cells = self.grid_finder.find_grid(img)

            if cells is None:
                log.warning("board_not_found", cid=cid)
                # Return empty board if detection fails
                board_state = [[BoardCell(color=None) for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            else:
                # Classify each cell
                board_state = self.classifier.classify_cells(cells)

            # Count detected pieces for logging
            piece_count = sum(1 for row in board_state for cell in row if cell.color is not None)

            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record(latency_ms)
            log.info("detection_complete", latency_ms=round(latency_ms, 2), pieces_detected=piece_count)

            return DetectBoardResponse(boardState=board_state)

        except Exception as e:
            log.error("detection_failed", error=str(e), cid=cid)
            # Return empty board on error
            board_state = [[BoardCell(color=None) for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            return DetectBoardResponse(boardState=board_state)
