import os
import time

from app.lib import FeatureMetrics, get_logger
from app.schemas import BOARD_SIZE, BoardCell, DetectBoardResponse

from .cell_classifier import CellClassifier
from .grid_finder import GridFinder
from .preprocessing import Preprocessor

metrics = FeatureMetrics()

# Check if foundation models should be used
USE_FOUNDATION_MODELS = os.environ.get("USE_FOUNDATION_MODELS", "true").lower() == "true"


class DetectionService:
    """Board detection from camera images using foundation models."""

    def __init__(self):
        self.preprocessor = Preprocessor()

        # Legacy OpenCV pipeline (only used when USE_FOUNDATION_MODELS=false)
        self.grid_finder = GridFinder()
        self.classifier = CellClassifier()

        # Foundation model pipeline - initialize eagerly and fail if unavailable
        self._board_detector = None
        self._board_segmenter = None
        self._feature_classifier = None

        if USE_FOUNDATION_MODELS:
            self._initialize_foundation_models()

    def _initialize_foundation_models(self):
        """Initialize all foundation models. Raises error if any are unavailable."""
        errors = []

        # Initialize Board Detector (Grounding DINO)
        try:
            from .board_detector import BoardDetector
            self._board_detector = BoardDetector()
        except ImportError as e:
            errors.append(f"BoardDetector (Grounding DINO): {e}")
        except Exception as e:
            errors.append(f"BoardDetector initialization failed: {e}")

        # Initialize Board Segmenter (SAM2)
        try:
            from .board_segmenter import BoardSegmenter
            self._board_segmenter = BoardSegmenter()
        except ImportError as e:
            errors.append(f"BoardSegmenter (SAM2): {e}")
        except Exception as e:
            errors.append(f"BoardSegmenter initialization failed: {e}")

        # Initialize Feature Classifier (DINOv3)
        try:
            from .feature_classifier import FeatureClassifier
            self._feature_classifier = FeatureClassifier()
        except ImportError as e:
            errors.append(f"FeatureClassifier (DINOv3): {e}")
        except Exception as e:
            errors.append(f"FeatureClassifier initialization failed: {e}")

        # If any errors, fail hard
        if errors:
            error_msg = "Failed to initialize required foundation models:\n" + "\n".join(f"  - {e}" for e in errors)
            raise RuntimeError(error_msg)

    @property
    def board_detector(self):
        """Get Grounding DINO board detector."""
        return self._board_detector

    @property
    def board_segmenter(self):
        """Get SAM2 board segmenter."""
        return self._board_segmenter

    @property
    def feature_classifier(self):
        """Get DINOv3 feature classifier."""
        return self._feature_classifier

    def detect(self, image_b64: str, cid: str) -> DetectBoardResponse:
        log = get_logger(cid)
        start = time.perf_counter()

        # Preprocess image
        img = self.preprocessor.process(image_b64)

        if USE_FOUNDATION_MODELS:
            # Use foundation model pipeline (no fallback)
            result = self._detect_foundation_models(img, cid)

            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record(latency_ms)
            piece_count = sum(1 for row in result.boardState for cell in row if cell.color is not None)
            log.info(
                "detection_complete",
                pipeline="foundation_models",
                latency_ms=round(latency_ms, 2),
                pieces_detected=piece_count,
            )
            return result
        else:
            # Use legacy OpenCV pipeline
            board_state = self._detect_legacy(img, cid)

            piece_count = sum(1 for row in board_state for cell in row if cell.color is not None)

            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record(latency_ms)
            log.info(
                "detection_complete",
                pipeline="legacy_opencv",
                latency_ms=round(latency_ms, 2),
                pieces_detected=piece_count,
            )

            return DetectBoardResponse(boardState=board_state)

    def _detect_foundation_models(self, img, cid: str) -> DetectBoardResponse:
        """
        Modern detection pipeline using Grounding DINO + SAM2 + DINOv3.

        Stage 1: Board Detection (~200ms)
            - Grounding DINO with prompt "game board" -> bounding box

        Stage 2: Board Segmentation (~500ms)
            - SAM2 to get precise board mask from bounding box
            - Corner detection from mask

        Stage 3: Cell Classification (~1-2s)
            - DINOv3 dense feature extraction on warped board
            - Hybrid classifier using features + color
        """
        log = get_logger(cid)
        import numpy as np

        # Stage 1: Board Detection with Grounding DINO
        detection_result = self.board_detector.detect(img, cid)

        if detection_result is None:
            raise RuntimeError("Board detection failed: Grounding DINO could not find a board in the image")

        bbox, confidence = detection_result
        log.info("grounding_dino_detection", confidence=round(confidence, 3))
        bbox = np.array(bbox)

        # Stage 2: Board Segmentation with SAM2
        mask = self.board_segmenter.segment(img, bbox, cid)
        corners = self.board_segmenter.get_corners_from_mask(mask, cid)

        if corners is None:
            raise RuntimeError("Board segmentation failed: SAM2 could not extract board corners from mask")

        warped = self.board_segmenter.get_warped_board(img, corners, output_size=400, cid=cid)

        # Stage 3: Cell Classification with DINOv3
        board_state = self.feature_classifier.classify_cells_with_color(warped, cid)

        return DetectBoardResponse(boardState=board_state)

    def _detect_legacy(self, img, cid: str) -> list[list[BoardCell]]:
        """Legacy OpenCV-based detection pipeline."""
        log = get_logger(cid)

        # Find grid and extract cells
        cells = self.grid_finder.find_grid(img)

        if cells is None:
            log.warning("board_not_found", cid=cid)
            return [[BoardCell(color=None) for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        # Classify each cell
        return self.classifier.classify_cells(cells)
