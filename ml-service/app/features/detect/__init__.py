import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from app.lib.s3 import download_sample_image, upload_cropped_image

from .detector_yolo import detect_board_yolo, detect_board_with_trained_model
from .perspective import correct_perspective, correct_perspective_all_methods
from .corner_detection import detect_corners_all_methods
from .schemas import DetectResponse, BoardDetection, BoundingBox, BoardCorners, Corner
from .grid_detection import detect_grid_colors, detect_grid_colors_from_bytes, GridPrediction

# API URL for callbacks
API_URL = os.environ.get("API_URL", "http://localhost:3009")

router = APIRouter()

DetectorModel = Literal["yolo"]


# ============================================================================
# Move Suggestions - Uses Expert heuristic AI
# ============================================================================


class SuggestMovesRequest(BaseModel):
    grid: list[list[Optional[str]]]  # 20x20, values: "blue"/"yellow"/"red"/"green"/None
    color: str  # Which player to suggest for
    usedPieces: dict[str, list[str]]  # { "blue": ["I5","L5",...], ... }


class SuggestedMove(BaseModel):
    pieceId: str
    pieceSize: int
    score: float
    row: int
    col: int
    orientationIdx: int
    cells: list[list[int]]  # [[r,c],...] absolute board coordinates


class SuggestMovesResponse(BaseModel):
    moves: list[SuggestedMove]
    processingTimeMs: float


def _reconstruct_game_from_grid(
    grid: list[list[Optional[str]]],
    color: str,
    used_pieces: dict[str, list[str]],
):
    """Reconstruct a BlokusGame from a detected grid state."""
    from app.features.alphazero.game import BlokusGame, BOARD_SIZE, EDGE_OFFSETS, DIAGONAL_OFFSETS
    from app.features.alphazero.pieces import PIECE_INDEX_BY_ID

    COLOR_MAP = {"blue": 1, "yellow": 2, "red": 3, "green": 4}
    COLOR_TO_PLAYER = {"blue": 0, "yellow": 1, "red": 2, "green": 3}

    game = BlokusGame()

    # 1. Populate board from grid
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = grid[r][c] if r < len(grid) and c < len(grid[r]) else None
            if cell and cell in COLOR_MAP:
                game.board[r, c] = COLOR_MAP[cell]

    # 2. Set remaining_pieces and has_played for each player
    for color_name, player_idx in COLOR_TO_PLAYER.items():
        used_ids = used_pieces.get(color_name, [])
        used_indices = set()
        for pid in used_ids:
            if pid in PIECE_INDEX_BY_ID:
                used_indices.add(PIECE_INDEX_BY_ID[pid])

        game.players[player_idx].remaining_pieces -= used_indices

        # Check if player has pieces on the board
        player_color = player_idx + 1
        has_pieces = bool(np.any(game.board == player_color))
        game.players[player_idx].has_played = has_pieces

        if has_pieces:
            # Compute score (count cells on board)
            game.players[player_idx].score = int(np.sum(game.board == player_color))

    # 3. Recompute valid corners for all players from scratch
    for p in range(4):
        game._valid_corners[p] = set()
        player_color = p + 1

        if not game.players[p].has_played:
            # Use starting corner
            from app.features.alphazero.game import STARTING_CORNERS
            game._valid_corners[p] = {STARTING_CORNERS[p]}
            continue

        # Scan board for diagonal attachment points
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r, c] != player_color:
                    continue
                for dr, dc in DIAGONAL_OFFSETS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                        continue
                    if game.board[nr, nc] != 0:
                        continue
                    # Check not edge-adjacent to own color
                    edge_adjacent = False
                    for edr, edc in EDGE_OFFSETS:
                        er, ec = nr + edr, nc + edc
                        if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
                            if game.board[er, ec] == player_color:
                                edge_adjacent = True
                                break
                    if not edge_adjacent:
                        game._valid_corners[p].add((nr, nc))

    # 4. Set current player
    game.current_player = COLOR_TO_PLAYER.get(color, 0)

    return game


@router.post("/suggest-moves", response_model=SuggestMovesResponse)
async def suggest_moves(request: SuggestMovesRequest):
    """
    Get top move suggestions from the Expert heuristic AI.
    """
    from app.features.alphazero.smart_player import SmartPlayer, SmartPlayerConfig
    from app.features.alphazero.pieces import PIECE_BY_INDEX

    start_time = time.perf_counter()

    try:
        game = _reconstruct_game_from_grid(request.grid, request.color, request.usedPieces)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to reconstruct game: {e}")

    # Get legal moves
    legal_moves = game.get_legal_moves()
    if not legal_moves:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SuggestMovesResponse(moves=[], processingTimeMs=round(elapsed_ms, 2))

    # Use Expert player to get scored candidates
    player = SmartPlayer(SmartPlayerConfig.expert())
    candidates = player._get_top_candidates(game, legal_moves)

    # Take top 3
    top_moves = []
    for score, move in candidates[:3]:
        piece = PIECE_BY_INDEX[move.piece_idx]
        cells = piece.get_cells(move.orientation_idx, move.row, move.col)
        top_moves.append(SuggestedMove(
            pieceId=piece.id,
            pieceSize=piece.size,
            score=round(score, 2),
            row=move.row,
            col=move.col,
            orientationIdx=move.orientation_idx,
            cells=cells.tolist(),
        ))

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(f"[suggest_moves] color={request.color}, legal_moves={len(legal_moves)}, top={len(top_moves)}, time={elapsed_ms:.0f}ms")

    return SuggestMovesResponse(moves=top_moves, processingTimeMs=round(elapsed_ms, 2))


# ============================================================================
# Bulk Detection - MUST be defined before /{sample_id} routes
# ============================================================================


class BulkDetectRequest(BaseModel):
    run_id: str
    model_id: str
    sample_ids: list[str]
    pipeline_components: Optional[list[str]] = None  # List of model IDs for pipeline fallback


class BulkDetectResponse(BaseModel):
    success: bool
    message: str


def store_detection_result(sample_id: str, model_id: str, response: DetectResponse) -> bool:
    """Store a single detection result by calling the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            # Build corners dict if present
            corners_dict = None
            if response.detection.corners:
                corners_dict = {
                    "topLeft": {"x": response.detection.corners.topLeft.x, "y": response.detection.corners.topLeft.y},
                    "topRight": {"x": response.detection.corners.topRight.x, "y": response.detection.corners.topRight.y},
                    "bottomRight": {"x": response.detection.corners.bottomRight.x, "y": response.detection.corners.bottomRight.y},
                    "bottomLeft": {"x": response.detection.corners.bottomLeft.x, "y": response.detection.corners.bottomLeft.y},
                }

            result = client.post(
                f"{API_URL}/api/ml/detect/{sample_id}/result",
                params={"modelId": model_id},
                json={
                    "sample_id": response.sample_id,
                    "detection": {
                        "hasBoardDetected": response.detection.hasBoardDetected,
                        "isEntireBoardVisible": response.detection.isEntireBoardVisible,
                        "boundingBox": {
                            "x": response.detection.boundingBox.x,
                            "y": response.detection.boundingBox.y,
                            "width": response.detection.boundingBox.width,
                            "height": response.detection.boundingBox.height,
                        } if response.detection.boundingBox else None,
                        "confidence": response.detection.confidence,
                        "classifierConfidence": response.detection.classifierConfidence,
                        "corners": corners_dict,
                        "cornerConfidence": response.detection.cornerConfidence,
                    },
                    "processing_time_ms": response.processing_time_ms,
                },
            )
            return result.status_code == 200
    except Exception as e:
        print(f"[bulk_detect] Failed to store result for {sample_id}: {e}")
        return False


def report_bulk_progress(
    run_id: str,
    processed_count: int,
    total_count: int,
    success_count: int,
    failed_count: int,
) -> None:
    """Report progress to the API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{API_URL}/api/ml/detection-runs/{run_id}/progress",
                json={
                    "processedCount": processed_count,
                    "totalCount": total_count,
                    "successCount": success_count,
                    "failedCount": failed_count,
                },
            )
    except Exception as e:
        print(f"[bulk_detect] Failed to report progress: {e}")


def report_bulk_complete(
    run_id: str,
    success_count: int,
    failed_count: int,
    duration_ms: int,
) -> None:
    """Report completion to the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                f"{API_URL}/api/ml/detection-runs/{run_id}/complete",
                json={
                    "successCount": success_count,
                    "failedCount": failed_count,
                    "durationMs": duration_ms,
                },
            )
    except Exception as e:
        print(f"[bulk_detect] Failed to report completion: {e}")


def run_bulk_detection(
    run_id: str,
    model_id: str,
    sample_ids: list[str],
    pipeline_components: Optional[list[str]] = None,
) -> None:
    """Run bulk detection in the background.

    If pipeline_components is provided, each sample is run through the models
    in order until a board is detected (fallback behavior).
    """
    is_pipeline = pipeline_components is not None and len(pipeline_components) > 1
    print(f"[bulk_detect] Starting bulk detection: run_id={run_id}, model_id={model_id}, samples={len(sample_ids)}, pipeline={is_pipeline}")

    start_time = time.perf_counter()
    success_count = 0
    failed_count = 0
    processed_count = 0
    total_count = len(sample_ids)

    # Pre-load the model(s)
    from .detector_yolo import get_trained_model

    models_to_load = pipeline_components if is_pipeline else [model_id]
    loaded_models = {}

    for mid in models_to_load:
        try:
            loaded_models[mid] = get_trained_model(mid)
            print(f"[bulk_detect] Model loaded: {mid}")
        except Exception as e:
            print(f"[bulk_detect] Failed to load model {mid}: {e}")
            # Continue with other models if pipeline
            if not is_pipeline:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                report_bulk_complete(run_id, 0, total_count, duration_ms)
                return

    if not loaded_models:
        print(f"[bulk_detect] No models could be loaded")
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        report_bulk_complete(run_id, 0, total_count, duration_ms)
        return

    for i, sample_id in enumerate(sample_ids):
        try:
            # Download image
            filepath = download_sample_image(sample_id)
            sample_start = time.perf_counter()

            detection = None
            used_model_id = model_id

            if is_pipeline:
                # Pipeline detection: try each model until one detects a board
                for component_id in pipeline_components:
                    if component_id not in loaded_models:
                        continue

                    detection = detect_board_with_trained_model(filepath, component_id)
                    used_model_id = component_id

                    # If board detected, stop trying other models
                    if detection.hasBoardDetected:
                        break
            else:
                # Single model detection
                detection = detect_board_with_trained_model(filepath, model_id)

            sample_elapsed_ms = (time.perf_counter() - sample_start) * 1000

            # Create response
            response = DetectResponse(
                sample_id=sample_id,
                detection=detection,
                processing_time_ms=round(sample_elapsed_ms, 2),
            )

            # Store result using the primary model ID (first component for pipelines)
            # This ensures all detections from a pipeline run are associated with the same model_id
            # which is needed for disagreement count calculation
            if store_detection_result(sample_id, model_id, response):
                success_count += 1
            else:
                failed_count += 1

        except Exception as e:
            print(f"[bulk_detect] Failed to process {sample_id}: {e}")
            failed_count += 1

        processed_count += 1

        # Report progress every 5 samples or on the last sample
        if processed_count % 5 == 0 or processed_count == total_count:
            report_bulk_progress(run_id, processed_count, total_count, success_count, failed_count)
            print(f"[bulk_detect] Progress: {processed_count}/{total_count} (success={success_count}, failed={failed_count})")

    # Report completion
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    report_bulk_complete(run_id, success_count, failed_count, duration_ms)
    print(f"[bulk_detect] Completed: success={success_count}, failed={failed_count}, duration={duration_ms}ms")


@router.post("/bulk", response_model=BulkDetectResponse)
async def bulk_detect(request: BulkDetectRequest, background_tasks: BackgroundTasks):
    """
    Start bulk detection on multiple samples.
    Returns immediately and processes in the background.

    If pipeline_components is provided, each sample is run through the models
    in order until a board is detected (fallback behavior).
    """
    is_pipeline = request.pipeline_components is not None and len(request.pipeline_components) > 1
    print(f"[bulk_detect] Received request: run_id={request.run_id}, model_id={request.model_id}, samples={len(request.sample_ids)}, pipeline={is_pipeline}")

    # Add background task
    background_tasks.add_task(
        run_bulk_detection,
        request.run_id,
        request.model_id,
        request.sample_ids,
        request.pipeline_components,
    )

    return BulkDetectResponse(
        success=True,
        message=f"Started bulk detection for {len(request.sample_ids)} samples",
    )


# ============================================================================
# Test Detection - accepts base64 image directly
# MUST be defined before /{sample_id} routes to avoid path conflicts
# ============================================================================


class TestDetectRequest(BaseModel):
    image: str  # base64 encoded image
    model_id: Optional[str] = None
    device: Optional[str] = None  # "cpu", "cuda", or None for auto


class TestDetectResponse(BaseModel):
    detection: BoardDetection
    processing_time_ms: float
    correctedBoardImage: Optional[str] = None  # base64 encoded corrected board image (best corner method)


def _generate_corrected_image(
    image_path: Path,
    corners: Optional[BoardCorners],
    output_size: int = 400,
) -> Optional[str]:
    """Generate a perspective-corrected board image from detected corners."""
    from .perspective import apply_perspective_transform, encode_image_base64

    if corners is None:
        return None

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]

        # Convert normalized corners to pixel coordinates
        pixel_corners = np.array([
            [corners.topLeft.x * w, corners.topLeft.y * h],
            [corners.topRight.x * w, corners.topRight.y * h],
            [corners.bottomRight.x * w, corners.bottomRight.y * h],
            [corners.bottomLeft.x * w, corners.bottomLeft.y * h],
        ], dtype=np.float32)

        warped = apply_perspective_transform(img, pixel_corners, output_size)
        return encode_image_base64(warped)
    except Exception as e:
        print(f"[test_detect] Failed to generate corrected image: {e}")
        return None


@router.post("/test", response_model=TestDetectResponse)
async def test_detect(request: TestDetectRequest):
    """
    Run detection on a base64 encoded image.
    Used for testing models with camera captures.

    Uses the same corner detection flow as the Corners button (best/hough/edge)
    for perspective correction, which produces much better results than the old
    heuristic methods.
    """
    # Decode base64 image and save to temp file
    try:
        # Handle data URL format (e.g., "data:image/jpeg;base64,...")
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        temp_path = Path(f.name)

    try:
        start_time = time.perf_counter()

        t0 = time.perf_counter()
        device = request.device
        try:
            if request.model_id:
                detection = detect_board_with_trained_model(temp_path, request.model_id, device=device)
            else:
                detection = detect_board_yolo(temp_path, device=device)
        except ValueError as e:
            if "CUDA" in str(e):
                print(f"[test_detect] CUDA not available, falling back to CPU")
                device = "cpu"
                if request.model_id:
                    detection = detect_board_with_trained_model(temp_path, request.model_id, device=device)
                else:
                    detection = detect_board_yolo(temp_path, device=device)
            else:
                raise
        t1 = time.perf_counter()
        print(f"[test_detect] detection: {(t1-t0)*1000:.0f}ms (model={request.model_id or 'default'}, device={device or 'auto'})")

        corrected_image = None

        if detection.hasBoardDetected and detection.isEntireBoardVisible and detection.boundingBox:
            t2 = time.perf_counter()
            corner_results = detect_corners_all_methods(temp_path, detection.boundingBox)
            t3 = time.perf_counter()
            print(f"[test_detect] corners: {(t3-t2)*1000:.0f}ms (best={corner_results.best.success}, hough={corner_results.hough_ransac.success}, edge={corner_results.edge_intersection.success})")

            t4 = time.perf_counter()
            corrected_image = _generate_corrected_image(temp_path, corner_results.best.corners)
            t5 = time.perf_counter()
            print(f"[test_detect] perspective warp: {(t5-t4)*1000:.0f}ms")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[test_detect] total: {elapsed_ms:.0f}ms")

        return TestDetectResponse(
            detection=detection,
            processing_time_ms=round(elapsed_ms, 2),
            correctedBoardImage=corrected_image,
        )
    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


# ============================================================================
# Bulk Corner Detection - MUST be before /{sample_id} routes to prevent collision
# ============================================================================


class BulkCornerDetectRequest(BaseModel):
    run_id: str
    sample_ids: list[str]


class BulkCornerDetectResponse(BaseModel):
    success: bool
    message: str


@router.post("/bulk-corners", response_model=BulkCornerDetectResponse)
async def bulk_corner_detect(request: BulkCornerDetectRequest, background_tasks: BackgroundTasks):
    """
    Start bulk corner detection on multiple samples.
    Runs all 3 corner detection methods on each sample.
    Returns immediately and processes in the background.
    """
    print(f"[bulk_corners] Received request: run_id={request.run_id}, samples={len(request.sample_ids)}")

    # Add background task
    background_tasks.add_task(
        run_bulk_corner_detection,
        request.run_id,
        request.sample_ids,
    )

    return BulkCornerDetectResponse(
        success=True,
        message=f"Started bulk corner detection for {len(request.sample_ids)} samples",
    )


# ============================================================================
# Grid Color Detection
# ============================================================================


class GridPredictRequest(BaseModel):
    image_url: str


@router.post("/grid-predict", response_model=GridPrediction)
async def predict_grid(request: GridPredictRequest):
    """
    Predict colors in a 20x20 grid from a cropped board image.
    Uses HSV color space analysis to classify each cell.
    """
    try:
        result = detect_grid_colors(request.image_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid prediction failed: {e}")


class TestGridPredictRequest(BaseModel):
    image: str  # base64 encoded corrected board image


@router.post("/test-grid-predict", response_model=GridPrediction)
async def test_grid_predict(request: TestGridPredictRequest):
    """
    Predict grid colors from a base64 encoded corrected board image.
    Used by the Test Model tab for live camera testing.
    """
    try:
        t0 = time.perf_counter()
        # Decode base64 image
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        t1 = time.perf_counter()

        result = detect_grid_colors_from_bytes(image_bytes)
        t2 = time.perf_counter()
        print(f"[test_grid_predict] decode: {(t1-t0)*1000:.0f}ms, grid predict: {(t2-t1)*1000:.0f}ms, total: {(t2-t0)*1000:.0f}ms")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid prediction failed: {e}")


# ============================================================================
# Bulk Grid Prediction
# ============================================================================


class BulkGridSample(BaseModel):
    sample_id: str
    cropped_url: str


class BulkGridPredictRequest(BaseModel):
    samples: list[BulkGridSample]
    api_base_url: str


class BulkGridPredictResponse(BaseModel):
    success: bool
    message: str


def store_grid_prediction(
    api_base_url: str,
    sample_id: str,
    grid: list[list[Optional[str]]],
    confidence: float,
    processing_time_ms: float,
) -> bool:
    """Store a single grid prediction result by calling the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            result = client.post(
                f"{api_base_url}/api/ml/grid-predictions/{sample_id}",
                json={
                    "grid": grid,
                    "confidence": confidence,
                    "processingTimeMs": int(processing_time_ms),
                },
            )
            return result.status_code == 200
    except Exception as e:
        print(f"[bulk_grid] Failed to store result for {sample_id}: {e}")
        return False


def report_grid_progress(
    api_base_url: str,
    processed_count: int,
    total_count: int,
    success_count: int,
    failed_count: int,
) -> None:
    """Report progress to the API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{api_base_url}/api/ml/grid-runs/progress",
                json={
                    "processedCount": processed_count,
                    "totalCount": total_count,
                    "successCount": success_count,
                    "failedCount": failed_count,
                },
            )
    except Exception as e:
        print(f"[bulk_grid] Failed to report progress: {e}")


def report_grid_complete(
    api_base_url: str,
    success_count: int,
    failed_count: int,
    duration_ms: int,
) -> None:
    """Report completion to the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                f"{api_base_url}/api/ml/grid-runs/complete",
                json={
                    "successCount": success_count,
                    "failedCount": failed_count,
                    "durationMs": duration_ms,
                },
            )
    except Exception as e:
        print(f"[bulk_grid] Failed to report completion: {e}")


def report_grid_failed(
    api_base_url: str,
    error_message: str,
    duration_ms: int,
) -> None:
    """Report failure to the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                f"{api_base_url}/api/ml/grid-runs/failed",
                json={
                    "errorMessage": error_message,
                    "durationMs": duration_ms,
                },
            )
    except Exception as e:
        print(f"[bulk_grid] Failed to report failure: {e}")


def run_bulk_grid_prediction(samples: list[BulkGridSample], api_base_url: str) -> None:
    """Run bulk grid prediction in the background."""
    import sys
    print(f"[bulk_grid] Starting bulk grid prediction: samples={len(samples)}", flush=True)
    sys.stdout.flush()

    start_time = time.perf_counter()
    success_count = 0
    failed_count = 0
    total_count = len(samples)

    try:
        for sample in samples:
            print(f"[bulk_grid] Processing sample {sample.sample_id}, url={sample.cropped_url[:50]}...", flush=True)
            try:
                # Run grid detection
                result = detect_grid_colors(sample.cropped_url)
                print(f"[bulk_grid] Got result, confidence={result.confidence}", flush=True)

                # Store result via API callback
                if store_grid_prediction(
                    api_base_url,
                    sample.sample_id,
                    result.grid,
                    result.confidence,
                    result.processingTimeMs,
                ):
                    success_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                print(f"[bulk_grid] Failed to process {sample.sample_id}: {e}")
                failed_count += 1

            # Report progress every 5 samples or on last sample
            processed_count = success_count + failed_count
            if processed_count % 5 == 0 or processed_count == total_count:
                report_grid_progress(api_base_url, processed_count, total_count, success_count, failed_count)
                print(f"[bulk_grid] Progress: {processed_count}/{total_count}")

        # Report completion
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        report_grid_complete(api_base_url, success_count, failed_count, duration_ms)
        print(f"[bulk_grid] Completed: success={success_count}, failed={failed_count}, duration={duration_ms}ms")

    except Exception as e:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        report_grid_failed(api_base_url, str(e), duration_ms)
        print(f"[bulk_grid] Failed: {e}")


@router.post("/bulk-grid-predict", response_model=BulkGridPredictResponse)
async def bulk_grid_predict(request: BulkGridPredictRequest, background_tasks: BackgroundTasks):
    """
    Start bulk grid prediction on multiple samples.
    Returns immediately and processes in the background.
    """
    print(f"[bulk_grid] Received request: samples={len(request.samples)}")

    # Add background task
    background_tasks.add_task(
        run_bulk_grid_prediction,
        request.samples,
        request.api_base_url,
    )

    return BulkGridPredictResponse(
        success=True,
        message=f"Started bulk grid prediction for {len(request.samples)} samples",
    )


# ============================================================================
# Single Sample Detection
# ============================================================================


@router.post("/{sample_id}", response_model=DetectResponse)
async def run_detection(
    sample_id: str,
    model: DetectorModel = Query(default="yolo"),
    model_id: Optional[str] = Query(default=None, description="Trained model ID to use"),
):
    """Run board detection on a sample image."""
    try:
        filepath = download_sample_image(sample_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sample not found: {e}")

    start_time = time.perf_counter()

    if model_id:
        # Use trained model
        detection = detect_board_with_trained_model(filepath, model_id)
    else:
        # Use default YOLO model
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
    model_id: Optional[str] = Query(default=None, description="Trained model ID to use"),
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

    if model_id:
        # Use trained model
        detection = detect_board_with_trained_model(filepath, model_id)
    else:
        # Use default YOLO model
        detection = detect_board_yolo(filepath)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return DetectResponse(
        sample_id=sample_id,
        detection=detection,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ============================================================================
# Perspective Correction for Stored Samples
# ============================================================================


class PerspectiveCorrectionRequest(BaseModel):
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float


class PerspectiveCorrectionMethod(BaseModel):
    method: str
    image: Optional[str] = None  # base64 encoded
    success: bool
    corners: Optional[list] = None


class PerspectiveCorrectionResponse(BaseModel):
    hough_lines_merged: PerspectiveCorrectionMethod
    hough_lines_extreme: PerspectiveCorrectionMethod
    color_segmentation: PerspectiveCorrectionMethod
    contour_detection: PerspectiveCorrectionMethod


@router.post("/{sample_id}/perspective", response_model=PerspectiveCorrectionResponse)
async def correct_sample_perspective(sample_id: str, request: PerspectiveCorrectionRequest):
    """
    Apply perspective correction to a stored sample using the provided bounding box.
    """
    try:
        filepath = download_sample_image(sample_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sample not found: {e}")

    # Create BoundingBox from request
    bbox = BoundingBox(
        x=request.bbox_x,
        y=request.bbox_y,
        width=request.bbox_width,
        height=request.bbox_height,
    )

    # Get correction result
    all_results = correct_perspective_all_methods(filepath, bbox)

    return PerspectiveCorrectionResponse(
        hough_lines_merged=PerspectiveCorrectionMethod(
            method=all_results.hough_lines_merged.method,
            image=all_results.hough_lines_merged.image,
            success=all_results.hough_lines_merged.success,
            corners=all_results.hough_lines_merged.corners,
        ),
        hough_lines_extreme=PerspectiveCorrectionMethod(
            method=all_results.hough_lines_extreme.method,
            image=all_results.hough_lines_extreme.image,
            success=all_results.hough_lines_extreme.success,
            corners=all_results.hough_lines_extreme.corners,
        ),
        color_segmentation=PerspectiveCorrectionMethod(
            method=all_results.color_segmentation.method,
            image=all_results.color_segmentation.image,
            success=all_results.color_segmentation.success,
            corners=all_results.color_segmentation.corners,
        ),
        contour_detection=PerspectiveCorrectionMethod(
            method=all_results.contour_detection.method,
            image=all_results.contour_detection.image,
            success=all_results.contour_detection.success,
            corners=all_results.contour_detection.corners,
        ),
    )


# ============================================================================
# Perspective Correction Using Saved Corners
# ============================================================================


class CornerPoint(BaseModel):
    x: float
    y: float


class CornersInput(BaseModel):
    topLeft: CornerPoint
    topRight: CornerPoint
    bottomRight: CornerPoint
    bottomLeft: CornerPoint


class PerspectiveFromCornersRequest(BaseModel):
    best: Optional[CornersInput] = None
    hough: Optional[CornersInput] = None
    edge: Optional[CornersInput] = None


class PerspectiveMethodResult(BaseModel):
    method: str
    image: Optional[str] = None  # base64 encoded
    success: bool


class PerspectiveFromCornersResponse(BaseModel):
    best: PerspectiveMethodResult
    hough: PerspectiveMethodResult
    edge: PerspectiveMethodResult


@router.post("/{sample_id}/perspective-from-corners", response_model=PerspectiveFromCornersResponse)
async def correct_perspective_from_corners(sample_id: str, request: PerspectiveFromCornersRequest):
    """
    Apply perspective correction using pre-computed corner coordinates.
    This uses the saved corner detection results (best, hough, edge) instead of
    re-running corner detection methods.
    """
    from .perspective import apply_perspective_transform, encode_image_base64

    try:
        filepath = download_sample_image(sample_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sample not found: {e}")

    # Load image
    img = cv2.imread(str(filepath))
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load image")

    h, w = img.shape[:2]
    output_size = 400

    def corners_to_pixels(corners: CornersInput) -> np.ndarray:
        """Convert normalized corners (0-1) to pixel coordinates."""
        return np.array([
            [corners.topLeft.x * w, corners.topLeft.y * h],
            [corners.topRight.x * w, corners.topRight.y * h],
            [corners.bottomRight.x * w, corners.bottomRight.y * h],
            [corners.bottomLeft.x * w, corners.bottomLeft.y * h],
        ], dtype=np.float32)

    def apply_correction(corners: Optional[CornersInput], method_name: str) -> PerspectiveMethodResult:
        if corners is None:
            return PerspectiveMethodResult(method=method_name, image=None, success=False)

        try:
            pixel_corners = corners_to_pixels(corners)
            warped = apply_perspective_transform(img, pixel_corners, output_size)
            encoded = encode_image_base64(warped)
            return PerspectiveMethodResult(method=method_name, image=encoded, success=True)
        except Exception as e:
            print(f"[perspective] Failed to apply {method_name}: {e}")
            return PerspectiveMethodResult(method=method_name, image=None, success=False)

    return PerspectiveFromCornersResponse(
        best=apply_correction(request.best, "best"),
        hough=apply_correction(request.hough, "hough"),
        edge=apply_correction(request.edge, "edge"),
    )


# ============================================================================
# Corner Detection
# ============================================================================


class CornerDetectionMethod(BaseModel):
    method: str
    corners: Optional[BoardCorners] = None
    success: bool
    debug_image: Optional[str] = None  # base64 encoded image with corners drawn


class CornerDetectionResponse(BaseModel):
    contour: CornerDetectionMethod
    hough_ransac: CornerDetectionMethod
    edge_intersection: CornerDetectionMethod


class CornerDetectionRequest(BaseModel):
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float


@router.post("/{sample_id}/corners", response_model=CornerDetectionResponse)
async def detect_sample_corners(sample_id: str, request: CornerDetectionRequest):
    """
    Run all corner detection methods on a stored sample.
    Returns corners detected by each method for comparison.
    """
    try:
        filepath = download_sample_image(sample_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sample not found: {e}")

    # Create BoundingBox from request
    bbox = BoundingBox(
        x=request.bbox_x,
        y=request.bbox_y,
        width=request.bbox_width,
        height=request.bbox_height,
    )

    # Run all corner detection methods
    all_results = detect_corners_all_methods(filepath, bbox)

    def convert_result(result) -> CornerDetectionMethod:
        return CornerDetectionMethod(
            method=result.method,
            corners=result.corners,
            success=result.success,
            debug_image=result.debug_image,
        )

    return CornerDetectionResponse(
        contour=convert_result(all_results.contour),
        hough_ransac=convert_result(all_results.hough_ransac),
        edge_intersection=convert_result(all_results.edge_intersection),
    )


# ============================================================================
# Bulk Corner Detection Helper Functions
# ============================================================================


def calculate_corner_error(detected: Optional[BoardCorners], human: Optional[BoardCorners]) -> Optional[float]:
    """
    Calculate error between detected corners and human-labeled corners.
    Returns the average Euclidean distance (normalized 0-1) across all 4 corners.
    Returns None if either is missing.
    """
    if detected is None or human is None:
        return None

    def point_distance(p1: Corner, p2: Corner) -> float:
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

    tl_err = point_distance(detected.topLeft, human.topLeft)
    tr_err = point_distance(detected.topRight, human.topRight)
    br_err = point_distance(detected.bottomRight, human.bottomRight)
    bl_err = point_distance(detected.bottomLeft, human.bottomLeft)

    return (tl_err + tr_err + br_err + bl_err) / 4.0


def report_corner_progress(
    run_id: str,
    processed_count: int,
    total_count: int,
    success_count: int,
    failed_count: int,
) -> None:
    """Report progress to the API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{API_URL}/api/ml/corner-runs/{run_id}/progress",
                json={
                    "processedCount": processed_count,
                    "totalCount": total_count,
                    "successCount": success_count,
                    "failedCount": failed_count,
                },
            )
    except Exception as e:
        print(f"[bulk_corners] Failed to report progress: {e}")


def report_corner_complete(
    run_id: str,
    success_count: int,
    failed_count: int,
    duration_ms: int,
) -> None:
    """Report completion to the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                f"{API_URL}/api/ml/corner-runs/{run_id}/complete",
                json={
                    "successCount": success_count,
                    "failedCount": failed_count,
                    "durationMs": duration_ms,
                },
            )
    except Exception as e:
        print(f"[bulk_corners] Failed to report completion: {e}")


def report_corner_failed(
    run_id: str,
    error_message: str,
    duration_ms: int,
) -> None:
    """Report failure to the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                f"{API_URL}/api/ml/corner-runs/{run_id}/failed",
                json={
                    "errorMessage": error_message,
                    "durationMs": duration_ms,
                },
            )
    except Exception as e:
        print(f"[bulk_corners] Failed to report failure: {e}")


def generate_and_upload_cropped_image(
    sample_id: str,
    filepath: Path,
    corners: Optional[BoardCorners],
    method: str,
) -> Optional[str]:
    """
    Generate a cropped/perspective-corrected image and upload to S3.

    Args:
        sample_id: The sample ID
        filepath: Path to the source image
        corners: Detected corners (normalized 0-1)
        method: The corner detection method (best, hough, edge)

    Returns:
        CloudFront URL for the cropped image, or None if failed
    """
    from .perspective import apply_perspective_transform

    if corners is None:
        return None

    try:
        # Load image
        img = cv2.imread(str(filepath))
        if img is None:
            return None

        h, w = img.shape[:2]
        output_size = 400

        # Convert normalized corners to pixel coordinates
        pixel_corners = np.array([
            [corners.topLeft.x * w, corners.topLeft.y * h],
            [corners.topRight.x * w, corners.topRight.y * h],
            [corners.bottomRight.x * w, corners.bottomRight.y * h],
            [corners.bottomLeft.x * w, corners.bottomLeft.y * h],
        ], dtype=np.float32)

        # Apply perspective transform
        warped = apply_perspective_transform(img, pixel_corners, output_size)

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        _, buffer = cv2.imencode('.jpg', warped, encode_params)
        image_bytes = buffer.tobytes()

        # Upload to S3
        url = upload_cropped_image(sample_id, method, image_bytes)
        return url

    except Exception as e:
        print(f"[bulk_corners] Failed to generate cropped image for {sample_id}/{method}: {e}")
        return None


def store_corner_result(
    run_id: str,
    sample_id: str,
    best_result,
    hough_result,
    edge_result,
    human_corners: Optional[dict],
    cropped_urls: Optional[dict] = None,
) -> bool:
    """Store a single corner detection result by calling the API."""
    try:
        def corners_to_dict(corners: Optional[BoardCorners]) -> Optional[dict]:
            if corners is None:
                return None
            return {
                "topLeft": {"x": corners.topLeft.x, "y": corners.topLeft.y},
                "topRight": {"x": corners.topRight.x, "y": corners.topRight.y},
                "bottomRight": {"x": corners.bottomRight.x, "y": corners.bottomRight.y},
                "bottomLeft": {"x": corners.bottomLeft.x, "y": corners.bottomLeft.y},
            }

        def quality_to_dict(result) -> Optional[dict]:
            if result.quality is None:
                return None
            return {
                "aspectRatio": result.quality.aspect_ratio,
                "areaRatio": result.quality.area_ratio,
                "avgAngle": result.quality.avg_angle,
            }

        # Convert human corners dict to BoardCorners for error calculation
        human_bc = None
        if human_corners:
            human_bc = BoardCorners(
                topLeft=Corner(x=human_corners["topLeft"]["x"], y=human_corners["topLeft"]["y"]),
                topRight=Corner(x=human_corners["topRight"]["x"], y=human_corners["topRight"]["y"]),
                bottomRight=Corner(x=human_corners["bottomRight"]["x"], y=human_corners["bottomRight"]["y"]),
                bottomLeft=Corner(x=human_corners["bottomLeft"]["x"], y=human_corners["bottomLeft"]["y"]),
            )

        with httpx.Client(timeout=30.0) as client:
            result = client.post(
                f"{API_URL}/api/ml/corner-runs/{run_id}/result",
                json={
                    "sample_id": sample_id,
                    "best": {
                        "success": best_result.success,
                        "corners": corners_to_dict(best_result.corners),
                        "error": calculate_corner_error(best_result.corners, human_bc),
                        "quality": quality_to_dict(best_result),
                        "croppedUrl": cropped_urls.get("best") if cropped_urls else None,
                    },
                    "hough": {
                        "success": hough_result.success,
                        "corners": corners_to_dict(hough_result.corners),
                        "error": calculate_corner_error(hough_result.corners, human_bc),
                        "quality": quality_to_dict(hough_result),
                        "croppedUrl": cropped_urls.get("hough") if cropped_urls else None,
                    },
                    "edge": {
                        "success": edge_result.success,
                        "corners": corners_to_dict(edge_result.corners),
                        "error": calculate_corner_error(edge_result.corners, human_bc),
                        "quality": quality_to_dict(edge_result),
                        "croppedUrl": cropped_urls.get("edge") if cropped_urls else None,
                    },
                },
            )
            return result.status_code == 200
    except Exception as e:
        print(f"[bulk_corners] Failed to store result for {sample_id}: {e}")
        return False


def get_sample_bbox_and_corners(sample_ids: list[str]) -> dict:
    """
    Fetch bounding boxes and human corners for samples from the API.
    Returns dict of sample_id -> { bbox, corners }
    """
    sample_data = {}
    try:
        with httpx.Client(timeout=60.0) as client:
            # Get detection info (bboxes) for all samples
            response = client.get(f"{API_URL}/api/ml/samples")
            if response.status_code == 200:
                data = response.json()
                for sample in data.get("samples", []):
                    sample_id = sample["id"]
                    if sample_id in sample_ids:
                        # Get bbox from human label first, then AI
                        bbox = None
                        if sample.get("human") and sample["human"].get("boundingBox"):
                            bbox = sample["human"]["boundingBox"]
                        elif sample.get("ai") and sample["ai"].get("boundingBox"):
                            bbox = sample["ai"]["boundingBox"]

                        corners = None
                        if sample.get("human") and sample["human"].get("corners"):
                            corners = sample["human"]["corners"]

                        sample_data[sample_id] = {
                            "bbox": bbox,
                            "corners": corners,
                        }
    except Exception as e:
        print(f"[bulk_corners] Failed to fetch sample data: {e}")

    return sample_data


def run_bulk_corner_detection(run_id: str, sample_ids: list[str]) -> None:
    """Run bulk corner detection in the background."""
    print(f"[bulk_corners] Starting bulk corner detection: run_id={run_id}, samples={len(sample_ids)}")

    start_time = time.perf_counter()
    success_count = 0
    failed_count = 0
    processed_count = 0
    total_count = len(sample_ids)

    try:
        # Get bbox and human corners for all samples
        sample_data = get_sample_bbox_and_corners(sample_ids)
        print(f"[bulk_corners] Fetched data for {len(sample_data)} samples")

        for sample_id in sample_ids:
            try:
                data = sample_data.get(sample_id, {})
                bbox_dict = data.get("bbox")
                human_corners = data.get("corners")

                if not bbox_dict:
                    print(f"[bulk_corners] Skipping {sample_id}: no bounding box")
                    failed_count += 1
                    processed_count += 1
                    continue

                # Download image
                filepath = download_sample_image(sample_id)

                # Create BoundingBox
                bbox = BoundingBox(
                    x=bbox_dict["x"],
                    y=bbox_dict["y"],
                    width=bbox_dict["width"],
                    height=bbox_dict["height"],
                )

                # Run corner detection
                all_results = detect_corners_all_methods(filepath, bbox)

                # Generate and upload cropped images for each method
                cropped_urls = {}
                if all_results.best.success and all_results.best.corners:
                    url = generate_and_upload_cropped_image(
                        sample_id, filepath, all_results.best.corners, "best"
                    )
                    if url:
                        cropped_urls["best"] = url

                if all_results.hough_ransac.success and all_results.hough_ransac.corners:
                    url = generate_and_upload_cropped_image(
                        sample_id, filepath, all_results.hough_ransac.corners, "hough"
                    )
                    if url:
                        cropped_urls["hough"] = url

                if all_results.edge_intersection.success and all_results.edge_intersection.corners:
                    url = generate_and_upload_cropped_image(
                        sample_id, filepath, all_results.edge_intersection.corners, "edge"
                    )
                    if url:
                        cropped_urls["edge"] = url

                # Store result
                if store_corner_result(
                    run_id,
                    sample_id,
                    all_results.best,
                    all_results.hough_ransac,
                    all_results.edge_intersection,
                    human_corners,
                    cropped_urls,
                ):
                    success_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                print(f"[bulk_corners] Failed to process {sample_id}: {e}")
                failed_count += 1

            processed_count += 1

            # Report progress every 5 samples or on the last sample
            if processed_count % 5 == 0 or processed_count == total_count:
                report_corner_progress(run_id, processed_count, total_count, success_count, failed_count)
                print(f"[bulk_corners] Progress: {processed_count}/{total_count} (success={success_count}, failed={failed_count})")

        # Report completion
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        report_corner_complete(run_id, success_count, failed_count, duration_ms)
        print(f"[bulk_corners] Completed: success={success_count}, failed={failed_count}, duration={duration_ms}ms")

    except Exception as e:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        report_corner_failed(run_id, str(e), duration_ms)
        print(f"[bulk_corners] Failed: {e}")
