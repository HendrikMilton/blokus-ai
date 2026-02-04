"""
AlphaZero-style training for Blokus AI.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import torch

from app.logging_config import log

router = APIRouter()

# Models directory
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:3009")


# ============================================================================
# Schemas
# ============================================================================

class AlphaZeroTrainRequest(BaseModel):
    run_id: str
    config: Dict[str, Any]


class AlphaZeroStopRequest(BaseModel):
    pass


class AlphaZeroMoveRequest(BaseModel):
    game_state: Dict[str, Any]
    model_path: str
    mcts_simulations: int = 400


class MoveResponse(BaseModel):
    piece_idx: int
    orientation_idx: int
    row: int
    col: int
    policy: List[float]
    value: float


# ============================================================================
# Routes
# ============================================================================

@router.post("/train")
async def start_training(request: AlphaZeroTrainRequest, background_tasks: BackgroundTasks):
    """Start AlphaZero training in background."""
    from .trainer import start_training

    log.info(f"Received training request for run: {request.run_id}")

    # Start training in background
    background_tasks.add_task(
        start_training,
        request.run_id,
        request.config,
        MODELS_DIR,
        API_BASE_URL,
    )

    return {"status": "started", "run_id": request.run_id}


@router.post("/stop/{run_id}")
async def stop_training(run_id: str):
    """Stop a running training."""
    from .trainer import stop_training

    stop_training(run_id)
    return {"status": "stop_requested", "run_id": run_id}


@router.delete("/models/{run_id}")
async def delete_models(run_id: str):
    """Delete model files for a training run."""
    import shutil

    model_dir = MODELS_DIR / "alphazero" / run_id
    if model_dir.exists():
        shutil.rmtree(model_dir)
        return {"status": "deleted", "path": str(model_dir)}

    return {"status": "not_found", "path": str(model_dir)}


@router.post("/move", response_model=MoveResponse)
async def get_move(request: AlphaZeroMoveRequest):
    """Get AI move for a game state using a trained model."""
    from .game import BlokusGame, Move, PLAYER_NAMES
    from .network import BlokusNet, create_network
    from .mcts import MCTS, MCTSConfig

    # Load model
    model_path = Path(request.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    num_res_blocks = config.get('num_res_blocks', 15)
    channels = config.get('channels', 256)

    # Detect input channels from checkpoint
    input_channels = checkpoint['model_state_dict']['stem.0.weight'].shape[1]

    # Create and load network
    network = create_network(num_res_blocks=num_res_blocks, channels=channels, input_channels=input_channels, device=device)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    # Reconstruct game state
    game = _dict_to_game(request.game_state)

    # Get legal moves
    legal_moves = game.get_legal_moves()
    if not legal_moves:
        raise HTTPException(status_code=400, detail="No legal moves available")

    # Run MCTS
    mcts_config = MCTSConfig(
        num_simulations=request.mcts_simulations,
        temperature_threshold=0,  # Always use low temperature for inference
    )
    mcts = MCTS(network, mcts_config, device)

    move, policy, root = mcts.select_move(game, add_noise=False, temperature=0.1)

    # Get value from root
    value = root.q_value if root.visit_count > 0 else 0.0

    return MoveResponse(
        piece_idx=move.piece_idx,
        orientation_idx=move.orientation_idx,
        row=move.row,
        col=move.col,
        policy=policy.tolist(),
        value=value,
    )


def _dict_to_game(state: Dict[str, Any]) -> 'BlokusGame':
    """Convert a game state dictionary to a BlokusGame object."""
    from .game import BlokusGame, PlayerState, PLAYER_NAMES
    from .pieces import PIECE_INDEX_BY_ID
    import numpy as np

    game = BlokusGame()

    # Set board
    board = state.get('board', [])
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell is not None:
                # Map color name to player index
                player_idx = PLAYER_NAMES.index(cell) if cell in PLAYER_NAMES else 0
                game.board[r, c] = player_idx + 1

    # Set player states
    players = state.get('players', {})
    for color, pstate in players.items():
        if color in PLAYER_NAMES:
            player_idx = PLAYER_NAMES.index(color)

            # Convert remaining pieces from IDs to indices
            remaining = pstate.get('remainingPieces', [])
            game.players[player_idx].remaining_pieces = {
                PIECE_INDEX_BY_ID.get(pid, 0) for pid in remaining
            }
            game.players[player_idx].has_played = pstate.get('hasPlayed', False)
            game.players[player_idx].is_eliminated = pstate.get('isEliminated', False)
            game.players[player_idx].score = pstate.get('score', 0)

    # Set current player
    current = state.get('currentPlayer', 'blue')
    if current in PLAYER_NAMES:
        game.current_player = PLAYER_NAMES.index(current)

    # Set other state
    game.turn_number = state.get('turnNumber', 1)
    game.is_game_over = state.get('isGameOver', False)
    game.consecutive_passes = state.get('consecutivePasses', 0)

    # Rebuild valid corners (simplified - just compute from scratch)
    game._valid_corners = [set() for _ in range(4)]
    for player_idx in range(4):
        if not game.players[player_idx].has_played:
            from .game import STARTING_CORNERS
            game._valid_corners[player_idx].add(STARTING_CORNERS[player_idx])
        else:
            # Scan board for diagonal attachment points
            player_color = player_idx + 1
            for r in range(20):
                for c in range(20):
                    if game.board[r, c] != 0:
                        continue
                    # Check if diagonally adjacent to own color
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 20 and 0 <= nc < 20 and game.board[nr, nc] == player_color:
                            # Check not edge adjacent
                            is_edge = False
                            for edr, edc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                er, ec = r + edr, c + edc
                                if 0 <= er < 20 and 0 <= ec < 20 and game.board[er, ec] == player_color:
                                    is_edge = True
                                    break
                            if not is_edge:
                                game._valid_corners[player_idx].add((r, c))
                                break

    return game


class OnnxExportRequest(BaseModel):
    checkpoint_id: Optional[str] = None


class OnnxExportResponse(BaseModel):
    run_id: str
    onnx_path: str
    file_size_mb: float
    input_shapes: Dict[str, Any]
    output_shapes: Dict[str, Any]
    output_names: List[str]


@router.post("/export/onnx/{run_id}", response_model=OnnxExportResponse)
async def export_to_onnx(run_id: str, request: OnnxExportRequest = None):
    """
    Export AlphaZero model to ONNX format.

    Args:
        run_id: Training run ID
        request: Optional request body with checkpoint_id

    Returns:
        Export metadata including path, size, and tensor shapes
    """
    from .network import BlokusNet, create_network, export_to_onnx as onnx_export

    # Determine checkpoint path
    model_dir = MODELS_DIR / "alphazero" / run_id
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")

    checkpoint_id = request.checkpoint_id if request else None
    if checkpoint_id:
        # Use specific checkpoint
        model_path = model_dir / f"{checkpoint_id}.pt"
    else:
        # Use best.pt by default, fall back to latest checkpoint
        model_path = model_dir / "best.pt"
        if not model_path.exists():
            # Find the latest checkpoint file
            checkpoint_files = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoint_files:
                model_path = checkpoint_files[0]
                log.info(f"No best.pt found, using latest checkpoint: {model_path.name}")

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"No checkpoint found in {model_dir}")

    log.info(f"Exporting model to ONNX: {model_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('config', {})
    num_res_blocks = config.get('num_res_blocks', 15)
    channels = config.get('channels', 256)

    # Detect input channels from the checkpoint's stem layer
    state_dict = checkpoint['model_state_dict']
    stem_weight_shape = state_dict['stem.0.weight'].shape
    input_channels = stem_weight_shape[1]  # Shape is [out_channels, in_channels, H, W]
    actual_channels = stem_weight_shape[0]
    log.info(f"Detected model: {input_channels} input channels, {actual_channels} channels, {num_res_blocks} res blocks")

    # Create network with correct architecture
    network = BlokusNet(
        num_res_blocks=num_res_blocks,
        channels=actual_channels,
    )
    # Override INPUT_CHANNELS if different from default
    if input_channels != BlokusNet.INPUT_CHANNELS:
        # Recreate stem with correct input channels
        import torch.nn as nn
        network.stem = nn.Sequential(
            nn.Conv2d(input_channels, actual_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(actual_channels),
            nn.ReLU(),
        )
    network = network.to(device)
    network.load_state_dict(state_dict)
    network.eval()

    # Export to ONNX
    onnx_path = model_dir / "model.onnx"
    export_metadata = onnx_export(network, str(onnx_path), input_channels=input_channels)

    log.info(f"ONNX export complete: {onnx_path} ({export_metadata['file_size_mb']} MB)")

    return OnnxExportResponse(
        run_id=run_id,
        onnx_path=str(onnx_path),
        file_size_mb=export_metadata['file_size_mb'],
        input_shapes=export_metadata['input_shapes'],
        output_shapes=export_metadata['output_shapes'],
        output_names=export_metadata['output_names'],
    )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
