"""Board detection using Claude Code CLI for vision analysis."""

import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.lib import get_logger
from app.schemas import BOARD_SIZE, BoardCell, PlayerColor


class ClaudeDetector:
    """
    Detect Blokus board state using Claude Code CLI.

    Instead of complex CV pipelines, this sends the image to Claude
    and asks it to return a 20x20 JSON array of cell colors.
    """

    def __init__(self, timeout: int = 120):
        """
        Initialize Claude detector.

        Args:
            timeout: Timeout in seconds for Claude CLI calls
        """
        self.timeout = timeout
        self._claude_path: Optional[str] = None
        self._available = False
        self._check_cli()

    def _check_cli(self):
        """Check if Claude Code CLI is available."""
        self._claude_path = self._find_claude_executable()

        if not self._claude_path:
            print("Claude Code CLI not found")
            return

        try:
            result = subprocess.run(
                [self._claude_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self._available = True
                print(f"Claude Code CLI available: {result.stdout.strip()}")
            else:
                print("Claude Code CLI not available or not authenticated")
        except Exception as e:
            print(f"Error checking Claude Code CLI: {e}")

    def _find_claude_executable(self) -> Optional[str]:
        """Find the claude executable."""
        # First try PATH lookup
        claude_path = shutil.which("claude")
        if claude_path:
            return claude_path

        # Common installation locations
        home = os.path.expanduser("~")
        possible_paths = [
            "/usr/local/bin/claude",
            "/usr/bin/claude",
            "/opt/homebrew/bin/claude",
            os.path.join(home, ".npm-global/bin/claude"),
        ]

        # Check NVM versions
        nvm_dir = os.path.join(home, ".nvm/versions/node")
        if os.path.isdir(nvm_dir):
            for version in os.listdir(nvm_dir):
                possible_paths.append(os.path.join(nvm_dir, version, "bin/claude"))

        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return None

    def is_available(self) -> bool:
        """Check if Claude Code CLI is available."""
        return self._available

    def detect(self, image: np.ndarray, cid: str | None = None) -> list[list[BoardCell]]:
        """
        Detect board state using Claude vision.

        Args:
            image: BGR image as numpy array
            cid: Correlation ID for logging

        Returns:
            20x20 grid of BoardCell objects
        """
        log = get_logger(cid)

        if not self._available:
            log.error("claude_cli_not_available")
            raise RuntimeError("Claude Code CLI is not available")

        # Save image to temp file
        temp_path = self._save_temp_image(image)

        try:
            # Build prompt
            prompt = self._build_prompt(temp_path)

            log.info("calling_claude_cli", image_path=temp_path)

            # Call Claude Code CLI
            result = subprocess.run(
                [
                    self._claude_path,
                    "-p", prompt,
                    "--output-format", "text",
                    "--allowedTools", "Read",
                    "--max-turns", "3",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(temp_path),
            )

            if result.returncode != 0:
                log.error("claude_cli_error", stderr=result.stderr)
                raise RuntimeError(f"Claude CLI error: {result.stderr}")

            response = result.stdout.strip()
            log.info("claude_response_received", length=len(response))

            # Parse response into board state
            board_state = self._parse_response(response, cid)

            piece_count = sum(1 for row in board_state for cell in row if cell.color is not None)
            log.info("claude_detection_complete", pieces_detected=piece_count)

            return board_state

        except subprocess.TimeoutExpired:
            log.error("claude_cli_timeout")
            raise RuntimeError("Claude CLI timed out")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _save_temp_image(self, image: np.ndarray) -> str:
        """Save image to a temporary file."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Resize if too large
        max_size = 1024
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        # Save to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".jpg", prefix="blokus_")
        os.close(fd)
        pil_image.save(temp_path, format="JPEG", quality=90)

        return temp_path

    def _build_prompt(self, image_path: str) -> str:
        """Build the analysis prompt for Claude."""
        return f"""Please read and analyze the image at: {image_path}

This is a photo of a Blokus game board. The board is a 20x20 grid where players place colored pieces.

Analyze the board and return a JSON array representing the board state:
- The array should have exactly 20 rows
- Each row should have exactly 20 cells
- Each cell should be one of: "blue", "yellow", "red", "green", or null (for empty)
- Row 0 is the TOP of the board, Row 19 is the BOTTOM
- Column 0 is the LEFT, Column 19 is the RIGHT

Return ONLY the JSON array, no explanation or markdown formatting. Example format:
[[null, null, "blue", ...], [null, "red", null, ...], ...]

Important:
- Look carefully at each cell of the 20x20 grid
- The board frame/border is NOT part of the playable area
- Only mark cells that clearly contain a colored game piece
- If the board appears empty, return all nulls
- Be precise about which exact cells contain pieces"""

    def _parse_response(self, response: str, cid: str | None = None) -> list[list[BoardCell]]:
        """Parse Claude's response into board state."""
        log = get_logger(cid)

        # Try to extract JSON from response
        # Claude might include some text before/after the JSON
        json_start = response.find("[[")
        json_end = response.rfind("]]") + 2

        if json_start == -1 or json_end <= json_start:
            log.warning("no_json_found_in_response", response_preview=response[:200])
            # Return empty board
            return [[BoardCell(color=None) for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        json_str = response[json_start:json_end]

        # Handle escaped quotes from CLI output
        # The CLI sometimes returns \" instead of "
        if '\\"' in json_str:
            json_str = json_str.replace('\\"', '"')

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning("json_parse_error", error=str(e), json_preview=json_str[:200])
            return [[BoardCell(color=None) for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        # Validate and convert to BoardCell grid
        if not isinstance(data, list) or len(data) != BOARD_SIZE:
            log.warning("invalid_board_dimensions", rows=len(data) if isinstance(data, list) else 0)
            return [[BoardCell(color=None) for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        board_state: list[list[BoardCell]] = []
        valid_colors = {"blue", "yellow", "red", "green"}

        for row_idx, row in enumerate(data):
            if not isinstance(row, list) or len(row) != BOARD_SIZE:
                log.warning("invalid_row", row_idx=row_idx, cols=len(row) if isinstance(row, list) else 0)
                # Pad with empty cells
                row_cells = [BoardCell(color=None) for _ in range(BOARD_SIZE)]
            else:
                row_cells = []
                for cell in row:
                    if cell is None or cell not in valid_colors:
                        row_cells.append(BoardCell(color=None))
                    else:
                        row_cells.append(BoardCell(color=PlayerColor(cell)))
            board_state.append(row_cells)

        return board_state


# Singleton instance
_detector: ClaudeDetector | None = None


def get_claude_detector() -> ClaudeDetector:
    """Get the singleton Claude detector instance."""
    global _detector
    if _detector is None:
        _detector = ClaudeDetector()
    return _detector
