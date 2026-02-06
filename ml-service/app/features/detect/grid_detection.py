"""
Grid color detection for Blokus board images.

Analyzes a 400x400 cropped board image and classifies each of the 20x20 grid cells
as empty, blue, yellow, red, or green using HSV color space analysis.
"""

import time
from typing import Optional

import cv2
import httpx
import numpy as np
from pydantic import BaseModel



class GridPrediction(BaseModel):
    grid: list[list[Optional[str]]]  # 20x20 grid of color names or None
    confidence: float  # Average confidence across all cells
    cellConfidences: list[list[float]]  # 20x20 grid of per-cell confidence
    processingTimeMs: float


# HSV color ranges for Blokus colors
# OpenCV uses H: 0-180, S: 0-255, V: 0-255
# Note: Hue is half of the standard 0-360 range

COLOR_RANGES = {
    # Blue: #0066CC -> HSV approx (210, 100%, 80%) -> OpenCV (105, 255, 204)
    # Extended hue range to catch edge cases (hue ~126-130)
    "blue": {
        "h_min": 95,
        "h_max": 130,
        "s_min": 80,
        "v_min": 80,
    },
    # Yellow: #FFD700 -> HSV approx (51, 100%, 100%) -> OpenCV (25, 255, 255)
    "yellow": {
        "h_min": 18,
        "h_max": 38,
        "s_min": 100,
        "v_min": 150,
    },
    # Red: #DC143C -> HSV approx (348, 91%, 86%) -> OpenCV (174 or ~0, 232, 220)
    # Red wraps around, so we check both ends
    "red": {
        "h_ranges": [(0, 12), (165, 180)],  # Two ranges for red hue wrap
        "s_min": 80,
        "v_min": 80,
    },
    # Green: #228B22 -> HSV approx (120, 76%, 55%) -> OpenCV (60, 194, 140)
    # Extended hue range to catch cyan-ish greens (hue ~86-90)
    "green": {
        "h_min": 40,
        "h_max": 95,
        "s_min": 40,
        "v_min": 50,
    },
}

# Saturation threshold below which a cell is considered empty (white/gray)
EMPTY_SATURATION_THRESHOLD = 45


def download_image(image_url: str) -> np.ndarray:
    """Download image from URL and decode to OpenCV format."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(image_url)
        response.raise_for_status()
        image_bytes = response.content

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def classify_cell(cell_bgr: np.ndarray, row: int = -1, col: int = -1, debug: bool = False) -> tuple[Optional[str], float]:
    """
    Classify a single cell's color.

    Args:
        cell_bgr: BGR image of the cell (typically 20x20 pixels)
        row: Row index for debug logging
        col: Column index for debug logging
        debug: Whether to print debug info

    Returns:
        Tuple of (color_name or None for empty, confidence 0-1)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)

    # Use center region to avoid grid lines (edges)
    # For a 20x20 cell, use the center 12x12 (pixels 4-16)
    h, w = hsv.shape[:2]
    margin = max(1, min(4, h // 5, w // 5))
    center = hsv[margin : h - margin, margin : w - margin]

    if center.size == 0:
        return None, 1.0

    # Get average saturation to check if empty
    avg_saturation = float(center[:, :, 1].mean())

    if avg_saturation < EMPTY_SATURATION_THRESHOLD:
        # Low saturation = white/gray = empty cell
        confidence = 1.0 - (avg_saturation / EMPTY_SATURATION_THRESHOLD)
        if debug:
            print(f"  [{row},{col}] EMPTY (low sat): avg_sat={avg_saturation:.1f} < {EMPTY_SATURATION_THRESHOLD}")
        return None, max(0.5, min(1.0, confidence))

    # Get hue histogram to find dominant hue
    hue_channel = center[:, :, 0].flatten()
    sat_channel = center[:, :, 1].flatten()

    # Only consider pixels with sufficient saturation
    mask = sat_channel > 40
    if not mask.any():
        return None, 0.5

    hue_values = hue_channel[mask]

    # Calculate histogram
    hist, _ = np.histogram(hue_values, bins=180, range=(0, 180))
    dominant_hue = int(np.argmax(hist))

    # Classify based on hue ranges
    best_color = None
    best_confidence = 0.0

    for color_name, ranges in COLOR_RANGES.items():
        if "h_ranges" in ranges:
            # Special handling for red (wraps around)
            in_range = False
            for h_min, h_max in ranges["h_ranges"]:
                if h_min <= dominant_hue <= h_max:
                    in_range = True
                    break
            if not in_range:
                continue
        else:
            h_min = ranges["h_min"]
            h_max = ranges["h_max"]
            if not (h_min <= dominant_hue <= h_max):
                continue

        # Check saturation and value thresholds
        s_min = ranges["s_min"]
        v_min = ranges["v_min"]
        avg_sat = float(center[:, :, 1].mean())
        avg_val = float(center[:, :, 2].mean())

        if avg_sat < s_min or avg_val < v_min:
            continue

        # Calculate confidence based on how well it matches the range
        if "h_ranges" in ranges:
            # For red, find the closest range center
            h_centers = [(r[0] + r[1]) / 2 for r in ranges["h_ranges"]]
            h_center = min(h_centers, key=lambda c: min(abs(dominant_hue - c), 180 - abs(dominant_hue - c)))
            h_dist = min(abs(dominant_hue - h_center), 180 - abs(dominant_hue - h_center))
            h_range_size = 15
        else:
            h_center = (h_min + h_max) / 2
            h_range_size = (h_max - h_min) / 2
            h_dist = abs(dominant_hue - h_center)

        hue_confidence = max(0, 1 - (h_dist / max(1, h_range_size)))
        sat_confidence = min(1.0, (avg_sat - s_min) / max(1, 255 - s_min))
        val_confidence = min(1.0, (avg_val - v_min) / max(1, 255 - v_min))

        confidence = (hue_confidence * 0.6 + sat_confidence * 0.2 + val_confidence * 0.2)

        if confidence > best_confidence:
            best_confidence = confidence
            best_color = color_name

    if best_color:
        if debug:
            print(f"  [{row},{col}] {best_color.upper()}: dom_hue={dominant_hue}, avg_sat={avg_saturation:.1f}, avg_val={float(center[:,:,2].mean()):.1f}, conf={best_confidence:.2f}")
        return best_color, best_confidence

    # No strong match - classify as empty
    if debug:
        avg_val = float(center[:, :, 2].mean())
        print(f"  [{row},{col}] EMPTY (no match): dom_hue={dominant_hue}, avg_sat={avg_saturation:.1f}, avg_val={avg_val:.1f}")
    return None, 0.3


def _detect_grid_from_image(img: np.ndarray) -> GridPrediction:
    """
    Core grid detection logic operating on an already-loaded BGR image.
    """
    start_time = time.perf_counter()

    # Resize to exactly 400x400 if needed
    if img.shape[0] != 400 or img.shape[1] != 400:
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)

    # Grid size
    grid_size = 20
    cell_size = 400 // grid_size  # 20 pixels per cell

    # Initialize result grids
    grid: list[list[Optional[str]]] = []
    cell_confidences: list[list[float]] = []

    total_confidence = 0.0
    cell_count = 0

    for row in range(grid_size):
        grid_row: list[Optional[str]] = []
        conf_row: list[float] = []

        for col in range(grid_size):
            # Extract cell
            y1 = row * cell_size
            y2 = y1 + cell_size
            x1 = col * cell_size
            x2 = x1 + cell_size

            cell = img[y1:y2, x1:x2]

            # Debug cells in the problem area (bottom-right corner)
            debug_cell = (row >= 17) and (col >= 17)
            if debug_cell:
                print(f"DEBUG: Processing cell [{row},{col}]")

            # Classify cell
            color, confidence = classify_cell(cell, row, col, debug=debug_cell)

            grid_row.append(color)
            conf_row.append(round(confidence, 3))
            total_confidence += confidence
            cell_count += 1

        grid.append(grid_row)
        cell_confidences.append(conf_row)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    avg_confidence = total_confidence / cell_count if cell_count > 0 else 0.0

    return GridPrediction(
        grid=grid,
        confidence=round(avg_confidence, 3),
        cellConfidences=cell_confidences,
        processingTimeMs=round(elapsed_ms, 2),
    )


def detect_grid_colors_from_bytes(image_bytes: bytes) -> GridPrediction:
    """
    Detect colors in a 20x20 grid from raw image bytes (e.g. base64-decoded).

    Args:
        image_bytes: Raw image bytes (JPEG/PNG)

    Returns:
        GridPrediction with the classified grid and confidence metrics
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return _detect_grid_from_image(img)


def detect_grid_colors(image_url: str) -> GridPrediction:
    """
    Detect colors in a 20x20 grid from a cropped board image.

    Args:
        image_url: URL to a 400x400 cropped board image

    Returns:
        GridPrediction with the classified grid and confidence metrics
    """
    img = download_image(image_url)
    return _detect_grid_from_image(img)
