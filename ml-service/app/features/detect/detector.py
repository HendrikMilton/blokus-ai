import cv2
import numpy as np
from pathlib import Path

from .schemas import BoundingBox, BoardDetection


def detect_board(image_path: Path) -> BoardDetection:
    """
    Detect a Blokus board in an image using contour-based detection.

    Algorithm:
    1. Convert to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Use adaptive threshold for varying lighting
    4. Find contours and filter for large quadrilaterals
    5. Score candidates by area and aspect ratio (prefer square shapes)
    6. Return normalized bounding box coordinates (0-1)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return BoardDetection(
            hasBoardDetected=False,
            isEntireBoardVisible=False,
            boundingBox=None,
            confidence=0.0,
        )

    h, w = img.shape[:2]

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter for quadrilaterals with good aspect ratio
    best_candidate: tuple[int, int, int, int] | None = None
    best_score = 0.0

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # We want quadrilaterals (4 vertices)
        if len(approx) == 4:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

            # Score: prefer large, square regions
            # Require at least 5% of image area and aspect ratio > 0.7
            # Reject if it covers nearly the entire image (likely false positive)
            min_area_ratio = 0.05
            max_area_ratio = 0.90
            min_aspect = 0.7

            area_ratio = area / (w * h)

            # Reject if covers almost entire image (all 4 edges touched)
            edge_margin = 5
            touches_all_edges = (
                x <= edge_margin and
                y <= edge_margin and
                x + bw >= w - edge_margin and
                y + bh >= h - edge_margin
            )

            if (area_ratio > min_area_ratio and
                area_ratio < max_area_ratio and
                aspect_ratio > min_aspect and
                not touches_all_edges):
                score = area * aspect_ratio
                if score > best_score:
                    best_score = score
                    best_candidate = (x, y, bw, bh)

    if best_candidate:
        x, y, bw, bh = best_candidate

        # Check if box touches image edges (partial board)
        margin = 5
        is_entire = (
            x > margin and
            y > margin and
            x + bw < w - margin and
            y + bh < h - margin
        )

        # Calculate confidence based on how well it matches expected properties
        # Max score would be full image area with perfect 1:1 aspect ratio
        max_possible_score = w * h * 1.0
        confidence = min(1.0, best_score / (max_possible_score * 0.5))

        return BoardDetection(
            hasBoardDetected=True,
            isEntireBoardVisible=is_entire,
            boundingBox=BoundingBox(
                x=x / w,
                y=y / h,
                width=bw / w,
                height=bh / h,
            ),
            confidence=confidence,
        )

    return BoardDetection(
        hasBoardDetected=False,
        isEntireBoardVisible=False,
        boundingBox=None,
        confidence=0.0,
    )
