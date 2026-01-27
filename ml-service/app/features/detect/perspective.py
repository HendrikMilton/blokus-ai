"""
Perspective correction for Blokus board images.

Multiple approaches:
1. Hough Lines (merged) - merge similar lines, find outermost
2. Hough Lines (extreme) - find single most extreme line in each direction
3. Color Segmentation - segment white board surface
4. Contour Detection - find largest quadrilateral contour
"""

import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .schemas import BoundingBox


@dataclass
class PerspectiveCorrectionResult:
    """Result from a single correction method."""
    method: str
    image: Optional[str]  # base64 encoded
    success: bool
    corners: Optional[list] = None  # The detected corners for debugging


@dataclass
class AllCorrectionsResult:
    """Results from all correction methods."""
    hough_lines_merged: PerspectiveCorrectionResult
    hough_lines_extreme: PerspectiveCorrectionResult
    color_segmentation: PerspectiveCorrectionResult
    contour_detection: PerspectiveCorrectionResult


def correct_perspective_all_methods(
    image_path: Path,
    bbox: BoundingBox,
    output_size: int = 400,
    padding: float = 0.1,
) -> AllCorrectionsResult:
    """
    Apply all perspective correction methods and return results.
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        empty = PerspectiveCorrectionResult(method="", image=None, success=False)
        return AllCorrectionsResult(
            hough_lines_merged=empty,
            hough_lines_extreme=empty,
            color_segmentation=empty,
            contour_detection=empty,
        )

    h, w = img.shape[:2]

    # Convert normalized bbox to pixel coordinates with padding
    pad_x = bbox.width * padding
    pad_y = bbox.height * padding

    x1 = int(max(0, (bbox.x - pad_x) * w))
    y1 = int(max(0, (bbox.y - pad_y) * h))
    x2 = int(min(w, (bbox.x + bbox.width + pad_x) * w))
    y2 = int(min(h, (bbox.y + bbox.height + pad_y) * h))

    # Crop to bbox region
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        empty = PerspectiveCorrectionResult(method="", image=None, success=False)
        return AllCorrectionsResult(
            hough_lines_merged=empty,
            hough_lines_extreme=empty,
            color_segmentation=empty,
            contour_detection=empty,
        )

    return AllCorrectionsResult(
        hough_lines_merged=try_hough_lines_merged(crop, output_size),
        hough_lines_extreme=try_hough_lines_extreme(crop, output_size),
        color_segmentation=try_color_segmentation(crop, output_size),
        contour_detection=try_contour_detection(crop, output_size),
    )


def correct_perspective(
    image_path: Path,
    bbox: BoundingBox,
    output_size: int = 400,
    padding: float = 0.05,
) -> Optional[str]:
    """
    Correct perspective distortion of the board region.
    Uses the best available method.
    """
    results = correct_perspective_all_methods(image_path, bbox, output_size, padding)

    # Try methods in order of preference
    for result in [results.hough_lines_extreme, results.color_segmentation,
                   results.contour_detection, results.hough_lines_merged]:
        if result.success and result.image:
            return result.image

    return None


# =============================================================================
# Method 1: Hough Lines (Merged) - Original approach
# =============================================================================

def try_hough_lines_merged(crop: np.ndarray, output_size: int) -> PerspectiveCorrectionResult:
    """Hough Lines with line merging in regions."""
    corners = find_board_corners_hough_merged(crop)
    if corners is not None:
        warped = apply_perspective_transform(crop, corners, output_size)
        return PerspectiveCorrectionResult(
            method="hough_lines_merged",
            image=encode_image_base64(warped),
            success=True,
            corners=corners.tolist(),
        )
    return PerspectiveCorrectionResult(
        method="hough_lines_merged",
        image=encode_image_base64(cv2.resize(crop, (output_size, output_size))),
        success=False,
    )


# =============================================================================
# Method 2: Hough Lines (Extreme) - Find single most extreme lines
# =============================================================================

def try_hough_lines_extreme(crop: np.ndarray, output_size: int) -> PerspectiveCorrectionResult:
    """Hough Lines finding the single most extreme line in each direction."""
    corners = find_board_corners_hough_extreme(crop)
    if corners is not None:
        warped = apply_perspective_transform(crop, corners, output_size)
        return PerspectiveCorrectionResult(
            method="hough_lines_extreme",
            image=encode_image_base64(warped),
            success=True,
            corners=corners.tolist(),
        )
    return PerspectiveCorrectionResult(
        method="hough_lines_extreme",
        image=encode_image_base64(cv2.resize(crop, (output_size, output_size))),
        success=False,
    )


# =============================================================================
# Method 3: Color Segmentation
# =============================================================================

def try_color_segmentation(crop: np.ndarray, output_size: int) -> PerspectiveCorrectionResult:
    """Segment white board surface and find corners."""
    corners = find_board_corners_color(crop)
    if corners is not None:
        warped = apply_perspective_transform(crop, corners, output_size)
        return PerspectiveCorrectionResult(
            method="color_segmentation",
            image=encode_image_base64(warped),
            success=True,
            corners=corners.tolist(),
        )
    return PerspectiveCorrectionResult(
        method="color_segmentation",
        image=encode_image_base64(cv2.resize(crop, (output_size, output_size))),
        success=False,
    )


# =============================================================================
# Method 4: Contour Detection
# =============================================================================

def try_contour_detection(crop: np.ndarray, output_size: int) -> PerspectiveCorrectionResult:
    """Find largest quadrilateral contour."""
    corners = find_board_corners_contour(crop)
    if corners is not None:
        warped = apply_perspective_transform(crop, corners, output_size)
        return PerspectiveCorrectionResult(
            method="contour_detection",
            image=encode_image_base64(warped),
            success=True,
            corners=corners.tolist(),
        )
    return PerspectiveCorrectionResult(
        method="contour_detection",
        image=encode_image_base64(cv2.resize(crop, (output_size, output_size))),
        success=False,
    )


def find_board_corners_hough_merged(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the 4 corners of the board using Hough Line detection.

    Returns:
        Array of 4 corner points [top-left, top-right, bottom-right, bottom-left]
        or None if detection fails
    """
    h, w = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Detect lines using Hough transform
    # Using HoughLinesP for line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min(h, w) // 4,
        maxLineGap=20
    )

    if lines is None or len(lines) < 4:
        return None

    # Separate lines into horizontal and vertical
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate angle
        if x2 - x1 == 0:
            angle = 90
        else:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize angle to 0-180 range
        angle = angle % 180

        # Horizontal: angle close to 0 or 180
        # Vertical: angle close to 90
        if angle < 30 or angle > 150:
            horizontal_lines.append((x1, y1, x2, y2))
        elif 60 < angle < 120:
            vertical_lines.append((x1, y1, x2, y2))

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None

    # Find the outermost lines
    # Top line: horizontal with smallest average y
    # Bottom line: horizontal with largest average y
    # Left line: vertical with smallest average x
    # Right line: vertical with largest average x

    def avg_y(line):
        return (line[1] + line[3]) / 2

    def avg_x(line):
        return (line[0] + line[2]) / 2

    # Sort and get extremes
    horizontal_lines.sort(key=avg_y)
    vertical_lines.sort(key=avg_x)

    # Merge nearby lines to get more robust edge detection
    top_line = merge_similar_lines([l for l in horizontal_lines if avg_y(l) < h * 0.4])
    bottom_line = merge_similar_lines([l for l in horizontal_lines if avg_y(l) > h * 0.6])
    left_line = merge_similar_lines([l for l in vertical_lines if avg_x(l) < w * 0.4])
    right_line = merge_similar_lines([l for l in vertical_lines if avg_x(l) > w * 0.6])

    if top_line is None or bottom_line is None or left_line is None or right_line is None:
        # Fallback: just use the extreme lines
        top_line = horizontal_lines[0] if horizontal_lines else None
        bottom_line = horizontal_lines[-1] if horizontal_lines else None
        left_line = vertical_lines[0] if vertical_lines else None
        right_line = vertical_lines[-1] if vertical_lines else None

    if None in [top_line, bottom_line, left_line, right_line]:
        return None

    # Calculate intersections
    top_left = line_intersection(top_line, left_line)
    top_right = line_intersection(top_line, right_line)
    bottom_right = line_intersection(bottom_line, right_line)
    bottom_left = line_intersection(bottom_line, left_line)

    if None in [top_left, top_right, bottom_right, bottom_left]:
        return None

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    # Validate corners are within reasonable bounds
    for corner in corners:
        if corner[0] < -w * 0.2 or corner[0] > w * 1.2:
            return None
        if corner[1] < -h * 0.2 or corner[1] > h * 1.2:
            return None

    # Validate that corners form a reasonable quadrilateral
    if not is_valid_quadrilateral(corners, w, h):
        return None

    return corners


def merge_similar_lines(lines: List[Tuple]) -> Optional[Tuple]:
    """
    Merge a list of similar lines into a single representative line.
    """
    if not lines:
        return None

    if len(lines) == 1:
        return lines[0]

    # Average all the line endpoints
    x1_avg = np.mean([l[0] for l in lines])
    y1_avg = np.mean([l[1] for l in lines])
    x2_avg = np.mean([l[2] for l in lines])
    y2_avg = np.mean([l[3] for l in lines])

    return (x1_avg, y1_avg, x2_avg, y2_avg)


def line_intersection(line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
    """
    Calculate the intersection point of two lines.
    Each line is defined by two points: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    return (px, py)


def is_valid_quadrilateral(corners: np.ndarray, img_w: int, img_h: int) -> bool:
    """
    Check if the detected corners form a valid quadrilateral.
    """
    # Check that the quadrilateral has reasonable area (at least 20% of image)
    area = cv2.contourArea(corners.astype(np.int32))
    img_area = img_w * img_h
    if area < 0.2 * img_area:
        return False

    # Check that corners are in roughly correct order (convex quadrilateral)
    # The cross product of consecutive edges should all have the same sign
    def cross_product_sign(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    signs = []
    n = len(corners)
    for i in range(n):
        sign = cross_product_sign(corners[i], corners[(i+1) % n], corners[(i+2) % n])
        signs.append(sign > 0)

    # All signs should be the same for a convex quadrilateral
    if not (all(signs) or not any(signs)):
        return False

    return True


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners as: top-left, top-right, bottom-right, bottom-left.
    """
    # Sort by y-coordinate (top vs bottom)
    sorted_by_y = corners[np.argsort(corners[:, 1])]

    # Top two points
    top_points = sorted_by_y[:2]
    # Bottom two points
    bottom_points = sorted_by_y[2:]

    # Sort top points by x (left to right)
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]

    # Sort bottom points by x (left to right)
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def apply_perspective_transform(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """
    Apply perspective transform to produce a square image.

    Args:
        image: Source image
        corners: 4 corner points [top-left, top-right, bottom-right, bottom-left]
        output_size: Size of output square

    Returns:
        Warped square image
    """
    # Define destination points (perfect square)
    dst_corners = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype=np.float32)

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, dst_corners)

    # Apply transform
    warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

    return warped


def encode_image_base64(image: np.ndarray, quality: int = 90) -> str:
    """Encode image as base64 JPEG string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', image, encode_params)
    return base64.b64encode(buffer).decode('utf-8')


def find_board_corners_hough_extreme(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find board corners using Hough Lines, picking the single most extreme line
    in each direction (topmost, bottommost, leftmost, rightmost).

    This avoids merging issues where inner grid lines get averaged in.
    Prioritizes longer lines (board edges) over short grid lines.
    """
    h, w = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection - use higher threshold to reduce grid line noise
    edges = cv2.Canny(blurred, 75, 200)

    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Detect lines using Hough transform
    # Require longer minimum line length to filter out short grid lines
    min_line_len = min(h, w) // 3  # Board edge should span at least 1/3 of image
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,  # Higher threshold for stronger edges
        minLineLength=min_line_len,
        maxLineGap=30  # Allow larger gaps for board edges that might be occluded
    )

    if lines is None or len(lines) < 4:
        return None

    def line_length(line):
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Separate lines into horizontal and vertical, filtering by length
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = line_length((x1, y1, x2, y2))

        # Calculate angle
        if x2 - x1 == 0:
            angle = 90
        else:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize angle to 0-180 range
        angle = angle % 180

        # Horizontal: angle close to 0 or 180
        # Vertical: angle close to 90
        if angle < 30 or angle > 150:
            horizontal_lines.append((x1, y1, x2, y2, length))
        elif 60 < angle < 120:
            vertical_lines.append((x1, y1, x2, y2, length))

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None

    # Find the most extreme lines, weighted by length
    # For each direction, consider lines in the outer 30% of the image
    # and pick the longest one (most likely to be the board edge)

    def avg_y(line):
        return (line[1] + line[3]) / 2

    def avg_x(line):
        return (line[0] + line[2]) / 2

    # Top line: horizontal lines in top 30%, pick longest
    top_candidates = [l for l in horizontal_lines if avg_y(l) < h * 0.35]
    if not top_candidates:
        top_candidates = sorted(horizontal_lines, key=avg_y)[:max(1, len(horizontal_lines) // 3)]
    top_line = max(top_candidates, key=lambda l: l[4])[:4]  # Get line without length

    # Bottom line: horizontal lines in bottom 30%, pick longest
    bottom_candidates = [l for l in horizontal_lines if avg_y(l) > h * 0.65]
    if not bottom_candidates:
        bottom_candidates = sorted(horizontal_lines, key=avg_y, reverse=True)[:max(1, len(horizontal_lines) // 3)]
    bottom_line = max(bottom_candidates, key=lambda l: l[4])[:4]

    # Left line: vertical lines in left 30%, pick longest
    left_candidates = [l for l in vertical_lines if avg_x(l) < w * 0.35]
    if not left_candidates:
        left_candidates = sorted(vertical_lines, key=avg_x)[:max(1, len(vertical_lines) // 3)]
    left_line = max(left_candidates, key=lambda l: l[4])[:4]

    # Right line: vertical lines in right 30%, pick longest
    right_candidates = [l for l in vertical_lines if avg_x(l) > w * 0.65]
    if not right_candidates:
        right_candidates = sorted(vertical_lines, key=avg_x, reverse=True)[:max(1, len(vertical_lines) // 3)]
    right_line = max(right_candidates, key=lambda l: l[4])[:4]

    # Calculate intersections
    top_left = line_intersection(top_line, left_line)
    top_right = line_intersection(top_line, right_line)
    bottom_right = line_intersection(bottom_line, right_line)
    bottom_left = line_intersection(bottom_line, left_line)

    if None in [top_left, top_right, bottom_right, bottom_left]:
        return None

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    # Validate corners are within reasonable bounds
    for corner in corners:
        if corner[0] < -w * 0.2 or corner[0] > w * 1.2:
            return None
        if corner[1] < -h * 0.2 or corner[1] > h * 1.2:
            return None

    # Validate quadrilateral
    if not is_valid_quadrilateral(corners, w, h):
        return None

    return corners


def find_board_corners_color(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find board corners by segmenting the light-colored board surface.
    Blokus boards are typically white/cream colored.
    """
    h, w = image.shape[:2]

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Segment light colored regions (white/cream board)
    # Low saturation, high value
    lower = np.array([0, 0, 150])
    upper = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Also try to include slightly colored regions (some boards have color)
    lower2 = np.array([0, 0, 120])
    upper2 = np.array([180, 100, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask, mask2)

    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Check if it's large enough
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < 0.2 * w * h:
        return None

    # Approximate to polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If we got 4 points, use them directly
    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
        corners = order_corners(corners)
        if is_valid_quadrilateral(corners, w, h):
            return corners

    # Otherwise, try to find the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    corners = order_corners(box.astype(np.float32))

    if is_valid_quadrilateral(corners, w, h):
        return corners

    return None


def find_board_corners_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find board corners using general contour detection.
    Looks for the largest quadrilateral-shaped contour.
    """
    h, w = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 200)

    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Look for quadrilateral contours
    for contour in contours[:10]:  # Check top 10 largest
        contour_area = cv2.contourArea(contour)

        # Skip if too small
        if contour_area < 0.15 * w * h:
            continue

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If 4 corners, we found a quadrilateral
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            corners = order_corners(corners)
            if is_valid_quadrilateral(corners, w, h):
                return corners

    # Fallback: use minimum area rectangle of largest contour
    if contours:
        largest = contours[0]
        if cv2.contourArea(largest) > 0.15 * w * h:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            corners = order_corners(box.astype(np.float32))
            if is_valid_quadrilateral(corners, w, h):
                return corners

    return None
