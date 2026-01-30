"""
Corner detection methods for Blokus board images.

Multiple approaches to detect the 4 corners of the board:
1. Best - Automatically selects between Hough and Edge based on quality metrics
2. Hough Lines + RANSAC - Detect grid lines, fit homography
3. Edge Intersection - Detect 4 outer edges, compute intersections

The "Best" method uses Hough by default, but falls back to Edge when
Hough's area_ratio < 0.83 (indicating it detected too small a region).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .schemas import BoundingBox, BoardCorners, Corner


@dataclass
class CornerQualityMetrics:
    """Quality metrics for evaluating corner detection results."""
    aspect_ratio: Optional[float] = None  # width/height, 1.0 = perfect square
    area_ratio: Optional[float] = None    # detected area / bbox area
    avg_angle: Optional[float] = None     # average corner angle in degrees, 90 = perfect


@dataclass
class CornerDetectionResult:
    """Result from a single corner detection method."""
    method: str
    corners: Optional[BoardCorners]
    success: bool
    debug_image: Optional[str] = None  # base64 encoded image with corners drawn
    quality: Optional[CornerQualityMetrics] = None


@dataclass
class AllCornerDetectionsResult:
    """Results from all corner detection methods."""
    best: CornerDetectionResult  # Automatically selected between hough and edge
    hough_ransac: CornerDetectionResult
    edge_intersection: CornerDetectionResult


# Thresholds for best method selection
BEST_METHOD_AREA_THRESHOLD = 0.83      # If Hough area_ratio < this, use Edge
BEST_METHOD_ASPECT_THRESHOLD = 0.70    # If Hough aspect_ratio < this, use Edge


def select_best_method(
    hough_result: CornerDetectionResult,
    edge_result: CornerDetectionResult,
) -> CornerDetectionResult:
    """
    Select the best corner detection result between Hough and Edge.

    Strategy: Use Hough by default (69% win rate overall), but fall back to Edge when:
    - Hough's area_ratio < 0.83 (indicating it detected too small a region)
    - Hough's aspect_ratio < 0.70 (indicating extreme aspect ratio / likely wrong detection)

    This achieves ~0.72% avg error vs 1.08% Hough-only.
    """
    # If Hough failed completely, use Edge
    if not hough_result.success or hough_result.corners is None:
        return CornerDetectionResult(
            method="best",
            corners=edge_result.corners,
            success=edge_result.success,
            debug_image=edge_result.debug_image,
            quality=edge_result.quality,
        )

    # If Edge failed completely, use Hough
    if not edge_result.success or edge_result.corners is None:
        return CornerDetectionResult(
            method="best",
            corners=hough_result.corners,
            success=hough_result.success,
            debug_image=hough_result.debug_image,
            quality=hough_result.quality,
        )

    # Both succeeded - use quality metrics to decide
    hough_area_ratio = hough_result.quality.area_ratio if hough_result.quality else None
    hough_aspect_ratio = hough_result.quality.aspect_ratio if hough_result.quality else None

    # If Hough area_ratio is too small (< 0.83), prefer Edge
    if hough_area_ratio is not None and hough_area_ratio < BEST_METHOD_AREA_THRESHOLD:
        return CornerDetectionResult(
            method="best",
            corners=edge_result.corners,
            success=edge_result.success,
            debug_image=edge_result.debug_image,
            quality=edge_result.quality,
        )

    # If Hough aspect_ratio is too extreme (< 0.70), prefer Edge
    if hough_aspect_ratio is not None and hough_aspect_ratio < BEST_METHOD_ASPECT_THRESHOLD:
        return CornerDetectionResult(
            method="best",
            corners=edge_result.corners,
            success=edge_result.success,
            debug_image=edge_result.debug_image,
            quality=edge_result.quality,
        )

    # Default: use Hough (better overall)
    return CornerDetectionResult(
        method="best",
        corners=hough_result.corners,
        success=hough_result.success,
        debug_image=hough_result.debug_image,
        quality=hough_result.quality,
    )


def detect_corners_all_methods(
    image_path: Path,
    bbox: BoundingBox,
    padding: float = 0.05,
) -> AllCornerDetectionsResult:
    """
    Apply all corner detection methods and return results.

    Args:
        image_path: Path to the image file
        bbox: Bounding box of the board (normalized 0-1)
        padding: Padding around bbox as fraction of bbox size (used for edge method)

    Returns:
        Results from all detection methods (best, hough, edge)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        empty = CornerDetectionResult(method="", corners=None, success=False)
        return AllCornerDetectionsResult(
            best=empty,
            hough_ransac=empty,
            edge_intersection=empty,
        )

    h, w = img.shape[:2]

    def make_crop(pad: float) -> Tuple[Optional[np.ndarray], int, int, int, int]:
        """Create a crop with specified padding. Returns (crop, x1, y1, crop_w, crop_h)."""
        pad_x = bbox.width * pad
        pad_y = bbox.height * pad

        x1 = int(max(0, (bbox.x - pad_x) * w))
        y1 = int(max(0, (bbox.y - pad_y) * h))
        x2 = int(min(w, (bbox.x + bbox.width + pad_x) * w))
        y2 = int(min(h, (bbox.y + bbox.height + pad_y) * h))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0, 0, 0, 0
        return crop, x1, y1, crop.shape[1], crop.shape[0]

    def make_converter(x1: int, y1: int, crop_w: int, crop_h: int):
        """Create a function to convert crop-relative corners to full-image normalized coordinates."""
        def convert_corners(result: CornerDetectionResult) -> CornerDetectionResult:
            if result.corners is None:
                return result

            def convert_point(corner: Corner) -> Corner:
                # Corner is in crop-relative normalized coords (0-1)
                # Convert to crop pixels
                px = corner.x * crop_w
                py = corner.y * crop_h
                # Add crop offset
                px += x1
                py += y1
                # Convert to full-image normalized
                return Corner(x=px / w, y=py / h)

            return CornerDetectionResult(
                method=result.method,
                corners=BoardCorners(
                    topLeft=convert_point(result.corners.topLeft),
                    topRight=convert_point(result.corners.topRight),
                    bottomRight=convert_point(result.corners.bottomRight),
                    bottomLeft=convert_point(result.corners.bottomLeft),
                ),
                success=result.success,
                debug_image=result.debug_image,
            )
        return convert_corners

    def add_quality_metrics(result: CornerDetectionResult) -> CornerDetectionResult:
        """Add quality metrics to a corner detection result."""
        if result.corners is None:
            return result
        quality = compute_quality_metrics(result.corners, bbox)
        return CornerDetectionResult(
            method=result.method,
            corners=result.corners,
            success=result.success,
            debug_image=result.debug_image,
            quality=quality,
        )

    # Create crops - Hough uses NO padding, Edge uses padding
    crop_padded, x1_p, y1_p, crop_w_p, crop_h_p = make_crop(padding)
    crop_no_pad, x1_n, y1_n, crop_w_n, crop_h_n = make_crop(0.0)

    empty = CornerDetectionResult(method="", corners=None, success=False)

    # Run Edge with padded crop
    if crop_padded is not None:
        convert_padded = make_converter(x1_p, y1_p, crop_w_p, crop_h_p)
        edge_result = add_quality_metrics(convert_padded(try_edge_intersection(crop_padded, crop_w_p, crop_h_p)))
    else:
        edge_result = empty

    # Run Hough with NO padding - board edges should be at crop boundary
    if crop_no_pad is not None:
        convert_no_pad = make_converter(x1_n, y1_n, crop_w_n, crop_h_n)
        hough_result = add_quality_metrics(convert_no_pad(try_hough_ransac(crop_no_pad, crop_w_n, crop_h_n)))
    else:
        hough_result = empty

    # Select best method based on quality metrics
    best_result = select_best_method(hough_result, edge_result)

    return AllCornerDetectionsResult(
        best=best_result,
        hough_ransac=hough_result,
        edge_intersection=edge_result,
    )


# =============================================================================
# Method 1: Hough Lines + RANSAC (moved from Method 2 position)
# =============================================================================


# =============================================================================
# Method 2: Hough Lines + RANSAC
# =============================================================================

def try_hough_ransac(crop: np.ndarray, w: int, h: int) -> CornerDetectionResult:
    """
    Detect corners using Hough lines to find grid intersections,
    then RANSAC to fit a homography to the ideal grid.
    """
    corners = find_corners_hough_ransac(crop)
    if corners is not None:
        normalized = corners_to_board_corners(corners, w, h)
        debug_img = draw_corners_on_image(crop, corners)
        return CornerDetectionResult(
            method="hough_ransac",
            corners=normalized,
            success=True,
            debug_image=debug_img,
        )

    debug_img = encode_image_base64(crop)
    return CornerDetectionResult(
        method="hough_ransac",
        corners=None,
        success=False,
        debug_image=debug_img,
    )


def find_corners_hough_ransac(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find board corners by detecting grid lines and fitting a homography.

    This is more robust than direct corner detection because:
    1. We detect many grid line intersections (not just 4 corners)
    2. RANSAC filters outliers when fitting the homography
    3. Corners are derived mathematically from the homography

    Returns:
        Array of 4 corner points in pixels [top-left, top-right, bottom-right, bottom-left]
        or None if detection fails
    """
    h, w = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min(h, w) // 6,  # Shorter lines to catch grid
        maxLineGap=10
    )

    if lines is None or len(lines) < 8:
        return None

    # Separate lines into horizontal and vertical
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            angle = 90
        else:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        angle = angle % 180

        if angle < 30 or angle > 150:
            horizontal_lines.append((x1, y1, x2, y2))
        elif 60 < angle < 120:
            vertical_lines.append((x1, y1, x2, y2))

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None

    # Find all line intersections
    intersections = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            pt = line_intersection(h_line, v_line)
            if pt is not None:
                px, py = pt
                # Only keep intersections within image bounds (with margin)
                if -w * 0.1 < px < w * 1.1 and -h * 0.1 < py < h * 1.1:
                    intersections.append(pt)

    if len(intersections) < 4:
        return None

    intersections = np.array(intersections, dtype=np.float32)

    # Cluster intersections to remove duplicates
    intersections = cluster_points(intersections, min_dist=min(w, h) // 30)

    if len(intersections) < 4:
        return None

    # Find the convex hull of intersections to get outer boundary
    hull = cv2.convexHull(intersections)

    # Approximate hull to 4 points
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # If we got 4 points, use them as corners
    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
        corners = order_corners(corners)
        if is_valid_quadrilateral(corners, w, h):
            return corners

    # Fallback: find extreme points
    if len(intersections) >= 4:
        corners = find_extreme_points(intersections, w, h)
        if corners is not None and is_valid_quadrilateral(corners, w, h):
            return corners

    return None


def cluster_points(points: np.ndarray, min_dist: float) -> np.ndarray:
    """
    Cluster nearby points and return cluster centers.
    Simple greedy clustering.
    """
    if len(points) == 0:
        return points

    clustered = []
    used = [False] * len(points)

    for i, pt in enumerate(points):
        if used[i]:
            continue

        # Find all points close to this one
        cluster = [pt]
        used[i] = True

        for j, other in enumerate(points):
            if used[j]:
                continue
            dist = np.sqrt((pt[0] - other[0])**2 + (pt[1] - other[1])**2)
            if dist < min_dist:
                cluster.append(other)
                used[j] = True

        # Add cluster center
        center = np.mean(cluster, axis=0)
        clustered.append(center)

    return np.array(clustered, dtype=np.float32)


def find_extreme_points(points: np.ndarray, w: int, h: int) -> Optional[np.ndarray]:
    """
    Find the 4 extreme corner points from a set of points.
    Uses sum and difference of coordinates to identify corners.
    """
    if len(points) < 4:
        return None

    # Top-left has smallest sum (x + y)
    # Bottom-right has largest sum
    # Top-right has smallest difference (y - x)
    # Bottom-left has largest difference

    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 1] - points[:, 0]

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return corners


# =============================================================================
# Method 3: Edge Intersection (4 outer edges → line intersections)
# =============================================================================

def try_edge_intersection(crop: np.ndarray, w: int, h: int) -> CornerDetectionResult:
    """
    Detect corners by finding the 4 outer edges and computing their intersections.
    """
    corners = find_corners_edge_intersection(crop)
    if corners is not None:
        normalized = corners_to_board_corners(corners, w, h)
        debug_img = draw_corners_on_image(crop, corners)
        return CornerDetectionResult(
            method="edge_intersection",
            corners=normalized,
            success=True,
            debug_image=debug_img,
        )

    debug_img = encode_image_base64(crop)
    return CornerDetectionResult(
        method="edge_intersection",
        corners=None,
        success=False,
        debug_image=debug_img,
    )


def compute_line_boundary_score(image: np.ndarray, line: Tuple, direction: str) -> float:
    """
    Compute a score for how likely a line is to be a board boundary.

    A true board edge should have white (board) on one side and darker (background) on the other.
    Internal grid lines have white on both sides → low contrast → low score.

    Args:
        image: BGR image
        line: (x1, y1, x2, y2, length) tuple
        direction: 'top', 'bottom', 'left', or 'right' - which edge we're looking for

    Returns:
        Score where higher = more likely to be a true board boundary
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = line[:4]

    # Convert to grayscale for brightness comparison
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sample points along the line
    num_samples = 10
    samples_inside = []
    samples_outside = []

    # Offset distance to sample (in pixels)
    offset = max(5, min(h, w) // 40)

    for i in range(num_samples):
        t = (i + 0.5) / num_samples
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))

        # Determine inside/outside based on direction
        if direction == 'top':
            inside_y, outside_y = py + offset, py - offset
            inside_x, outside_x = px, px
        elif direction == 'bottom':
            inside_y, outside_y = py - offset, py + offset
            inside_x, outside_x = px, px
        elif direction == 'left':
            inside_x, outside_x = px + offset, px - offset
            inside_y, outside_y = py, py
        else:  # right
            inside_x, outside_x = px - offset, px + offset
            inside_y, outside_y = py, py

        # Sample inside (should be white/bright - the board)
        if 0 <= inside_x < w and 0 <= inside_y < h:
            samples_inside.append(gray[inside_y, inside_x])

        # Sample outside (should be darker - the background)
        if 0 <= outside_x < w and 0 <= outside_y < h:
            samples_outside.append(gray[outside_y, outside_x])

    if not samples_inside or not samples_outside:
        return 0.0

    avg_inside = np.mean(samples_inside)
    avg_outside = np.mean(samples_outside)

    # Board edge: inside should be brighter (white board) than outside (background)
    # Score = brightness difference (inside - outside), normalized
    # Also reward if inside is actually bright (white)
    brightness_diff = avg_inside - avg_outside
    inside_brightness_bonus = avg_inside / 255.0  # 0-1, higher if inside is white

    # Combined score: contrast + bonus for white inside
    score = brightness_diff + inside_brightness_bonus * 50

    return score


def find_corners_edge_intersection(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find board corners by detecting the 4 outer edges and computing intersections.

    This approach:
    1. Detects all lines using Hough transform
    2. Finds the best boundary line in each direction (top, bottom, left, right)
       using color contrast (white board vs darker background)
    3. Computes the 4 corner intersections

    Returns:
        Array of 4 corner points in pixels [top-left, top-right, bottom-right, bottom-left]
        or None if detection fails
    """
    h, w = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection with higher threshold to focus on strong edges
    edges = cv2.Canny(blurred, 75, 200)

    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Detect lines - require longer lines (more likely to be board edges)
    min_line_len = min(h, w) // 4  # Slightly shorter to catch more candidates
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,  # Lower threshold to get more candidates
        minLineLength=min_line_len,
        maxLineGap=30
    )

    if lines is None or len(lines) < 4:
        return None

    # Separate lines into horizontal and vertical with length info
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if x2 - x1 == 0:
            angle = 90
        else:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        angle = angle % 180

        if angle < 30 or angle > 150:
            horizontal_lines.append((x1, y1, x2, y2, length))
        elif 60 < angle < 120:
            vertical_lines.append((x1, y1, x2, y2, length))

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None

    def avg_y(line):
        return (line[1] + line[3]) / 2

    def avg_x(line):
        return (line[0] + line[2]) / 2

    def select_best_line(candidates: List[Tuple], direction: str) -> Tuple:
        """Select the best line based on boundary score (color contrast)."""
        if len(candidates) == 1:
            return candidates[0][:4]

        # Score each candidate by how likely it is to be a true board boundary
        scored = []
        for line in candidates:
            boundary_score = compute_line_boundary_score(image, line, direction)
            # Combine boundary score with length (longer lines are better, but boundary score matters more)
            combined_score = boundary_score + line[4] * 0.1  # length is index 4
            scored.append((combined_score, line))

        # Return the line with highest score
        best = max(scored, key=lambda x: x[0])
        return best[1][:4]

    # Find candidate lines in each region
    # Top line: horizontal in top 40%
    top_candidates = [l for l in horizontal_lines if avg_y(l) < h * 0.4]
    if not top_candidates:
        top_candidates = sorted(horizontal_lines, key=avg_y)[:max(1, len(horizontal_lines) // 3)]
    top_line = select_best_line(top_candidates, 'top')

    # Bottom line: horizontal in bottom 40%
    bottom_candidates = [l for l in horizontal_lines if avg_y(l) > h * 0.6]
    if not bottom_candidates:
        bottom_candidates = sorted(horizontal_lines, key=avg_y, reverse=True)[:max(1, len(horizontal_lines) // 3)]
    bottom_line = select_best_line(bottom_candidates, 'bottom')

    # Left line: vertical in left 40%
    left_candidates = [l for l in vertical_lines if avg_x(l) < w * 0.4]
    if not left_candidates:
        left_candidates = sorted(vertical_lines, key=avg_x)[:max(1, len(vertical_lines) // 3)]
    left_line = select_best_line(left_candidates, 'left')

    # Right line: vertical in right 40%
    right_candidates = [l for l in vertical_lines if avg_x(l) > w * 0.6]
    if not right_candidates:
        right_candidates = sorted(vertical_lines, key=avg_x, reverse=True)[:max(1, len(vertical_lines) // 3)]
    right_line = select_best_line(right_candidates, 'right')

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

    if not is_valid_quadrilateral(corners, w, h):
        return None

    return corners


# =============================================================================
# Utility functions
# =============================================================================

def line_intersection(line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
    """
    Calculate the intersection point of two lines.
    Each line is defined by two points: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = line1[:4]
    x3, y3, x4, y4 = line2[:4]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    return (px, py)


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


def is_valid_quadrilateral(corners: np.ndarray, img_w: int, img_h: int) -> bool:
    """
    Check if the detected corners form a valid quadrilateral.
    """
    # Check that the quadrilateral has reasonable area (at least 15% of image)
    area = cv2.contourArea(corners.astype(np.int32))
    img_area = img_w * img_h
    if area < 0.15 * img_area:
        return False

    # Check that corners are in roughly correct order (convex quadrilateral)
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


def corners_to_board_corners(corners: np.ndarray, w: int, h: int) -> BoardCorners:
    """
    Convert pixel corners to normalized BoardCorners.
    """
    return BoardCorners(
        topLeft=Corner(x=corners[0][0] / w, y=corners[0][1] / h),
        topRight=Corner(x=corners[1][0] / w, y=corners[1][1] / h),
        bottomRight=Corner(x=corners[2][0] / w, y=corners[2][1] / h),
        bottomLeft=Corner(x=corners[3][0] / w, y=corners[3][1] / h),
    )


def draw_corners_on_image(image: np.ndarray, corners: np.ndarray) -> str:
    """
    Draw corners on image and return as base64 encoded JPEG.
    """
    img_copy = image.copy()

    # Draw lines connecting corners
    pts = corners.astype(np.int32)
    cv2.polylines(img_copy, [pts], True, (0, 255, 255), 2)  # Cyan lines

    # Draw corner points with labels
    colors = [
        (0, 0, 255),    # Red - TL
        (0, 255, 0),    # Green - TR
        (255, 0, 0),    # Blue - BR
        (0, 255, 255),  # Yellow - BL
    ]
    labels = ['TL', 'TR', 'BR', 'BL']

    for i, (pt, color, label) in enumerate(zip(pts, colors, labels)):
        cv2.circle(img_copy, tuple(pt), 8, color, -1)
        cv2.circle(img_copy, tuple(pt), 8, (255, 255, 255), 2)
        cv2.putText(img_copy, label, (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return encode_image_base64(img_copy)


def encode_image_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode image as base64 JPEG string."""
    import base64
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', image, encode_params)
    return base64.b64encode(buffer).decode('utf-8')


def compute_quality_metrics(
    corners: BoardCorners,
    bbox: BoundingBox,
) -> CornerQualityMetrics:
    """
    Compute quality metrics for detected corners.

    Args:
        corners: Detected corners (normalized 0-1)
        bbox: Original bounding box (normalized 0-1)

    Returns:
        Quality metrics for the detection
    """
    # Extract corner coordinates
    tl = np.array([corners.topLeft.x, corners.topLeft.y])
    tr = np.array([corners.topRight.x, corners.topRight.y])
    br = np.array([corners.bottomRight.x, corners.bottomRight.y])
    bl = np.array([corners.bottomLeft.x, corners.bottomLeft.y])

    # 1. Aspect ratio: width / height
    # Width = average of top edge and bottom edge lengths
    top_edge_len = np.linalg.norm(tr - tl)
    bottom_edge_len = np.linalg.norm(br - bl)
    avg_width = (top_edge_len + bottom_edge_len) / 2

    # Height = average of left edge and right edge lengths
    left_edge_len = np.linalg.norm(bl - tl)
    right_edge_len = np.linalg.norm(br - tr)
    avg_height = (left_edge_len + right_edge_len) / 2

    aspect_ratio = avg_width / avg_height if avg_height > 0 else 0

    # 2. Area ratio: detected area / bbox area
    # Use shoelace formula for quadrilateral area
    quad_pts = np.array([tl, tr, br, bl])
    n = len(quad_pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += quad_pts[i][0] * quad_pts[j][1]
        area -= quad_pts[j][0] * quad_pts[i][1]
    detected_area = abs(area) / 2

    bbox_area = bbox.width * bbox.height
    area_ratio = detected_area / bbox_area if bbox_area > 0 else 0

    # 3. Average corner angle (should be close to 90 degrees for a square)
    def angle_at_corner(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
        """Compute angle at vertex between vectors to p1 and p2."""
        v1 = p1 - vertex
        v2 = p2 - vertex
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))

    angle_tl = angle_at_corner(bl, tl, tr)
    angle_tr = angle_at_corner(tl, tr, br)
    angle_br = angle_at_corner(tr, br, bl)
    angle_bl = angle_at_corner(br, bl, tl)

    avg_angle = (angle_tl + angle_tr + angle_br + angle_bl) / 4

    return CornerQualityMetrics(
        aspect_ratio=float(aspect_ratio),
        area_ratio=float(area_ratio),
        avg_angle=float(avg_angle),
    )
