import cv2
import numpy as np

from app.schemas import BOARD_SIZE, BoardCell, PlayerColor

# HSV color ranges for each player color
# HSV: Hue (0-180), Saturation (0-255), Value (0-255)
COLOR_RANGES = {
    PlayerColor.BLUE: {
        "lower": np.array([100, 100, 50]),
        "upper": np.array([130, 255, 255]),
    },
    PlayerColor.YELLOW: {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([40, 255, 255]),
    },
    PlayerColor.RED: {
        # Red wraps around in HSV, so we need two ranges
        "lower1": np.array([0, 100, 100]),
        "upper1": np.array([10, 255, 255]),
        "lower2": np.array([160, 100, 100]),
        "upper2": np.array([180, 255, 255]),
    },
    PlayerColor.GREEN: {
        "lower": np.array([35, 100, 50]),
        "upper": np.array([85, 255, 255]),
    },
}

# Minimum percentage of colored pixels to consider cell occupied
MIN_COLORED_RATIO = 0.30


class CellClassifier:
    """Classify cell colors using HSV thresholding."""

    def __init__(self, min_colored_ratio: float = MIN_COLORED_RATIO):
        self.min_colored_ratio = min_colored_ratio

    def get_color_mask(self, hsv: np.ndarray, color: PlayerColor) -> np.ndarray:
        """Create a binary mask for a specific color."""
        ranges = COLOR_RANGES[color]

        if color == PlayerColor.RED:
            # Red needs two ranges due to HSV wraparound
            mask1 = cv2.inRange(hsv, ranges["lower1"], ranges["upper1"])
            mask2 = cv2.inRange(hsv, ranges["lower2"], ranges["upper2"])
            return cv2.bitwise_or(mask1, mask2)
        else:
            return cv2.inRange(hsv, ranges["lower"], ranges["upper"])

    def classify_cell(self, cell_img: np.ndarray) -> PlayerColor | None:
        """
        Classify a single cell image.
        Returns the detected color or None if empty.
        """
        if cell_img.size == 0:
            return None

        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        total_pixels = cell_img.shape[0] * cell_img.shape[1]

        best_color = None
        best_ratio = 0.0

        for color in PlayerColor:
            mask = self.get_color_mask(hsv, color)
            colored_pixels = cv2.countNonZero(mask)
            ratio = colored_pixels / total_pixels

            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color

        # Only return color if it exceeds minimum threshold
        if best_ratio >= self.min_colored_ratio:
            return best_color

        return None

    def classify_cells(self, cells: list[list[np.ndarray]]) -> list[list[BoardCell]]:
        """
        Classify all cells in the grid.
        Returns 20x20 grid of BoardCell objects.
        """
        board_state = []

        for row in range(BOARD_SIZE):
            row_cells = []
            for col in range(BOARD_SIZE):
                cell_img = cells[row][col]
                color = self.classify_cell(cell_img)
                row_cells.append(BoardCell(color=color))
            board_state.append(row_cells)

        return board_state
