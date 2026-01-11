import cv2
import numpy as np

from app.schemas import BOARD_SIZE


class GridFinder:
    """Find and extract the Blokus board grid from an image."""

    def __init__(self, grid_size: int = 400):
        # Output grid size (grid_size x grid_size pixels)
        self.grid_size = grid_size
        self.cell_size = grid_size // BOARD_SIZE  # 20 pixels per cell

    def find_board_contour(self, img: np.ndarray) -> np.ndarray | None:
        """Find the largest quadrilateral contour (the board)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect edge segments
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Sort by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:  # Check top 5 largest
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # If it has 4 corners, it's likely the board
            if len(approx) == 4:
                return approx.reshape(4, 2)

        return None

    def order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum of coordinates: top-left has smallest, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        # Difference: top-right has smallest, bottom-left has largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def perspective_transform(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Apply perspective transform to get bird's-eye view of board."""
        ordered = self.order_corners(corners.astype(np.float32))

        # Destination points (square grid)
        dst = np.array([
            [0, 0],
            [self.grid_size - 1, 0],
            [self.grid_size - 1, self.grid_size - 1],
            [0, self.grid_size - 1],
        ], dtype=np.float32)

        # Compute perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered, dst)

        # Apply transform
        warped = cv2.warpPerspective(img, matrix, (self.grid_size, self.grid_size))

        return warped

    def extract_cells(self, grid_img: np.ndarray) -> list[list[np.ndarray]]:
        """Extract individual cell images from the grid."""
        cells = []

        for row in range(BOARD_SIZE):
            row_cells = []
            for col in range(BOARD_SIZE):
                y1 = row * self.cell_size
                y2 = (row + 1) * self.cell_size
                x1 = col * self.cell_size
                x2 = (col + 1) * self.cell_size

                # Extract cell with small margin to avoid grid lines
                margin = 2
                cell = grid_img[y1 + margin:y2 - margin, x1 + margin:x2 - margin]
                row_cells.append(cell)

            cells.append(row_cells)

        return cells

    def find_grid(self, img: np.ndarray) -> list[list[np.ndarray]] | None:
        """
        Main method: find board and extract cell images.
        Returns 20x20 list of cell images, or None if board not found.
        """
        corners = self.find_board_contour(img)

        if corners is None:
            # If no board found, assume entire image is the board
            h, w = img.shape[:2]
            corners = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ])

        grid_img = self.perspective_transform(img, corners)
        cells = self.extract_cells(grid_img)

        return cells
