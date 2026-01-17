"""Board segmentation using SAM2 for precise mask generation."""

import cv2
import numpy as np

from app.lib import get_logger


class BoardSegmenter:
    """
    Segment the Blokus board using SAM2 (Segment Anything Model 2).

    Takes a bounding box from Grounding DINO and produces a precise
    segmentation mask for the board region.
    """

    # Model size options: tiny, small, base, large
    MODEL_SIZES = {
        "tiny": "sam2_hiera_t",
        "small": "sam2_hiera_s",
        "base": "sam2_hiera_b+",
        "large": "sam2_hiera_l",
    }

    def __init__(self, device: str = "cuda", model_size: str = "tiny"):
        """
        Initialize board segmenter.

        Args:
            device: Device to run on (cuda, cpu)
            model_size: SAM2 model size (tiny, small, base, large)
        """
        self.device = device
        self.model_size = model_size
        self.model = None
        self.predictor = None

    def _load_model(self):
        """Lazy load SAM2 model."""
        if self.predictor is not None:
            return

        import os

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model_name = self.MODEL_SIZES.get(self.model_size, "sam2_hiera_t")
            checkpoint_name = f"{model_name}.pt"
            config_name = f"sam2_hiera_{self.model_size[0]}.yaml"

            # Try to find checkpoint in common locations
            possible_paths = [
                checkpoint_name,
                f"checkpoints/{checkpoint_name}",
                os.path.expanduser(f"~/.cache/sam2/{checkpoint_name}"),
            ]

            checkpoint_path = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    checkpoint_path = abs_path
                    break

            if checkpoint_path is None:
                raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_name}")

            sam2_model = build_sam2(config_name, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.model = sam2_model

        except ImportError:
            # SAM2 not installed - we'll use fallback
            self.predictor = None
            self.model = None

    def segment(self, image: np.ndarray, bbox: np.ndarray, cid: str | None = None) -> np.ndarray:
        """
        Segment the board region using SAM2.

        Args:
            image: BGR image as numpy array (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2]
            cid: Correlation ID for logging

        Returns:
            Boolean mask of the board region (H, W)
        """
        log = get_logger(cid)
        self._load_model()

        if self.predictor is None:
            log.info("sam2_not_available_using_box_mask")
            return self._create_box_mask(image.shape[:2], bbox)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set image in predictor
        self.predictor.set_image(rgb_image)

        # Predict mask from bounding box
        masks, scores, _ = self.predictor.predict(
            box=bbox,
            multimask_output=False,
        )

        mask = masks[0].astype(bool)
        log.info("sam2_segmentation_complete", score=round(float(scores[0]), 3), mask_pixels=int(mask.sum()))

        return mask

    def _create_box_mask(self, image_size: tuple[int, int], bbox: np.ndarray) -> np.ndarray:
        """Create a simple rectangular mask from bounding box."""
        h, w = image_size
        mask = np.zeros((h, w), dtype=bool)

        x1, y1, x2, y2 = bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        mask[y1:y2, x1:x2] = True
        return mask

    def get_corners_from_mask(self, mask: np.ndarray, cid: str | None = None) -> np.ndarray | None:
        """
        Extract board corners from segmentation mask.

        Uses contour approximation to find the quadrilateral corners.

        Args:
            mask: Boolean mask of board region
            cid: Correlation ID for logging

        Returns:
            Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or None
        """
        log = get_logger(cid)

        # Convert boolean mask to uint8
        mask_uint8 = (mask.astype(np.uint8) * 255)

        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            log.warning("no_contours_found_in_mask")
            return None

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)

        # Approximate to polygon
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            log.info("corners_extracted_from_mask", corners=corners.tolist())
            return corners

        # If not a quadrilateral, use bounding rectangle corners
        x, y, w, h = cv2.boundingRect(largest)
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ])
        log.info("using_bounding_rect_corners", corners=corners.tolist())
        return corners

    def get_warped_board(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        output_size: int = 400,
        cid: str | None = None,
    ) -> np.ndarray:
        """
        Apply perspective transform to get bird's-eye view of board.

        Args:
            image: BGR image as numpy array
            corners: 4 corner points of the board
            output_size: Size of output square image
            cid: Correlation ID for logging

        Returns:
            Warped board image (output_size x output_size)
        """
        # Order corners: top-left, top-right, bottom-right, bottom-left
        ordered = self._order_corners(corners.astype(np.float32))

        # Destination points (square grid)
        dst = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ], dtype=np.float32)

        # Compute perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered, dst)

        # Apply transform
        warped = cv2.warpPerspective(image, matrix, (output_size, output_size))

        return warped

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
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
