"""Board detection using Grounding DINO for zero-shot object detection."""

import numpy as np
import torch
from PIL import Image

from app.lib import get_logger


class BoardDetector:
    """
    Detect Blokus game boards using Grounding DINO zero-shot detection.

    Instead of fragile edge detection + contour heuristics, this uses
    text-prompt based detection: "blokus game board" -> bounding box.
    """

    # HuggingFace model ID for Grounding DINO
    MODEL_ID = "IDEA-Research/grounding-dino-base"

    # Detection prompts to try (in order of preference)
    DETECTION_PROMPTS = [
        "game board",
        "square board game",
        "blokus board",
        "grid board",
    ]

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.3):
        """
        Initialize board detector.

        Args:
            device: Device to run on (cuda, cpu)
            confidence_threshold: Minimum confidence for detection
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None

    def _load_model(self):
        """Lazy load the Grounding DINO model."""
        if self.model is not None:
            return

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray, cid: str | None = None) -> tuple[np.ndarray, float] | None:
        """
        Detect the game board in the image.

        Args:
            image: BGR image as numpy array (H, W, 3)
            cid: Correlation ID for logging

        Returns:
            Tuple of (bounding box [x1, y1, x2, y2], confidence) or None if not found
        """
        log = get_logger(cid)
        self._load_model()

        # Convert BGR to RGB PIL image
        rgb_image = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb_image)

        best_detection = None
        best_score = 0.0

        # Try different prompts
        for prompt in self.DETECTION_PROMPTS:
            detection = self._detect_with_prompt(pil_image, prompt)
            if detection is not None:
                bbox, score = detection
                if score > best_score:
                    best_score = score
                    best_detection = (bbox, score)
                    log.debug("board_detection_candidate", prompt=prompt, score=round(score, 3))

        if best_detection is not None:
            bbox, score = best_detection
            log.info("board_detected", score=round(score, 3), bbox=bbox.tolist())
            return best_detection

        log.warning("board_not_detected_by_grounding_dino")
        return None

    def _detect_with_prompt(self, pil_image: Image.Image, text_prompt: str) -> tuple[np.ndarray, float] | None:
        """
        Run detection with a specific text prompt.

        Args:
            pil_image: RGB PIL image
            text_prompt: Text prompt describing what to find

        Returns:
            Tuple of (bounding box, confidence) or None
        """
        # Process inputs
        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.confidence_threshold,
            text_threshold=0.25,
            target_sizes=[pil_image.size[::-1]],  # (height, width)
        )[0]

        if len(results["boxes"]) == 0:
            return None

        # Find the largest detection (most likely to be the board)
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Get index of largest box that meets confidence threshold
        best_idx = None
        best_area = 0
        for i, (area, score) in enumerate(zip(areas, scores)):
            if score >= self.confidence_threshold and area > best_area:
                best_area = area
                best_idx = i

        if best_idx is None:
            return None

        return boxes[best_idx], scores[best_idx]

    def detect_with_fallback(
        self, image: np.ndarray, fallback_corners: np.ndarray | None = None, cid: str | None = None
    ) -> np.ndarray:
        """
        Detect board with fallback to full image or provided corners.

        Args:
            image: BGR image as numpy array
            fallback_corners: Optional fallback corners if detection fails
            cid: Correlation ID for logging

        Returns:
            Bounding box [x1, y1, x2, y2] - always returns something
        """
        log = get_logger(cid)

        # Try Grounding DINO detection
        result = self.detect(image, cid)
        if result is not None:
            bbox, _ = result
            return bbox

        # Fallback to provided corners
        if fallback_corners is not None:
            x_min = fallback_corners[:, 0].min()
            y_min = fallback_corners[:, 1].min()
            x_max = fallback_corners[:, 0].max()
            y_max = fallback_corners[:, 1].max()
            log.info("using_fallback_corners")
            return np.array([x_min, y_min, x_max, y_max])

        # Ultimate fallback: entire image
        h, w = image.shape[:2]
        log.info("using_full_image_fallback")
        return np.array([0, 0, w, h])
