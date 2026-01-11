import base64

import cv2
import numpy as np


class Preprocessor:
    """Image preprocessing for board detection."""

    def __init__(self, target_size: int = 800):
        self.target_size = target_size

    def decode_base64(self, image_b64: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        # Handle data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        return img

    def resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio."""
        h, w = img.shape[:2]
        scale = self.target_size / max(h, w)

        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return img

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)

        lab = cv2.merge([lightness, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        return cv2.GaussianBlur(img, (5, 5), 0)

    def process(self, image_b64: str) -> np.ndarray:
        """Full preprocessing pipeline."""
        img = self.decode_base64(image_b64)
        img = self.resize(img)
        img = self.apply_clahe(img)
        img = self.denoise(img)
        return img
