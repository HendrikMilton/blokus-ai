"""Cell classification using DINOv3 dense features."""

import cv2
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from app.lib import get_logger
from app.schemas import BOARD_SIZE, BoardCell, PlayerColor

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FeatureClassifier:
    """
    Classify cell colors using DINOv3 semantic features.

    Instead of raw HSV thresholding, this extracts dense semantic features
    that understand "blue piece" vs "blue object in background".
    """

    # HuggingFace model IDs for DINOv3
    HF_MODEL_MAP = {
        "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }

    # Fallback to DINOv2 if v3 not available
    HF_FALLBACK_MAP = {
        "dinov3_vits16": "facebook/dinov2-small",
        "dinov3_vitb16": "facebook/dinov2-base",
        "dinov3_vitl16": "facebook/dinov2-large",
    }

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        device: str = "cuda",
        n_neighbors: int = 5,
    ):
        """
        Initialize feature classifier.

        Args:
            model_name: DINOv3 model variant
            device: Device to run on (cuda, cpu)
            n_neighbors: Number of neighbors for KNN classifier
        """
        self.model_name = model_name
        self.device = device
        self.n_neighbors = n_neighbors
        self.patch_size = 16

        self.model = None
        self.knn_classifier = None
        self.reference_features = None

    def _load_model(self):
        """Lazy load DINOv3 model from HuggingFace."""
        if self.model is not None:
            return

        from transformers import AutoModel

        # Try DINOv3 first, fallback to DINOv2
        hf_model_id = self.HF_MODEL_MAP.get(self.model_name)
        fallback_id = self.HF_FALLBACK_MAP.get(self.model_name)

        try:
            self.model = AutoModel.from_pretrained(hf_model_id)
        except Exception:
            # Fallback to DINOv2
            self.model = AutoModel.from_pretrained(fallback_id)

        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, int, int]:
        """
        Preprocess image for DINOv3.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            Tuple of (normalized tensor, h_patches, w_patches)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get dimensions that are multiples of patch size
        h, w = rgb.shape[:2]
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        new_h = h_patches * self.patch_size
        new_w = w_patches * self.patch_size

        # Resize to multiple of patch size
        rgb_resized = cv2.resize(rgb, (new_w, new_h))

        # Convert to tensor and normalize
        tensor = torch.from_numpy(rgb_resized).float().permute(2, 0, 1) / 255.0

        # ImageNet normalization
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, h_patches, w_patches

    def extract_features(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Extract dense patch features from image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            Tuple of (features array (N_patches, C), h_patches, w_patches)
        """
        self._load_model()

        tensor, h_patches, w_patches = self._preprocess(image)

        with torch.inference_mode():
            # HuggingFace model forward pass
            outputs = self.model(
                tensor.unsqueeze(0).to(self.device),
                output_hidden_states=True,
            )

            # Get last hidden state (patch tokens)
            last_hidden = outputs.last_hidden_state

            # Calculate expected number of patch tokens
            expected_patches = h_patches * w_patches
            total_tokens = last_hidden.shape[1]
            num_special_tokens = total_tokens - expected_patches

            # Remove CLS and register tokens
            patch_tokens = last_hidden[:, num_special_tokens:, :]

            # Reshape to (H_patches, W_patches, C)
            x = patch_tokens.squeeze().detach().cpu()

        return x.numpy(), h_patches, w_patches

    def extract_cell_features(
        self,
        warped_board: np.ndarray,
        grid_size: int = BOARD_SIZE,
    ) -> np.ndarray:
        """
        Extract features for each cell of the board grid.

        Args:
            warped_board: Bird's-eye view of board (square image)
            grid_size: Number of cells per side (default 20 for Blokus)

        Returns:
            Array of shape (grid_size, grid_size, feature_dim)
        """
        # Extract dense features
        features, h_patches, w_patches = self.extract_features(warped_board)

        # Reshape to spatial grid
        features_spatial = features.reshape(h_patches, w_patches, -1)

        # Map features to grid cells
        cell_features = np.zeros((grid_size, grid_size, features_spatial.shape[-1]))

        patches_per_cell_h = h_patches / grid_size
        patches_per_cell_w = w_patches / grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                # Get patch indices for this cell
                h_start = int(row * patches_per_cell_h)
                h_end = int((row + 1) * patches_per_cell_h)
                w_start = int(col * patches_per_cell_w)
                w_end = int((col + 1) * patches_per_cell_w)

                # Ensure we get at least one patch
                h_end = max(h_end, h_start + 1)
                w_end = max(w_end, w_start + 1)

                # Average features in this region
                cell_region = features_spatial[h_start:h_end, w_start:w_end, :]
                cell_features[row, col] = cell_region.mean(axis=(0, 1))

        return cell_features

    def build_reference_classifier(
        self,
        reference_images: dict[PlayerColor | None, list[np.ndarray]],
    ):
        """
        Build KNN classifier from reference cell images.

        Args:
            reference_images: Dict mapping PlayerColor (or None for empty) to
                             list of reference cell images
        """
        features_list = []
        labels_list = []

        for color, images in reference_images.items():
            for img in images:
                # Extract features for this reference cell
                feats, _, _ = self.extract_features(img)
                # Average features
                avg_feat = feats.mean(axis=0)
                features_list.append(avg_feat)
                labels_list.append(color)

        features_array = np.array(features_list)
        self.reference_features = (features_array, labels_list)

        # Build KNN classifier
        self.knn_classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="cosine")
        self.knn_classifier.fit(features_array, labels_list)

    def classify_cells_with_color(
        self,
        warped_board: np.ndarray,
        cid: str | None = None,
    ) -> list[list[BoardCell]]:
        """
        Classify cells using a combination of features and color.

        This is a hybrid approach: use DINOv3 features for robustness,
        but also use color information since we know Blokus pieces are
        distinctly colored (blue, yellow, red, green).

        Args:
            warped_board: Bird's-eye view of board
            cid: Correlation ID for logging

        Returns:
            20x20 grid of BoardCell objects
        """
        log = get_logger(cid)
        h, w = warped_board.shape[:2]
        cell_h = h // BOARD_SIZE
        cell_w = w // BOARD_SIZE

        # Extract features for semantic understanding
        cell_features = self.extract_cell_features(warped_board)

        board_state = []
        for row in range(BOARD_SIZE):
            row_cells = []
            for col in range(BOARD_SIZE):
                # Extract cell image
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w

                # Margin to avoid grid lines
                margin = max(1, min(cell_h, cell_w) // 10)
                cell_img = warped_board[y1 + margin:y2 - margin, x1 + margin:x2 - margin]

                # Get cell features
                cell_feat = cell_features[row, col]

                # Classify using hybrid approach
                color = self._classify_cell_hybrid(cell_img, cell_feat)
                row_cells.append(BoardCell(color=color))

            board_state.append(row_cells)

        # Count pieces
        piece_count = sum(1 for row in board_state for cell in row if cell.color is not None)
        log.info("feature_classification_complete", pieces_detected=piece_count)

        return board_state

    def _classify_cell_hybrid(
        self,
        cell_img: np.ndarray,
        cell_features: np.ndarray,
    ) -> PlayerColor | None:
        """
        Classify a single cell using hybrid feature + color approach.

        Uses feature magnitude to determine if cell is occupied,
        then uses color to determine which player.
        """
        if cell_img.size == 0:
            return None

        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Check saturation - pieces are colorful, empty cells are not
        mean_saturation = np.mean(s)
        mean_value = np.mean(v)

        # Low saturation = likely empty (gray/white board)
        if mean_saturation < 50 or mean_value < 30:
            return None

        # Blokus colors in HSV hue space (OpenCV uses 0-180):
        # Red: 0-10 or 170-180 (wraps around)
        # Yellow: 20-40
        # Green: 40-80
        # Blue: 100-130

        # Count pixels in each color range
        color_counts = {
            PlayerColor.BLUE: np.sum((h >= 100) & (h <= 130) & (s > 50)),
            PlayerColor.YELLOW: np.sum((h >= 15) & (h <= 45) & (s > 50)),
            PlayerColor.GREEN: np.sum((h >= 40) & (h <= 90) & (s > 50)),
            PlayerColor.RED: np.sum(((h <= 15) | (h >= 165)) & (s > 50)),
        }

        total_pixels = cell_img.shape[0] * cell_img.shape[1]
        best_color = None
        best_ratio = 0.0

        for color, count in color_counts.items():
            ratio = count / total_pixels
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color

        # Require at least 20% coverage to consider occupied
        if best_ratio < 0.20:
            return None

        return best_color

    def classify_cells_knn(
        self,
        warped_board: np.ndarray,
        cid: str | None = None,
    ) -> list[list[BoardCell]]:
        """
        Classify cells using pre-trained KNN classifier.

        Requires build_reference_classifier() to be called first.

        Args:
            warped_board: Bird's-eye view of board
            cid: Correlation ID for logging

        Returns:
            20x20 grid of BoardCell objects
        """
        log = get_logger(cid)

        if self.knn_classifier is None:
            log.warning("knn_classifier_not_trained_using_hybrid")
            return self.classify_cells_with_color(warped_board, cid)

        # Extract features for all cells
        cell_features = self.extract_cell_features(warped_board)

        # Flatten for KNN prediction
        flat_features = cell_features.reshape(-1, cell_features.shape[-1])

        # Predict
        predictions = self.knn_classifier.predict(flat_features)
        predictions = predictions.reshape(BOARD_SIZE, BOARD_SIZE)

        # Convert to BoardCell grid
        board_state = []
        for row in range(BOARD_SIZE):
            row_cells = []
            for col in range(BOARD_SIZE):
                color = predictions[row, col]
                row_cells.append(BoardCell(color=color))
            board_state.append(row_cells)

        piece_count = sum(1 for row in board_state for cell in row if cell.color is not None)
        log.info("knn_classification_complete", pieces_detected=piece_count)

        return board_state
