"""Service for training data management and model training."""

from __future__ import annotations

import base64
import io
import pickle
import time
from uuid import UUID

import cv2
import numpy as np
from PIL import Image

from app.lib.log import get_logger
from app.schemas.admin import (
    AdminStats,
    SampleSource,
    SampleStatus,
    TrainingSampleResponse,
    TrainingSampleUpdate,
    TrainResponse,
)
from app.schemas.ml import BoardCell, PlayerColor

from .repository import TrainingSampleRepository, get_repository


class TrainingDataService:
    """Service for managing training data and model training."""

    def __init__(self, repository: TrainingSampleRepository | None = None):
        """Initialize service with repository."""
        self.repository = repository or get_repository()
        self._detection_service = None
        self._claude_detector = None

    @property
    def detection_service(self):
        """Lazy load detection service to avoid circular imports."""
        if self._detection_service is None:
            from app.features.detection.service import DetectionService
            self._detection_service = DetectionService()
        return self._detection_service

    @property
    def claude_detector(self):
        """Lazy load Claude detector."""
        if self._claude_detector is None:
            from app.features.detection.claude_detector import get_claude_detector
            self._claude_detector = get_claude_detector()
        return self._claude_detector

    def upload_sample(
        self,
        image_b64: str,
        source: SampleSource = SampleSource.UPLOAD,
        notes: str | None = None,
        cid: str | None = None,
        use_claude: bool = True,
    ) -> TrainingSampleResponse:
        """Upload an image, run detection, and store as a training sample."""
        log = get_logger(cid)
        start = time.perf_counter()

        # Decode image
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            image_width, image_height = image.size

            # Convert to bytes for storage (as JPEG)
            buffer = io.BytesIO()
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=90)
            image_bytes = buffer.getvalue()
        except Exception as e:
            log.error("image_decode_failed", error=str(e))
            raise ValueError(f"Failed to decode image: {e}")

        # Run detection using Claude or legacy pipeline
        if use_claude and self.claude_detector.is_available():
            log.info("running_claude_detection")
            # Convert to numpy array for Claude detector
            img_array = np.array(image)
            # Convert RGB to BGR for OpenCV compatibility
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            detected_state = self.claude_detector.detect(img_bgr, cid)
        else:
            log.info("running_legacy_detection")
            detection_result = self.detection_service.detect(image_b64, cid or "")
            detected_state = detection_result.boardState

        # Store sample
        sample = self.repository.create(
            image_bytes=image_bytes,
            image_width=image_width,
            image_height=image_height,
            detected_state=detected_state,
            source=source,
            notes=notes,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        log.info("sample_created", sample_id=str(sample.id), latency_ms=round(latency_ms, 2))

        return sample

    def get_sample(self, sample_id: UUID, cid: str | None = None) -> TrainingSampleResponse | None:
        """Get a training sample by ID."""
        return self.repository.get(sample_id)

    def list_samples(
        self,
        status: SampleStatus | None = None,
        page: int = 1,
        page_size: int = 20,
        cid: str | None = None,
    ) -> tuple[list[TrainingSampleResponse], int, int]:
        """List training samples with pagination."""
        samples, total = self.repository.list(status=status, page=page, page_size=page_size)
        total_pages = (total + page_size - 1) // page_size
        return samples, total, total_pages

    def update_sample(
        self,
        sample_id: UUID,
        update: TrainingSampleUpdate,
        cid: str | None = None,
    ) -> TrainingSampleResponse | None:
        """Update a training sample."""
        log = get_logger(cid)
        log.info("updating_sample", sample_id=str(sample_id))
        return self.repository.update(sample_id, update)

    def delete_sample(self, sample_id: UUID, cid: str | None = None) -> bool:
        """Delete a training sample."""
        log = get_logger(cid)
        log.info("deleting_sample", sample_id=str(sample_id))
        return self.repository.delete(sample_id)

    def get_stats(self, cid: str | None = None) -> AdminStats:
        """Get admin statistics."""
        return self.repository.get_stats()

    def repredict_sample(
        self,
        sample_id: UUID,
        cid: str | None = None,
        use_claude: bool = True,
    ) -> TrainingSampleResponse | None:
        """Re-run detection on a sample."""
        log = get_logger(cid)

        sample = self.repository.get(sample_id)
        if sample is None:
            return None

        # Load image
        image_path = self.repository.get_image_path(sample_id)
        if image_path is None or not image_path.exists():
            log.error("image_not_found", sample_id=str(sample_id))
            return None

        # Run detection using Claude or legacy pipeline
        if use_claude and self.claude_detector.is_available():
            log.info("repredicting_sample_with_claude", sample_id=str(sample_id))
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                log.error("failed_to_load_image", sample_id=str(sample_id))
                return None
            detected_state = self.claude_detector.detect(img_bgr, cid)
        else:
            log.info("repredicting_sample_legacy", sample_id=str(sample_id))
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode()
            detection_result = self.detection_service.detect(image_b64, cid or "")
            detected_state = detection_result.boardState

        # Update detected state
        return self.repository.update_detected_state(sample_id, detected_state)

    def get_sample_image_base64(self, sample_id: UUID) -> str | None:
        """Get the base64-encoded image for a sample."""
        image_path = self.repository.get_image_path(sample_id)
        if image_path is None or not image_path.exists():
            return None

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def train_model(
        self,
        validation_split: float = 0.2,
        min_samples: int = 10,
        cid: str | None = None,
    ) -> TrainResponse:
        """Train the KNN classifier using labeled samples."""
        log = get_logger(cid)
        start = time.perf_counter()

        # Get labeled samples
        labeled_samples = self.repository.get_labeled_samples()
        log.info("training_started", num_samples=len(labeled_samples))

        if len(labeled_samples) < min_samples:
            return TrainResponse(
                success=False,
                message=f"Not enough labeled samples. Need at least {min_samples}, have {len(labeled_samples)}",
                samples_used=len(labeled_samples),
            )

        try:
            # Collect training data: (cell_image, color_label) pairs
            training_data: list[tuple[np.ndarray, str]] = []
            validation_data: list[tuple[np.ndarray, str]] = []

            # Split samples into train/validation
            np.random.shuffle(labeled_samples)
            split_idx = int(len(labeled_samples) * (1 - validation_split))
            train_samples = labeled_samples[:split_idx]
            val_samples = labeled_samples[split_idx:]

            log.info("split_samples", train=len(train_samples), val=len(val_samples))

            # Extract cell images and labels from training samples
            for sample in train_samples:
                cells = self._extract_cell_data(sample, cid)
                training_data.extend(cells)

            for sample in val_samples:
                cells = self._extract_cell_data(sample, cid)
                validation_data.extend(cells)

            if not training_data:
                return TrainResponse(
                    success=False,
                    message="No valid training cells extracted",
                    samples_used=len(labeled_samples),
                )

            log.info(
                "training_data_extracted",
                train_cells=len(training_data),
                val_cells=len(validation_data),
            )

            # Build and train classifier
            train_accuracy, val_accuracy, color_accuracies = self._train_classifier(
                training_data,
                validation_data,
                cid,
            )

            latency_ms = (time.perf_counter() - start) * 1000
            log.info(
                "training_complete",
                train_accuracy=round(train_accuracy, 4),
                val_accuracy=round(val_accuracy, 4) if val_accuracy else None,
                latency_ms=round(latency_ms, 2),
            )

            return TrainResponse(
                success=True,
                message=f"Model trained successfully with {len(labeled_samples)} samples",
                samples_used=len(labeled_samples),
                train_accuracy=train_accuracy,
                validation_accuracy=val_accuracy,
                color_accuracies=color_accuracies,
                model_path=str(self.repository.models_path / "knn_classifier.pkl"),
            )

        except Exception as e:
            log.error("training_failed", error=str(e))
            return TrainResponse(
                success=False,
                message=f"Training failed: {str(e)}",
                samples_used=len(labeled_samples),
            )

    def _extract_cell_data(
        self,
        sample: TrainingSampleResponse,
        cid: str | None = None,
    ) -> list[tuple[np.ndarray, str]]:
        """Extract cell images with their labels from a sample."""
        log = get_logger(cid)

        # Load image
        image_path = self.repository.get_image_path(sample.id)
        if image_path is None or not image_path.exists():
            log.warning("sample_image_not_found", sample_id=str(sample.id))
            return []

        img = cv2.imread(str(image_path))
        if img is None:
            log.warning("failed_to_load_image", sample_id=str(sample.id))
            return []

        # Get board dimensions and calculate cell size
        height, width = img.shape[:2]
        cell_h = height // 20
        cell_w = width // 20

        cells = []
        labeled_state = sample.labeled_state
        if not labeled_state:
            return []

        for row_idx, row in enumerate(labeled_state):
            for col_idx, cell in enumerate(row):
                # Extract cell region
                y1 = row_idx * cell_h
                y2 = (row_idx + 1) * cell_h
                x1 = col_idx * cell_w
                x2 = (col_idx + 1) * cell_w

                cell_img = img[y1:y2, x1:x2]
                if cell_img.size == 0:
                    continue

                # Resize to standard size
                cell_img = cv2.resize(cell_img, (20, 20))

                # Get label
                label = cell.color.value if cell.color else "empty"
                cells.append((cell_img, label))

        return cells

    def _train_classifier(
        self,
        training_data: list[tuple[np.ndarray, str]],
        validation_data: list[tuple[np.ndarray, str]],
        cid: str | None = None,
    ) -> tuple[float, float | None, dict[str, float]]:
        """Train KNN classifier on cell data."""
        log = get_logger(cid)

        # Extract features using color histograms
        def extract_features(cell_img: np.ndarray) -> np.ndarray:
            hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
            features = np.concatenate([h_hist, s_hist, v_hist])
            return features / (features.sum() + 1e-8)

        # Build feature matrix
        X_train = np.array([extract_features(img) for img, _ in training_data])
        y_train = np.array([label for _, label in training_data])

        # Train KNN
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_train, y_train)

        # Calculate training accuracy
        train_predictions = knn.predict(X_train)
        train_accuracy = (train_predictions == y_train).mean()

        # Calculate validation accuracy
        val_accuracy = None
        color_accuracies = {}

        if validation_data:
            X_val = np.array([extract_features(img) for img, _ in validation_data])
            y_val = np.array([label for _, label in validation_data])

            val_predictions = knn.predict(X_val)
            val_accuracy = (val_predictions == y_val).mean()

            # Per-color accuracy
            for color in ["blue", "yellow", "red", "green", "empty"]:
                mask = y_val == color
                if mask.sum() > 0:
                    color_acc = (val_predictions[mask] == y_val[mask]).mean()
                    color_accuracies[color] = round(float(color_acc), 4)

        # Save model
        model_path = self.repository.models_path / "knn_classifier.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(knn, f)

        log.info("model_saved", path=str(model_path))

        return train_accuracy, val_accuracy, color_accuracies


# Singleton instance
_service: TrainingDataService | None = None


def get_service() -> TrainingDataService:
    """Get the singleton service instance."""
    global _service
    if _service is None:
        _service = TrainingDataService()
    return _service
