"""YOLO-Pose training for corner detection on cropped board regions."""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import httpx
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO


# API base URL for fetching training data
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:3009")

# Models directory
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


class PoseTrainingSample(BaseModel):
    sample_id: str
    image_url: str
    bbox: dict  # { x, y, width, height } - all 0-1 normalized
    corners: dict  # { topLeft, topRight, bottomRight, bottomLeft } - each { x, y }


class PoseTrainMetrics(BaseModel):
    accuracy: float  # PCK - percentage of correct keypoints
    loss: float
    error: float | None = None  # Mean pixel error for keypoints (normalized 0-1)


def fetch_pose_training_data() -> list[PoseTrainingSample]:
    """Fetch samples with corner labels from the API."""
    url = f"{API_BASE_URL}/api/ml/pose-training-data"
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    return [PoseTrainingSample(**s) for s in data["samples"]]


def download_image(url: str, dest_path: Path) -> bool:
    """Download an image from URL to destination path."""
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return True
    except Exception as e:
        print(f"[pose] Failed to download {url}: {e}")
        return False


def crop_to_bbox(image_path: Path, bbox: dict, output_path: Path, padding: float = 0.05) -> tuple[int, int, int, int]:
    """
    Crop image to bounding box with optional padding.

    Args:
        image_path: Path to source image
        bbox: { x, y, width, height } normalized 0-1
        output_path: Path to save cropped image
        padding: Fraction of bbox size to add as padding (default 5%)

    Returns:
        (crop_x, crop_y, crop_w, crop_h) in pixels - the actual crop region
    """
    img = Image.open(image_path)
    img_w, img_h = img.size

    # Convert normalized bbox to pixels
    x = bbox["x"] * img_w
    y = bbox["y"] * img_h
    w = bbox["width"] * img_w
    h = bbox["height"] * img_h

    # Add padding
    pad_w = w * padding
    pad_h = h * padding

    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(img_w, int(x + w + pad_w))
    y2 = min(img_h, int(y + h + pad_h))

    # Crop and save
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(output_path, "JPEG", quality=95)

    return (x1, y1, x2 - x1, y2 - y1)


def corners_to_crop_coords(corners: dict, bbox: dict, crop_region: tuple[int, int, int, int], img_size: tuple[int, int]) -> dict:
    """
    Convert corners from full-image normalized coords to crop-relative normalized coords.

    Args:
        corners: { topLeft: {x, y}, ... } in full-image normalized coords
        bbox: Original bbox (not used directly, but kept for reference)
        crop_region: (crop_x, crop_y, crop_w, crop_h) in pixels
        img_size: (img_w, img_h) original image size in pixels

    Returns:
        corners in crop-relative normalized coords (0-1 within crop)
    """
    img_w, img_h = img_size
    crop_x, crop_y, crop_w, crop_h = crop_region

    def transform_point(pt: dict) -> dict:
        # Convert from normalized full-image to pixels
        px = pt["x"] * img_w
        py = pt["y"] * img_h

        # Convert to crop-relative pixels
        rel_x = px - crop_x
        rel_y = py - crop_y

        # Normalize to crop size (0-1)
        norm_x = max(0, min(1, rel_x / crop_w))
        norm_y = max(0, min(1, rel_y / crop_h))

        return {"x": norm_x, "y": norm_y}

    return {
        "topLeft": transform_point(corners["topLeft"]),
        "topRight": transform_point(corners["topRight"]),
        "bottomRight": transform_point(corners["bottomRight"]),
        "bottomLeft": transform_point(corners["bottomLeft"]),
    }


def corners_to_yolo_keypoints(corners: dict) -> list[tuple[float, float, int]]:
    """
    Convert corners to YOLO keypoint format.
    Input: { topLeft: {x, y}, topRight: {x, y}, bottomRight: {x, y}, bottomLeft: {x, y} }
    Output: [(x1, y1, v1), (x2, y2, v2), ...] where v=2 means visible

    Order: TL, TR, BR, BL (clockwise from top-left)
    """
    return [
        (corners["topLeft"]["x"], corners["topLeft"]["y"], 2),
        (corners["topRight"]["x"], corners["topRight"]["y"], 2),
        (corners["bottomRight"]["x"], corners["bottomRight"]["y"], 2),
        (corners["bottomLeft"]["x"], corners["bottomLeft"]["y"], 2),
    ]


def create_pose_dataset(
    samples: list[PoseTrainingSample], dataset_dir: Path
) -> tuple[int, int]:
    """
    Create YOLO-Pose dataset structure with cropped images and labels.

    Images are cropped to bounding box + padding, and corner coordinates
    are converted to be relative to the crop.

    YOLO-Pose label format per line:
    class_id cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...

    Returns: (train_count, val_count)
    """
    # Create directory structure
    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"
    temp_download = dataset_dir / "temp_download"

    for d in [images_train, images_val, labels_train, labels_val, temp_download]:
        d.mkdir(parents=True, exist_ok=True)

    # Split: 80% train, 20% val
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Ensure at least 1 sample in each set
    if len(val_samples) == 0 and len(train_samples) > 1:
        val_samples = [train_samples.pop()]
    elif len(train_samples) == 0 and len(val_samples) > 0:
        train_samples = [val_samples.pop()]

    train_count = 0
    val_count = 0

    def process_sample(sample: PoseTrainingSample, images_dir: Path, labels_dir: Path) -> bool:
        """Process a single sample: download, crop, convert coords, save label."""
        # Download original image to temp location
        temp_img = temp_download / f"{sample.sample_id}.jpg"
        if not download_image(sample.image_url, temp_img):
            return False

        try:
            # Get original image size
            with Image.open(temp_img) as img:
                img_size = img.size

            # Crop to bbox with padding
            crop_img_path = images_dir / f"{sample.sample_id}.jpg"
            crop_region = crop_to_bbox(temp_img, sample.bbox, crop_img_path)

            # Convert corners to crop-relative coords
            crop_corners = corners_to_crop_coords(sample.corners, sample.bbox, crop_region, img_size)

            # For cropped images, the "bbox" in the label is the entire crop (0.5, 0.5, 1.0, 1.0)
            # since the crop IS the board region
            cx, cy, w, h = 0.5, 0.5, 1.0, 1.0

            keypoints = corners_to_yolo_keypoints(crop_corners)

            # Format: class_id cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
            kp_str = " ".join(f"{kp[0]:.6f} {kp[1]:.6f} {kp[2]}" for kp in keypoints)
            label_path = labels_dir / f"{sample.sample_id}.txt"
            label_path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp_str}\n")

            return True
        except Exception as e:
            print(f"[pose] Failed to process {sample.sample_id}: {e}")
            return False
        finally:
            # Clean up temp file
            if temp_img.exists():
                temp_img.unlink()

    # Process training samples
    for sample in train_samples:
        if process_sample(sample, images_train, labels_train):
            train_count += 1

    # Process validation samples
    for sample in val_samples:
        if process_sample(sample, images_val, labels_val):
            val_count += 1

    # Clean up temp directory
    shutil.rmtree(temp_download, ignore_errors=True)

    # Create data.yaml for pose estimation
    # Note: kpt_shape is [num_keypoints, num_dims] where dims=3 for (x,y,visibility)
    yaml_content = f"""# YOLO-Pose dataset config (cropped to bbox)
path: {dataset_dir}
train: images/train
val: images/val

# Classes
names:
  0: board

# Keypoints configuration
# 4 corners: TL, TR, BR, BL
kpt_shape: [4, 3]

# Skeleton connections (for visualization)
# Connect corners in order: TL-TR-BR-BL-TL
flip_idx: [0, 3, 2, 1]
"""
    (dataset_dir / "data.yaml").write_text(yaml_content)

    print(f"[pose] Dataset created: {train_count} train, {val_count} val samples (cropped to bbox)")
    return train_count, val_count


def train_pose(
    run_id: str,
    samples: list[PoseTrainingSample],
    progress_callback: Callable[[int, int, float, float], None] | None = None,
    epochs: int = 100,
) -> PoseTrainMetrics:
    """
    Train YOLO-Pose model for corner detection on cropped board regions.

    Args:
        run_id: Unique identifier for this training run
        samples: List of training samples with corners
        progress_callback: Optional callback(epoch, total_epochs, loss, accuracy)
        epochs: Number of training epochs

    Returns:
        PoseTrainMetrics with final accuracy and loss
    """
    print(f"[pose] Starting pose training with {len(samples)} samples")

    if len(samples) < 2:
        raise ValueError(f"Need at least 2 samples for training, got {len(samples)}")

    # Create temp directory for dataset
    with tempfile.TemporaryDirectory(prefix="yolo_pose_") as temp_dir:
        dataset_dir = Path(temp_dir)

        # Create YOLO-Pose dataset (cropped to bbox)
        print("[pose] Creating YOLO-Pose dataset (cropped to bbox)...")
        train_count, val_count = create_pose_dataset(samples, dataset_dir)

        if train_count == 0:
            raise ValueError("No training images were downloaded successfully")

        # Load YOLO-Pose base model
        # Using yolo26x-pose (extra-large) - YOLO26 has RLE for better keypoint accuracy
        # See: https://docs.ultralytics.com/models/yolo26/
        print("[pose] Loading YOLO26x-Pose base model...")
        model = YOLO("yolo26x-pose.pt")

        # Track metrics for final return
        last_loss = 0.0
        last_accuracy = 0.0
        last_error = None  # Will be computed at the end

        # Add callback for progress reporting
        def on_fit_epoch_end(trainer):
            nonlocal last_loss, last_accuracy
            epoch = trainer.epoch + 1

            # Get loss from trainer
            loss = float(trainer.loss_items[0]) if trainer.loss_items is not None else 0.0
            last_loss = loss

            # Get pose metrics - OKS (Object Keypoint Similarity) based mAP
            accuracy = 0.0
            if hasattr(trainer, 'validator') and trainer.validator and hasattr(trainer.validator, 'metrics'):
                val_metrics = trainer.validator.metrics
                # Pose models report mAP for keypoints
                if hasattr(val_metrics, 'pose'):
                    accuracy = float(val_metrics.pose.map50)
                elif hasattr(val_metrics, 'box'):
                    # Fallback to box mAP if pose metrics not available
                    accuracy = float(val_metrics.box.map50)

            last_accuracy = accuracy
            print(f"[pose] Epoch {epoch}/{epochs} - loss: {loss:.4f}, mAP50: {accuracy:.4f}")

            if progress_callback:
                progress_callback(epoch, epochs, loss, accuracy)

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        # Train the model
        print("[pose] Starting training...")
        start_time = time.time()

        # Model output directory
        model_dir = MODELS_DIR / run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        results = model.train(
            data=str(dataset_dir / "data.yaml"),
            epochs=epochs,
            imgsz=640,
            batch=8,
            patience=10,  # Early stopping
            save=True,
            project=str(MODELS_DIR),
            name=f"{run_id}_pose",
            exist_ok=True,
            verbose=True,
        )

        training_time = time.time() - start_time
        print(f"[pose] Training completed in {training_time:.1f}s")

        # Get final metrics
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        print(f"[pose] Results dict keys: {list(metrics.keys())}")

        # Extract pose-specific metrics
        accuracy = (
            metrics.get("metrics/mAP50(P)", 0.0) or  # Pose mAP
            metrics.get("metrics/mAP50(B)", 0.0) or  # Box mAP fallback
            last_accuracy or
            0.0
        )

        # Copy model weights
        pose_dir = MODELS_DIR / f"{run_id}_pose"
        weights_dir = pose_dir / "weights"
        best_pt = weights_dir / "best.pt"

        if best_pt.exists():
            # Copy to main model directory as pose.pt
            shutil.copy(best_pt, model_dir / "pose.pt")
            print(f"[pose] Model saved to {model_dir / 'pose.pt'}")
        else:
            last_pt = weights_dir / "last.pt"
            if last_pt.exists():
                shutil.copy(last_pt, model_dir / "pose.pt")
                print(f"[pose] Model saved from last.pt to {model_dir / 'pose.pt'}")

        # Clean up the pose training directory (keep only the weights we need)
        if pose_dir.exists() and pose_dir != model_dir:
            shutil.rmtree(pose_dir)

        # Extract pose loss as error metric (normalized keypoint distance)
        pose_loss = metrics.get("val/pose_loss", None) or metrics.get("pose_loss", None)
        if pose_loss is not None:
            last_error = float(pose_loss)

        print(f"[pose] Final metrics: accuracy={accuracy:.4f}, loss={last_loss:.4f}, error={last_error}")

        return PoseTrainMetrics(
            accuracy=float(accuracy),
            loss=float(last_loss),
            error=last_error,
        )
