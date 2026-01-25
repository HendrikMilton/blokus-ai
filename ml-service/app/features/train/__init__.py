import os
import shutil
import tempfile
import time
from pathlib import Path

import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from ultralytics import YOLO

router = APIRouter()

# API base URL for fetching training data
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:3009")

# Models directory
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


class TrainRequest(BaseModel):
    run_id: str


class TrainResponse(BaseModel):
    accuracy: float
    loss: float
    val_accuracy: float
    val_loss: float


def report_progress(run_id: str, epoch: int, total_epochs: int, loss: float, accuracy: float):
    """Report training progress back to the API."""
    try:
        url = f"{API_BASE_URL}/api/ml/training-runs/{run_id}/progress"
        httpx.post(
            url,
            json={
                "epoch": epoch,
                "totalEpochs": total_epochs,
                "loss": loss,
                "accuracy": accuracy,
            },
            timeout=5.0,
        )
    except Exception as e:
        print(f"[train] Failed to report progress: {e}")


class TrainingSample(BaseModel):
    sample_id: str
    image_url: str
    bbox: dict  # { x, y, width, height } - all 0-1 normalized


def fetch_training_data() -> list[TrainingSample]:
    """Fetch verified training samples from the API."""
    url = f"{API_BASE_URL}/api/ml/training-data"
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    return [TrainingSample(**s) for s in data["samples"]]


def download_image(url: str, dest_path: Path) -> bool:
    """Download an image from URL to destination path."""
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return True
    except Exception as e:
        print(f"[train] Failed to download {url}: {e}")
        return False


def bbox_to_yolo(bbox: dict) -> tuple[float, float, float, float]:
    """
    Convert bbox from corner format to YOLO center format.
    Input: { x, y, width, height } where x,y is top-left corner (0-1 normalized)
    Output: (center_x, center_y, width, height) all 0-1 normalized
    """
    x = bbox["x"]
    y = bbox["y"]
    w = bbox["width"]
    h = bbox["height"]

    center_x = x + w / 2
    center_y = y + h / 2

    return (center_x, center_y, w, h)


def create_yolo_dataset(
    samples: list[TrainingSample], dataset_dir: Path
) -> tuple[int, int]:
    """
    Create YOLO dataset structure with images and labels.

    Returns: (train_count, val_count)
    """
    # Create directory structure
    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
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

    # Process training samples
    for sample in train_samples:
        img_path = images_train / f"{sample.sample_id}.jpg"
        label_path = labels_train / f"{sample.sample_id}.txt"

        if download_image(sample.image_url, img_path):
            # Write YOLO label: class_id center_x center_y width height
            cx, cy, w, h = bbox_to_yolo(sample.bbox)
            label_path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            train_count += 1

    # Process validation samples
    for sample in val_samples:
        img_path = images_val / f"{sample.sample_id}.jpg"
        label_path = labels_val / f"{sample.sample_id}.txt"

        if download_image(sample.image_url, img_path):
            cx, cy, w, h = bbox_to_yolo(sample.bbox)
            label_path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            val_count += 1

    # Create data.yaml
    yaml_content = f"""# YOLO dataset config
path: {dataset_dir}
train: images/train
val: images/val

# Classes
names:
  0: board
"""
    (dataset_dir / "data.yaml").write_text(yaml_content)

    print(f"[train] Dataset created: {train_count} train, {val_count} val samples")
    return train_count, val_count


@router.post("/", response_model=TrainResponse)
async def start_training(request: TrainRequest):
    """
    Start model training with verified labeled images.
    Fetches training data from API, creates YOLO dataset, trains model.
    """
    print(f"[train] Starting training run: {request.run_id}")

    # Fetch training data from API
    print("[train] Fetching training data from API...")
    samples = fetch_training_data()
    print(f"[train] Got {len(samples)} verified samples")

    if len(samples) < 2:
        raise ValueError(f"Need at least 2 samples for training, got {len(samples)}")

    # Create temp directory for dataset
    with tempfile.TemporaryDirectory(prefix="yolo_train_") as temp_dir:
        dataset_dir = Path(temp_dir)

        # Create YOLO dataset
        print("[train] Creating YOLO dataset...")
        train_count, val_count = create_yolo_dataset(samples, dataset_dir)

        if train_count == 0:
            raise ValueError("No training images were downloaded successfully")

        # Load base model and train
        print("[train] Loading YOLOv8 base model...")
        model = YOLO("yolov8n.pt")  # Start with nano model for fine-tuning

        # Training parameters
        total_epochs = 50

        # Add callback for progress reporting
        def on_train_epoch_end(trainer):
            """Called at the end of each training epoch."""
            epoch = trainer.epoch + 1  # 0-indexed to 1-indexed
            metrics = trainer.metrics or {}
            loss = metrics.get("train/box_loss", 0.0)
            accuracy = metrics.get("metrics/mAP50", 0.0)
            print(f"[train] Epoch {epoch}/{total_epochs} - loss: {loss:.4f}, mAP50: {accuracy:.4f}")
            report_progress(request.run_id, epoch, total_epochs, float(loss), float(accuracy))

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # Train the model
        print("[train] Starting training...")
        start_time = time.time()

        results = model.train(
            data=str(dataset_dir / "data.yaml"),
            epochs=total_epochs,
            imgsz=640,
            batch=8,
            patience=10,  # Early stopping
            save=True,
            project=str(MODELS_DIR),
            name=request.run_id,
            exist_ok=True,
            verbose=True,
        )

        training_time = time.time() - start_time
        print(f"[train] Training completed in {training_time:.1f}s")

        # Get metrics from training results
        # results.results_dict contains final metrics
        metrics = results.results_dict if hasattr(results, "results_dict") else {}

        # Extract metrics - YOLO stores these with specific keys
        # For detection: metrics/precision, metrics/recall, metrics/mAP50, metrics/mAP50-95
        # For loss: train/box_loss, train/cls_loss, train/dfl_loss, val/box_loss, etc.
        accuracy = metrics.get("metrics/mAP50", 0.0)
        val_accuracy = metrics.get("metrics/mAP50-95", 0.0)

        # Loss values (use box loss as primary)
        loss = metrics.get("train/box_loss", 0.0)
        val_loss = metrics.get("val/box_loss", 0.0)

        # Ensure model directory structure
        model_dir = MODELS_DIR / request.run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # YOLO saves best.pt and last.pt in the run directory
        # Move best.pt to our expected location if needed
        weights_dir = model_dir / "weights"
        best_pt = weights_dir / "best.pt"

        if best_pt.exists():
            # Copy to model_dir root for easy access
            shutil.copy(best_pt, model_dir / "best.pt")
            print(f"[train] Model saved to {model_dir / 'best.pt'}")
        else:
            # Fallback to last.pt
            last_pt = weights_dir / "last.pt"
            if last_pt.exists():
                shutil.copy(last_pt, model_dir / "best.pt")
                print(f"[train] Model saved from last.pt to {model_dir / 'best.pt'}")

        print(
            f"[train] Metrics: accuracy={accuracy:.4f}, loss={loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}"
        )

        return TrainResponse(
            accuracy=float(accuracy),
            loss=float(loss),
            val_accuracy=float(val_accuracy),
            val_loss=float(val_loss),
        )
