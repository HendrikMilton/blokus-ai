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
    sample_ids: list[str] | None = None


class TrainResponse(BaseModel):
    accuracy: float
    loss: float
    val_accuracy: float
    val_loss: float
    classifier_accuracy: float | None = None
    classifier_loss: float | None = None
    pose_accuracy: float | None = None
    pose_loss: float | None = None
    pose_image_count: int | None = None


def report_progress(
    run_id: str,
    phase: str,
    epoch: int,
    total_epochs: int,
    loss: float,
    accuracy: float,
):
    """Report training progress back to the API."""
    try:
        url = f"{API_BASE_URL}/api/ml/training-runs/{run_id}/progress"
        print(f"[DEBUG] Sending progress to {url}: phase={phase}, epoch={epoch}, loss={loss:.4f}, accuracy={accuracy:.4f}")
        response = httpx.post(
            url,
            json={
                "phase": phase,
                "epoch": epoch,
                "totalEpochs": total_epochs,
                "loss": loss,
                "accuracy": accuracy,
            },
            timeout=5.0,
        )
        print(f"[DEBUG] Progress API response: {response.status_code}")
    except Exception as e:
        print(f"[train] Failed to report progress: {e}")


def report_completion(
    run_id: str,
    duration_ms: int,
    accuracy: float,
    loss: float,
    val_accuracy: float,
    val_loss: float,
    classifier_accuracy: float | None = None,
    classifier_loss: float | None = None,
    pose_accuracy: float | None = None,
    pose_loss: float | None = None,
    pose_error: float | None = None,
    pose_image_count: int | None = None,
):
    """Report training completion back to the API."""
    try:
        url = f"{API_BASE_URL}/api/ml/training-runs/{run_id}/complete"
        print(f"[train] Reporting completion to {url}")
        response = httpx.post(
            url,
            json={
                "accuracy": accuracy,
                "loss": loss,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
                "classifier_accuracy": classifier_accuracy,
                "classifier_loss": classifier_loss,
                "pose_accuracy": pose_accuracy,
                "pose_loss": pose_loss,
                "pose_error": pose_error,
                "pose_image_count": pose_image_count,
                "duration_ms": duration_ms,
            },
            timeout=10.0,
        )
        print(f"[train] Completion API response: {response.status_code}")
    except Exception as e:
        print(f"[train] Failed to report completion: {e}")


def report_failure(run_id: str, duration_ms: int, error_message: str):
    """Report training failure back to the API."""
    try:
        url = f"{API_BASE_URL}/api/ml/training-runs/{run_id}/failed"
        print(f"[train] Reporting failure to {url}: {error_message}")
        response = httpx.post(
            url,
            json={
                "error_message": error_message,
                "duration_ms": duration_ms,
            },
            timeout=10.0,
        )
        print(f"[train] Failure API response: {response.status_code}")
    except Exception as e:
        print(f"[train] Failed to report failure: {e}")


class TrainingSample(BaseModel):
    sample_id: str
    image_url: str
    bbox: dict  # { x, y, width, height } - all 0-1 normalized


def fetch_training_data(sample_ids: list[str] | None = None) -> list[TrainingSample]:
    """Fetch training samples from the API. If sample_ids provided, only fetch those."""
    url = f"{API_BASE_URL}/api/ml/training-data"
    if sample_ids:
        url += f"?sampleIds={','.join(sample_ids)}"
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
    Reports completion/failure back to API via callback endpoints.
    """
    print(f"[train] Starting training run: {request.run_id}")
    start_time = time.time()

    try:
        return await _do_training(request, start_time)
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_message = str(e)
        print(f"[train] Training failed: {error_message}")
        import traceback
        traceback.print_exc()
        report_failure(request.run_id, duration_ms, error_message)
        raise


async def _do_training(request: TrainRequest, start_time: float) -> TrainResponse:
    """Internal training logic."""
    # Fetch training data from API
    print("[train] Fetching training data from API...")
    samples = fetch_training_data(request.sample_ids)
    if request.sample_ids:
        print(f"[train] Got {len(samples)} samples (filtered from {len(request.sample_ids)} selected)")
    else:
        print(f"[train] Got {len(samples)} samples (all human-labeled)")

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
        # Using YOLO26x (extra-large) - the latest and most accurate model
        # YOLO26x has 55.7M params and 57.5 mAP on COCO (vs yolov8n's 3.2M/37.3 mAP)
        # YOLO26 (Jan 2026) removes NMS/DFL bottlenecks for better convergence
        print("[train] Loading YOLO26x base model...")
        model = YOLO("yolo26x.pt")

        # Training parameters - increased epochs for better convergence
        # CIoU is already the default loss function (best for bbox regression)
        total_epochs = 100

        # Track last known loss for final metrics (not available in results_dict)
        last_detection_loss = 0.0

        # Add callback for progress reporting after validation (metrics available)
        def on_fit_epoch_end(trainer):
            """Called at the end of each epoch after validation."""
            nonlocal last_detection_loss
            epoch = trainer.epoch + 1  # 0-indexed to 1-indexed
            # Get loss from trainer's loss_items (box, cls, dfl losses)
            loss = float(trainer.loss_items[0]) if trainer.loss_items is not None else 0.0
            last_detection_loss = loss  # Store for final metrics
            # Get mAP50 from validator metrics
            accuracy = 0.0
            if hasattr(trainer, 'validator') and trainer.validator and hasattr(trainer.validator, 'metrics'):
                val_metrics = trainer.validator.metrics
                if hasattr(val_metrics, 'box'):
                    accuracy = float(val_metrics.box.map50)
            print(f"[train] Detection Epoch {epoch}/{total_epochs} - loss: {loss:.4f}, mAP50: {accuracy:.4f}")
            report_progress(request.run_id, "detection", epoch, total_epochs, loss, accuracy)

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

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
        print(f"[DEBUG] results.results_dict keys: {list(metrics.keys())}")
        print(f"[DEBUG] results.results_dict: {metrics}")

        # Extract metrics - YOLO stores these with (B) suffix for box metrics
        accuracy = (
            metrics.get("metrics/mAP50(B)", 0.0) or
            metrics.get("metrics/mAP50", 0.0) or
            0.0
        )
        val_accuracy = (
            metrics.get("metrics/mAP50-95(B)", 0.0) or
            metrics.get("metrics/mAP50-95", 0.0) or
            0.0
        )

        # Loss values - use last known from training callback
        loss = last_detection_loss
        val_loss = 0.0  # Not tracked separately

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
            f"[train] YOLO Metrics: accuracy={accuracy:.4f}, loss={loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}"
        )

        # Train classifier after YOLO
        classifier_accuracy = None
        classifier_loss = None

        try:
            from .classifier import fetch_classifier_training_data, train_classifier

            print("[train] Fetching classifier training data...")
            classifier_samples = fetch_classifier_training_data()
            print(f"[train] Got {len(classifier_samples)} classifier samples")

            if len(classifier_samples) >= 2:
                print("[train] Starting classifier training...")
                classifier_epochs = 20  # Match the default in classifier.py
                classifier_metrics = train_classifier(
                    run_id=request.run_id,
                    samples=classifier_samples,
                    progress_callback=lambda epoch, total, loss, acc: report_progress(
                        request.run_id, "classifier", epoch, total, loss, acc
                    ),
                )
                classifier_accuracy = classifier_metrics.accuracy
                classifier_loss = classifier_metrics.loss
                print(
                    f"[train] Classifier Metrics: accuracy={classifier_accuracy:.4f}, "
                    f"loss={classifier_loss:.4f}"
                )
            else:
                print(f"[train] Skipping classifier: need >= 2 samples, got {len(classifier_samples)}")
        except Exception as e:
            print(f"[train] Classifier training failed: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the whole training run if classifier fails

        # Train pose model for corner detection
        pose_accuracy = None
        pose_loss = None
        pose_error = None
        pose_image_count = None

        try:
            from .pose import fetch_pose_training_data, train_pose

            print("[train] Fetching pose training data...")
            pose_samples = fetch_pose_training_data()
            pose_image_count = len(pose_samples)
            print(f"[train] Got {pose_image_count} pose samples with corner labels")

            if pose_image_count >= 2:
                print("[train] Starting pose training...")
                pose_metrics = train_pose(
                    run_id=request.run_id,
                    samples=pose_samples,
                    progress_callback=lambda epoch, total, loss, acc: report_progress(
                        request.run_id, "pose", epoch, total, loss, acc
                    ),
                    epochs=100,
                )
                pose_accuracy = pose_metrics.accuracy
                pose_loss = pose_metrics.loss
                pose_error = pose_metrics.error
                print(
                    f"[train] Pose Metrics: accuracy={pose_accuracy:.4f}, "
                    f"loss={pose_loss:.4f}, error={pose_error}"
                )
            else:
                print(f"[train] Skipping pose: need >= 2 samples, got {pose_image_count}")
        except Exception as e:
            print(f"[train] Pose training failed: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the whole training run if pose fails

        # Report completion to API
        duration_ms = int((time.time() - start_time) * 1000)
        report_completion(
            run_id=request.run_id,
            duration_ms=duration_ms,
            accuracy=float(accuracy),
            loss=float(loss),
            val_accuracy=float(val_accuracy),
            val_loss=float(val_loss),
            classifier_accuracy=classifier_accuracy,
            classifier_loss=classifier_loss,
            pose_accuracy=pose_accuracy,
            pose_loss=pose_loss,
            pose_error=pose_error,
            pose_image_count=pose_image_count,
        )

        return TrainResponse(
            accuracy=float(accuracy),
            loss=float(loss),
            val_accuracy=float(val_accuracy),
            val_loss=float(val_loss),
            classifier_accuracy=classifier_accuracy,
            classifier_loss=classifier_loss,
            pose_accuracy=pose_accuracy,
            pose_loss=pose_loss,
            pose_image_count=pose_image_count,
        )
