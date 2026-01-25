"""
Board visibility classifier training module.

Trains an EfficientNet-B0 based binary classifier to predict
whether the entire board is visible in a cropped board region.
"""

import os
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple

import httpx
import torch
import torch.nn as nn
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# API base URL for fetching training data
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:3009")

# Models directory
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


class ClassifierSample(BaseModel):
    """Classifier training sample from API."""

    sample_id: str
    image_url: str
    bbox: dict  # { x, y, width, height } - all 0-1 normalized
    isEntireBoardVisible: bool


class ClassifierMetrics(NamedTuple):
    """Training metrics for the classifier."""

    accuracy: float
    loss: float


def fetch_classifier_training_data() -> list[ClassifierSample]:
    """Fetch classifier training samples from the API."""
    url = f"{API_BASE_URL}/api/ml/classifier-training-data"
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return [ClassifierSample(**s) for s in response.json()["samples"]]


def download_image(url: str, dest_path: Path) -> bool:
    """Download an image from URL to destination path."""
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return True
    except Exception as e:
        print(f"[classifier] Failed to download {url}: {e}")
        return False


def crop_board_region(image_path: Path, bbox: dict) -> Image.Image:
    """
    Crop board region from image using normalized bbox coordinates.

    Args:
        image_path: Path to the full image
        bbox: Dict with x, y, width, height (0-1 normalized)

    Returns:
        Cropped PIL Image
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x1 = int(bbox["x"] * w)
    y1 = int(bbox["y"] * h)
    x2 = int((bbox["x"] + bbox["width"]) * w)
    y2 = int((bbox["y"] + bbox["height"]) * h)

    # Ensure valid crop bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return img.crop((x1, y1, x2, y2))


class BoardVisibilityDataset(Dataset):
    """Dataset for board visibility classification."""

    def __init__(
        self,
        samples: list[tuple[Path, dict, bool]],
        transform: transforms.Compose | None = None,
    ):
        """
        Args:
            samples: List of (image_path, bbox, is_entire_visible) tuples
            transform: Optional torchvision transforms
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, bbox, label = self.samples[idx]
        img = crop_board_region(img_path, bbox)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(1.0 if label else 0.0)


def create_classifier_model() -> nn.Module:
    """
    Create EfficientNet-B0 model for binary classification.

    Uses pretrained ImageNet weights and replaces the classifier head
    with a single output for binary classification.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace classifier head: original is (dropout, Linear(1280, 1000))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def train_classifier(
    run_id: str,
    samples: list[ClassifierSample],
    progress_callback: Callable[[int, int, float, float], None] | None = None,
) -> ClassifierMetrics:
    """
    Train the board visibility classifier.

    Args:
        run_id: Training run ID for saving the model
        samples: List of classifier training samples
        progress_callback: Optional callback(epoch, total_epochs, loss, accuracy)

    Returns:
        ClassifierMetrics with final accuracy and loss
    """
    print(f"[classifier] Starting training with {len(samples)} samples")

    # Create temp directory for downloaded images
    with tempfile.TemporaryDirectory(prefix="classifier_train_") as temp_dir:
        temp_path = Path(temp_dir)

        # Download images and prepare dataset
        dataset_samples: list[tuple[Path, dict, bool]] = []

        for sample in samples:
            img_path = temp_path / f"{sample.sample_id}.jpg"
            if download_image(sample.image_url, img_path):
                dataset_samples.append((img_path, sample.bbox, sample.isEntireBoardVisible))

        if len(dataset_samples) < 2:
            raise ValueError(
                f"Need at least 2 samples for classifier training, got {len(dataset_samples)}"
            )

        print(f"[classifier] Downloaded {len(dataset_samples)} images")

        # Split into train/val (80/20)
        split_idx = int(len(dataset_samples) * 0.8)
        train_samples = dataset_samples[:split_idx]
        val_samples = dataset_samples[split_idx:]

        # Ensure at least 1 in each set
        if len(val_samples) == 0 and len(train_samples) > 1:
            val_samples = [train_samples.pop()]
        elif len(train_samples) == 0 and len(val_samples) > 0:
            train_samples = [val_samples.pop()]

        print(f"[classifier] Split: {len(train_samples)} train, {len(val_samples)} val")

        # Count class distribution
        train_pos = sum(1 for _, _, label in train_samples if label)
        train_neg = len(train_samples) - train_pos
        val_pos = sum(1 for _, _, label in val_samples if label)
        val_neg = len(val_samples) - val_pos
        print(
            f"[classifier] Train: {train_pos} visible, {train_neg} partial | "
            f"Val: {val_pos} visible, {val_neg} partial"
        )

        # Create transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Create datasets and dataloaders
        train_dataset = BoardVisibilityDataset(train_samples, train_transform)
        val_dataset = BoardVisibilityDataset(val_samples, val_transform)

        batch_size = min(8, len(train_samples))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[classifier] Using device: {device}")

        model = create_classifier_model()
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        # Training loop
        total_epochs = 20
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(total_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device).unsqueeze(1)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            scheduler.step(val_loss)

            print(
                f"[classifier] Epoch {epoch + 1}/{total_epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            )

            # Report progress
            if progress_callback:
                progress_callback(epoch + 1, total_epochs, val_loss, val_acc)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

        # Save the best model
        if best_model_state is not None:
            model_dir = MODELS_DIR / run_id
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "classifier.pt"

            # Save model state dict
            torch.save(best_model_state, model_path)
            print(f"[classifier] Model saved to {model_path}")

        # Calculate final metrics on validation set using best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        final_loss = 0.0
        final_correct = 0
        final_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                final_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                final_correct += (preds == labels).sum().item()
                final_total += labels.size(0)

        final_loss /= final_total
        final_acc = final_correct / final_total

        print(f"[classifier] Final metrics - accuracy: {final_acc:.4f}, loss: {final_loss:.4f}")

        return ClassifierMetrics(accuracy=final_acc, loss=final_loss)
