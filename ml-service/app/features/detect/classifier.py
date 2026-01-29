"""
Board visibility classifier inference module.

Uses EfficientNet-B0 with a trained classifier head to predict
whether the entire board is visible in a cropped board region.
"""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Models directory
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"

# Cache for loaded classifier models
_classifier_models: dict[str, nn.Module] = {}

# Inference transform (same as validation transform during training)
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_classifier_model() -> nn.Module:
    """
    Create EfficientNet-B0 model for binary classification.

    Uses pretrained ImageNet weights and replaces the classifier head
    with a single output for binary classification.
    """
    model = models.efficientnet_b0(weights=None)
    # Replace classifier head: original is (dropout, Linear(1280, 1000))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def get_classifier(model_id: str) -> nn.Module | None:
    """
    Load classifier model with caching.

    Args:
        model_id: Training run ID to load model for

    Returns:
        Loaded model or None if model doesn't exist
    """
    if model_id not in _classifier_models:
        model_path = MODELS_DIR / model_id / "classifier.pt"

        if not model_path.exists():
            print(f"[classifier] No classifier found at {model_path}")
            return None

        print(f"[classifier] Loading classifier model: {model_path}")

        # Create model and load state dict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = create_classifier_model()
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        _classifier_models[model_id] = model

    return _classifier_models[model_id]


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


def classify_board_visibility(
    image_path: Path,
    bbox: dict,
    model_id: str,
) -> tuple[bool, float]:
    """
    Classify if entire board is visible in the cropped region.

    Args:
        image_path: Path to the full image
        bbox: Dict with x, y, width, height (0-1 normalized)
        model_id: Training run ID of the model to use

    Returns:
        Tuple of (is_entire_visible, confidence)
        If no classifier is available, returns (False, 0.0)
    """
    model = get_classifier(model_id)

    if model is None:
        # No classifier available - fall back to heuristic
        return (False, 0.0)

    try:
        # Get device from model
        device = next(model.parameters()).device

        # Crop and transform image
        crop = crop_board_region(image_path, bbox)
        tensor = _transform(crop).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            logit = model(tensor)
            prob = torch.sigmoid(logit).item()

        is_visible = prob > 0.5
        confidence = prob if is_visible else (1.0 - prob)

        print(
            f"[classifier] EfficientNet-B0 prediction: is_entire_visible={is_visible}, "
            f"confidence={confidence:.4f}"
        )

        return (is_visible, confidence)

    except Exception as e:
        print(f"[classifier] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return (False, 0.0)
