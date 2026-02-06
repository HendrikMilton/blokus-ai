from pathlib import Path

from ultralytics import YOLO

from .schemas import BoundingBox, BoardDetection, BoardCorners, Corner

# Load YOLO26x (largest model) - downloads automatically on first use
_model: YOLO | None = None

# Cache for trained models
_trained_models: dict[str, YOLO] = {}

# Cache for pose models
_pose_models: dict[str, YOLO] = {}

# Models directory (same as train module)
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


def get_model() -> YOLO:
    """Lazy load the default YOLO model."""
    global _model
    if _model is None:
        _model = YOLO("yolo26x.pt")
    return _model


def get_trained_model(model_id: str) -> YOLO:
    """Load a trained model by ID, with caching."""
    if model_id not in _trained_models:
        model_path = MODELS_DIR / model_id / "best.pt"
        if not model_path.exists():
            # Try weights subdirectory
            model_path = MODELS_DIR / model_id / "weights" / "best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        print(f"[detect] Loading trained model: {model_path}")
        _trained_models[model_id] = YOLO(str(model_path))
    return _trained_models[model_id]


def get_pose_model(model_id: str) -> YOLO | None:
    """Load a trained pose model by ID, with caching. Returns None if not found."""
    if model_id not in _pose_models:
        model_path = MODELS_DIR / model_id / "pose.pt"
        if not model_path.exists():
            print(f"[detect] Pose model not found: {model_path}")
            return None
        print(f"[detect] Loading pose model: {model_path}")
        _pose_models[model_id] = YOLO(str(model_path))
    return _pose_models[model_id]


def detect_corners_with_pose_model(
    image_path: Path,
    model_id: str,
    bbox: dict,
    padding: float = 0.05
) -> tuple[BoardCorners | None, float]:
    """
    Detect board corners using a trained YOLO-Pose model on a cropped region.

    The image is first cropped to the bounding box (with padding), then
    pose detection runs on the crop, and coordinates are converted back
    to full-image normalized coordinates.

    Args:
        image_path: Path to the full image
        model_id: Training run ID for the pose model
        bbox: { x, y, width, height } normalized 0-1 (detected board bbox)
        padding: Fraction of bbox size to add as padding (default 5%)

    Returns: (corners, confidence) or (None, 0.0) if not detected
    """
    model = get_pose_model(model_id)
    if model is None:
        return None, 0.0

    print(f"[detect_corners] Running pose detection on cropped region of {image_path}")

    # Load image and get dimensions
    from PIL import Image
    img = Image.open(image_path)
    img_w, img_h = img.size

    # Calculate crop region with padding
    x = bbox["x"] * img_w
    y = bbox["y"] * img_h
    w = bbox["width"] * img_w
    h = bbox["height"] * img_h

    pad_w = w * padding
    pad_h = h * padding

    crop_x1 = max(0, int(x - pad_w))
    crop_y1 = max(0, int(y - pad_h))
    crop_x2 = min(img_w, int(x + w + pad_w))
    crop_y2 = min(img_h, int(y + h + pad_h))

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    # Crop the image
    cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Save to temp file for YOLO (or use numpy array)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cropped.save(tmp.name, "JPEG", quality=95)
        tmp_path = tmp.name

    try:
        # Run pose detection on cropped image
        results = model(tmp_path, verbose=False)

        if not results or len(results) == 0:
            print("[detect_corners] No results from pose model")
            return None, 0.0

        result = results[0]

        # Check for keypoints
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            print("[detect_corners] No keypoints in result")
            return None, 0.0

        keypoints = result.keypoints
        if keypoints.xy is None or len(keypoints.xy) == 0:
            print("[detect_corners] No keypoint coordinates")
            return None, 0.0

        # Get the first detection's keypoints
        # Shape: [num_objects, num_keypoints, 2] for xy
        kpts = keypoints.xy[0].cpu().numpy()  # First detection
        conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else None

        if len(kpts) < 4:
            print(f"[detect_corners] Expected 4 keypoints, got {len(kpts)}")
            return None, 0.0

        # Debug: print raw keypoint values and crop dimensions
        print(f"[detect_corners] Crop region: x1={crop_x1}, y1={crop_y1}, w={crop_w}, h={crop_h}")
        print(f"[detect_corners] Raw keypoints (pixels in crop):")
        for i, kp in enumerate(kpts[:4]):
            print(f"  kp[{i}]: ({kp[0]:.1f}, {kp[1]:.1f}) - in crop bounds: x={0 <= kp[0] <= crop_w}, y={0 <= kp[1] <= crop_h}")

        # Keypoints are in crop coordinates (pixels)
        # Convert to full-image normalized coordinates

        def crop_to_full_image(kp_x: float, kp_y: float) -> tuple[float, float]:
            """Convert crop pixel coords to full-image normalized coords."""
            # Clamp keypoint to crop bounds (model may predict outside)
            kp_x = max(0, min(crop_w, kp_x))
            kp_y = max(0, min(crop_h, kp_y))
            # Add crop offset to get full-image pixel coords
            full_x = crop_x1 + kp_x
            full_y = crop_y1 + kp_y
            # Normalize to 0-1
            return (full_x / img_w, full_y / img_h)

        tl = crop_to_full_image(kpts[0][0], kpts[0][1])
        tr = crop_to_full_image(kpts[1][0], kpts[1][1])
        br = crop_to_full_image(kpts[2][0], kpts[2][1])
        bl = crop_to_full_image(kpts[3][0], kpts[3][1])

        corners = BoardCorners(
            topLeft=Corner(x=float(tl[0]), y=float(tl[1])),
            topRight=Corner(x=float(tr[0]), y=float(tr[1])),
            bottomRight=Corner(x=float(br[0]), y=float(br[1])),
            bottomLeft=Corner(x=float(bl[0]), y=float(bl[1])),
        )

        # Average confidence across keypoints
        avg_conf = float(conf.mean()) if conf is not None else 0.5

        print(f"[detect_corners] Detected corners with conf={avg_conf:.3f} (on cropped region)")
        print(f"  TL: ({corners.topLeft.x:.3f}, {corners.topLeft.y:.3f})")
        print(f"  TR: ({corners.topRight.x:.3f}, {corners.topRight.y:.3f})")
        print(f"  BR: ({corners.bottomRight.x:.3f}, {corners.bottomRight.y:.3f})")
        print(f"  BL: ({corners.bottomLeft.x:.3f}, {corners.bottomLeft.y:.3f})")

        return corners, avg_conf

    finally:
        # Clean up temp file
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def detect_board_yolo(image_path: Path, device: str | None = None) -> BoardDetection:
    """
    Detect a Blokus board in an image using YOLO26x.

    Uses the largest YOLO26 model for best accuracy.
    Looks for objects that could be a board game (table, board, etc.)
    and returns the best candidate based on size and confidence.
    """
    print(f"[detect_board] START image_path={image_path}")

    model = get_model()
    print("[detect_board] model loaded: yolo26x")

    predict_kwargs: dict = {"verbose": False}
    if device:
        predict_kwargs["device"] = device

    results = model(str(image_path), **predict_kwargs)

    if not results or len(results) == 0:
        print("[detect_board] no results from model")
        return BoardDetection(
            hasBoardDetected=False,
            isEntireBoardVisible=False,
            boundingBox=None,
            confidence=0.0,
        )

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("[detect_board] no boxes detected")
        return BoardDetection(
            hasBoardDetected=False,
            isEntireBoardVisible=False,
            boundingBox=None,
            confidence=0.0,
        )

    img_h, img_w = result.orig_shape
    print(f"[detect_board] image: {img_w}x{img_h}, boxes: {len(boxes)}")

    # Find the best candidate - prefer large rectangular objects
    # COCO classes that might contain a board: dining table (60), book (73), etc.
    # But we'll consider any large detection as potential board
    best_box = None
    best_score = 0.0

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        x1, y1, x2, y2 = xyxy
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

        # Score: area * aspect_ratio * confidence (prefer large, square, confident)
        area_ratio = area / (img_w * img_h)

        print(
            f"[detect_board] box[{i}]: class={cls_id}, conf={conf:.3f}, "
            f"area_ratio={area_ratio:.3f}, aspect={aspect_ratio:.3f}, "
            f"bbox=[{x1:.1f}, {y1:.1f}, {bw:.1f}, {bh:.1f}]"
        )

        # Only consider detections covering at least 5% of image
        if area_ratio > 0.05 and aspect_ratio > 0.5:
            score = area_ratio * aspect_ratio * conf
            if score > best_score:
                best_score = score
                best_box = (x1, y1, bw, bh, conf)
                print(f"[detect_board] new best candidate: box[{i}] score={score:.4f}")

    if best_box:
        x, y, bw, bh, conf = best_box

        # Check if box touches image edges (partial board)
        margin = 5
        is_entire = (
            x > margin and
            y > margin and
            x + bw < img_w - margin and
            y + bh < img_h - margin
        )

        print(
            f"[detect_board] SUCCESS conf={conf:.3f}, is_entire={is_entire}, "
            f"bbox=[{x/img_w:.3f}, {y/img_h:.3f}, {bw/img_w:.3f}, {bh/img_h:.3f}]"
        )

        return BoardDetection(
            hasBoardDetected=True,
            isEntireBoardVisible=is_entire,
            boundingBox=BoundingBox(
                x=x / img_w,
                y=y / img_h,
                width=bw / img_w,
                height=bh / img_h,
            ),
            confidence=conf,
        )

    print(f"[detect_board] no valid candidates from {len(boxes)} boxes")
    return BoardDetection(
        hasBoardDetected=False,
        isEntireBoardVisible=False,
        boundingBox=None,
        confidence=0.0,
    )


def detect_board_with_trained_model(image_path: Path, model_id: str, device: str | None = None) -> BoardDetection:
    """
    Detect a Blokus board using a trained model.

    The trained model is fine-tuned specifically for board detection,
    so it should output class 0 = "board" directly.
    """
    print(f"[detect_board_trained] START image_path={image_path}, model_id={model_id}")

    model = get_trained_model(model_id)
    print(f"[detect_board_trained] model loaded: {model_id}")

    predict_kwargs: dict = {"verbose": False}
    if device:
        predict_kwargs["device"] = device

    results = model(str(image_path), **predict_kwargs)

    if not results or len(results) == 0:
        print("[detect_board_trained] no results from model")
        return BoardDetection(
            hasBoardDetected=False,
            isEntireBoardVisible=False,
            boundingBox=None,
            confidence=0.0,
        )

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("[detect_board_trained] no boxes detected")
        return BoardDetection(
            hasBoardDetected=False,
            isEntireBoardVisible=False,
            boundingBox=None,
            confidence=0.0,
        )

    img_h, img_w = result.orig_shape
    print(f"[detect_board_trained] image: {img_w}x{img_h}, boxes: {len(boxes)}")

    # Find the best "board" detection (class 0)
    best_box = None
    best_conf = 0.0

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        x1, y1, x2, y2 = xyxy
        bw = x2 - x1
        bh = y2 - y1

        print(
            f"[detect_board_trained] box[{i}]: class={cls_id}, conf={conf:.3f}, "
            f"bbox=[{x1:.1f}, {y1:.1f}, {bw:.1f}, {bh:.1f}]"
        )

        # Only consider class 0 (board) with high confidence
        # Real boards typically have 99%+ confidence, false positives are 30-70%
        if cls_id == 0 and conf > best_conf and conf > 0.80:
            best_conf = conf
            best_box = (x1, y1, bw, bh, conf)
            print(f"[detect_board_trained] new best: box[{i}] conf={conf:.4f}")

    if best_box:
        x, y, bw, bh, conf = best_box

        # Calculate normalized bbox
        bbox_dict = {
            "x": x / img_w,
            "y": y / img_h,
            "width": bw / img_w,
            "height": bh / img_h,
        }

        # Run classifier to determine if entire board is visible
        from .classifier import classify_board_visibility

        is_entire, cls_conf = classify_board_visibility(image_path, bbox_dict, model_id)

        # If classifier didn't run (no model), fall back to edge heuristic
        if cls_conf == 0.0:
            margin = 5
            is_entire = (
                x > margin and
                y > margin and
                x + bw < img_w - margin and
                y + bh < img_h - margin
            )
            cls_conf = None

        # Detect corners using pose model if entire board is visible
        corners = None
        corner_conf = None
        if is_entire:
            corners, corner_conf = detect_corners_with_pose_model(image_path, model_id, bbox_dict)

        print(
            f"[detect_board_trained] SUCCESS conf={conf:.3f}, is_entire={is_entire}, "
            f"cls_conf={cls_conf}, corners={corners is not None}, "
            f"bbox=[{bbox_dict['x']:.3f}, {bbox_dict['y']:.3f}, "
            f"{bbox_dict['width']:.3f}, {bbox_dict['height']:.3f}]"
        )

        return BoardDetection(
            hasBoardDetected=True,
            isEntireBoardVisible=is_entire,
            boundingBox=BoundingBox(
                x=bbox_dict["x"],
                y=bbox_dict["y"],
                width=bbox_dict["width"],
                height=bbox_dict["height"],
            ),
            confidence=conf,
            classifierConfidence=cls_conf,
            corners=corners,
            cornerConfidence=corner_conf if corner_conf and corner_conf > 0 else None,
        )

    print(f"[detect_board_trained] no board detected from {len(boxes)} boxes")
    return BoardDetection(
        hasBoardDetected=False,
        isEntireBoardVisible=False,
        boundingBox=None,
        confidence=0.0,
        classifierConfidence=None,
    )
