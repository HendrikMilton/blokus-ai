from pathlib import Path

from ultralytics import YOLO

from .schemas import BoundingBox, BoardDetection

# Load YOLO26x (largest model) - downloads automatically on first use
_model: YOLO | None = None


def get_model() -> YOLO:
    """Lazy load the YOLO model."""
    global _model
    if _model is None:
        _model = YOLO("yolo26x.pt")
    return _model


def detect_board_yolo(image_path: Path) -> BoardDetection:
    """
    Detect a Blokus board in an image using YOLO26x.

    Uses the largest YOLO26 model for best accuracy.
    Looks for objects that could be a board game (table, board, etc.)
    and returns the best candidate based on size and confidence.
    """
    print(f"[detect_board] START image_path={image_path}")

    model = get_model()
    print("[detect_board] model loaded: yolo26x")

    results = model(str(image_path), verbose=False)

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
