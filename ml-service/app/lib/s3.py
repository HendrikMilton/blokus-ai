import os
import tempfile
from pathlib import Path

import boto3

S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "blokus-ai-images")
AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1")
THUMBNAILS_PREFIX = "thumbnails/"

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def download_sample_image(sample_id: str) -> Path:
    """
    Download sample thumbnail from S3 to a temp file and return the path.

    Uses thumbnails (800px max, EXIF-corrected) instead of originals to ensure
    consistency with labeling (which is done on thumbnails in the browser).
    This fixes EXIF rotation issues where originals have orientation metadata
    that isn't applied consistently across different image loaders.
    """
    s3 = get_s3_client()
    key = f"{THUMBNAILS_PREFIX}{sample_id}.jpg"

    # Create temp file
    temp_dir = Path(tempfile.gettempdir()) / "blokus-ai-samples"
    temp_dir.mkdir(exist_ok=True)
    local_path = temp_dir / f"{sample_id}.jpg"

    # Check if already downloaded
    if local_path.exists():
        return local_path

    # Download from S3
    s3.download_file(S3_BUCKET, key, str(local_path))
    return local_path


def list_sample_ids() -> list[str]:
    """List all sample IDs from S3."""
    s3 = get_s3_client()

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=ORIGINALS_PREFIX)

    sample_ids = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".png"):
            sample_id = key.replace(ORIGINALS_PREFIX, "").replace(".png", "")
            sample_ids.append(sample_id)

    return sample_ids


def sample_exists(sample_id: str) -> bool:
    """Check if a sample exists in S3."""
    s3 = get_s3_client()
    key = f"{ORIGINALS_PREFIX}{sample_id}.png"

    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


CROPPED_PREFIX = "cropped/"
CLOUDFRONT_DOMAIN = os.environ.get("CLOUDFRONT_DOMAIN", "d26azxov2byjpl.cloudfront.net")


def upload_cropped_image(sample_id: str, method: str, image_bytes: bytes) -> str:
    """
    Upload a cropped/perspective-corrected image to S3.

    Args:
        sample_id: The sample ID
        method: The corner detection method used (best, hough, edge)
        image_bytes: JPEG image bytes

    Returns:
        CloudFront URL for the uploaded image
    """
    s3 = get_s3_client()
    key = f"{CROPPED_PREFIX}{method}/{sample_id}.jpg"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )

    return f"https://{CLOUDFRONT_DOMAIN}/{key}"


def get_cropped_image_url(sample_id: str, method: str) -> str:
    """Get the CloudFront URL for a cropped image."""
    key = f"{CROPPED_PREFIX}{method}/{sample_id}.jpg"
    return f"https://{CLOUDFRONT_DOMAIN}/{key}"


# ============================================================================
# Model S3 Storage (separate bucket for YOLO training run models)
# ============================================================================

MODELS_BUCKET = os.environ.get("AWS_S3_MODELS_BUCKET", "blokus-ai-models")
MODELS_S3_PREFIX = "models/"


def upload_model_directory(run_id: str, local_dir: Path) -> str:
    """
    Upload all files from a training run directory to S3.

    Uploads everything (weights, CSVs, PNGs, YAMLs) for debugging purposes.
    Returns the S3 key prefix.
    """
    s3 = get_s3_client()
    s3_prefix = f"{MODELS_S3_PREFIX}{run_id}/"

    uploaded = 0
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_dir)
            key = f"{s3_prefix}{relative}"
            print(f"[s3] Uploading {relative} -> s3://{MODELS_BUCKET}/{key}")
            s3.upload_file(str(file_path), MODELS_BUCKET, key)
            uploaded += 1

    print(f"[s3] Uploaded {uploaded} files to s3://{MODELS_BUCKET}/{s3_prefix}")
    return s3_prefix


def download_model_file(run_id: str, filename: str, local_path: Path) -> Path:
    """
    Download a single model file from S3 to local path.
    Skips download if file already exists locally (cache).
    """
    if local_path.exists():
        return local_path

    s3 = get_s3_client()
    key = f"{MODELS_S3_PREFIX}{run_id}/{filename}"

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[s3] Downloading s3://{MODELS_BUCKET}/{key} -> {local_path}")
    s3.download_file(MODELS_BUCKET, key, str(local_path))
    return local_path


def ensure_model_local(run_id: str, models_dir: Path) -> Path:
    """
    Ensure model .pt files are available locally for inference.
    Downloads from S3 if not present on disk. Returns local model dir.
    Only downloads .pt files needed for inference, not training artifacts.
    """
    local_dir = models_dir / run_id

    # Check if we already have the main model file
    best_pt = local_dir / "best.pt"
    weights_best_pt = local_dir / "weights" / "best.pt"
    if best_pt.exists() or weights_best_pt.exists():
        return local_dir

    # List .pt files in S3 for this run and download them
    s3 = get_s3_client()
    s3_prefix = f"{MODELS_S3_PREFIX}{run_id}/"

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=MODELS_BUCKET, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".pt"):
                    relative = key[len(s3_prefix):]
                    local_path = local_dir / relative
                    if not local_path.exists():
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        print(f"[s3] Downloading s3://{MODELS_BUCKET}/{key} -> {local_path}")
                        s3.download_file(MODELS_BUCKET, key, str(local_path))
    except Exception as e:
        print(f"[s3] Failed to download model files for {run_id}: {e}")
        raise FileNotFoundError(f"Model not found in S3: {run_id}") from e

    return local_dir


def delete_model_from_s3(run_id: str) -> int:
    """
    Delete all files for a training run from S3.
    Returns the number of files deleted.
    """
    s3 = get_s3_client()
    s3_prefix = f"{MODELS_S3_PREFIX}{run_id}/"

    deleted = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=MODELS_BUCKET, Prefix=s3_prefix):
            objects = page.get("Contents", [])
            if objects:
                delete_keys = [{"Key": obj["Key"]} for obj in objects]
                s3.delete_objects(
                    Bucket=MODELS_BUCKET,
                    Delete={"Objects": delete_keys},
                )
                deleted += len(delete_keys)
    except Exception as e:
        print(f"[s3] Failed to delete model files for {run_id}: {e}")

    if deleted > 0:
        print(f"[s3] Deleted {deleted} files from s3://{MODELS_BUCKET}/{s3_prefix}")
    return deleted
