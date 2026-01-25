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
