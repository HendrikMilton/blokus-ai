"""
One-time script to migrate existing YOLO training run model files to S3.

Usage:
    cd ml-service
    python -m scripts.migrate_models_to_s3

After running, also update the database:
    docker exec postgres17 psql -U c_connect_user -d blokus_ai -c \
        "UPDATE training_runs SET model_s3_url = 'models/' || id || '/' WHERE status = 'completed' AND model_s3_url IS NULL;"
"""

import re
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lib.s3 import upload_model_directory, MODELS_BUCKET

MODELS_DIR = Path(__file__).parent.parent / "models"

# UUID pattern to identify training run directories
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def main():
    if not MODELS_DIR.exists():
        print(f"Models directory not found: {MODELS_DIR}")
        return

    # Find all UUID-named directories (training run models)
    run_dirs = [
        d for d in MODELS_DIR.iterdir()
        if d.is_dir() and UUID_PATTERN.match(d.name)
    ]

    if not run_dirs:
        print("No training run model directories found.")
        return

    print(f"Found {len(run_dirs)} training run directories to migrate")
    print(f"Target bucket: s3://{MODELS_BUCKET}/")
    print()

    total_size = 0
    for run_dir in run_dirs:
        dir_size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
        total_size += dir_size

    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print()

    uploaded = 0
    failed = 0

    for run_dir in sorted(run_dirs):
        run_id = run_dir.name
        dir_size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
        file_count = sum(1 for f in run_dir.rglob("*") if f.is_file())

        print(f"[{uploaded + failed + 1}/{len(run_dirs)}] {run_id} ({file_count} files, {dir_size / (1024**2):.1f} MB)")

        try:
            s3_prefix = upload_model_directory(run_id, run_dir)
            print(f"  -> Uploaded to {s3_prefix}")
            uploaded += 1
        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed += 1

    print()
    print(f"Migration complete: {uploaded} succeeded, {failed} failed")
    print()
    print("Now update the database:")
    print('  docker exec postgres17 psql -U c_connect_user -d blokus_ai -c \\')
    print("    \"UPDATE training_runs SET model_s3_url = 'models/' || id || '/' WHERE status = 'completed' AND model_s3_url IS NULL;\"")


if __name__ == "__main__":
    main()
