import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.lib.s3 import delete_model_from_s3

router = APIRouter()

# Models directory
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model and all its files (local and S3)."""
    model_path = MODELS_DIR / model_id
    local_deleted = False
    s3_deleted = 0

    # Delete local files
    if model_path.exists():
        try:
            shutil.rmtree(model_path)
            local_deleted = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete local model: {e}")

    # Delete from S3
    try:
        s3_deleted = delete_model_from_s3(model_id)
    except Exception as e:
        print(f"[models] Failed to delete model from S3: {e}")

    if not local_deleted and s3_deleted == 0:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"success": True, "local_deleted": local_deleted, "s3_files_deleted": s3_deleted}
