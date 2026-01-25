import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

# Models directory
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model and all its files."""
    model_path = MODELS_DIR / model_id

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        shutil.rmtree(model_path)
        return {"success": True, "deleted": str(model_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")
