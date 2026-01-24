from fastapi import APIRouter
from pydantic import BaseModel
import base64
import uuid
from pathlib import Path

router = APIRouter()

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "images"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class UploadRequest(BaseModel):
    image: str  # base64 encoded image


class UploadResponse(BaseModel):
    id: str
    filename: str


@router.post("/", response_model=UploadResponse)
async def upload_image(request: UploadRequest):
    image_id = str(uuid.uuid4())
    filename = f"{image_id}.png"
    filepath = DATA_DIR / filename

    image_data = base64.b64decode(request.image)
    filepath.write_bytes(image_data)

    return UploadResponse(id=image_id, filename=filename)
