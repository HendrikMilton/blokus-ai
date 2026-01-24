from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from typing import Literal
import os

router = APIRouter()

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "images"


class Sample(BaseModel):
    id: str
    filename: str
    status: Literal["pending", "labeled", "verified"]
    created_at: str


class SamplesResponse(BaseModel):
    samples: list[Sample]


@router.get("/", response_model=SamplesResponse)
async def list_samples():
    samples = []

    if DATA_DIR.exists():
        for filepath in DATA_DIR.glob("*.png"):
            sample_id = filepath.stem
            stat = filepath.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

            samples.append(Sample(
                id=sample_id,
                filename=filepath.name,
                status="pending",
                created_at=created_at,
            ))

    # Sort by created_at descending (newest first)
    samples.sort(key=lambda s: s.created_at, reverse=True)

    return SamplesResponse(samples=samples)


@router.get("/{sample_id}", response_model=Sample)
async def get_sample(sample_id: str):
    filepath = DATA_DIR / f"{sample_id}.png"

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Sample not found")

    stat = filepath.stat()
    created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

    return Sample(
        id=sample_id,
        filename=filepath.name,
        status="pending",
        created_at=created_at,
    )


@router.get("/{sample_id}/image")
async def get_sample_image(sample_id: str):
    filepath = DATA_DIR / f"{sample_id}.png"

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Sample not found")

    return FileResponse(
        filepath,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000"}
    )
