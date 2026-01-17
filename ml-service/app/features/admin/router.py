"""Router for admin endpoints."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from app.schemas.admin import (
    AdminStats,
    PredictResponse,
    SampleStatus,
    TrainingSampleCreate,
    TrainingSampleListResponse,
    TrainingSampleResponse,
    TrainingSampleUpdate,
    TrainRequest,
    TrainResponse,
)

from .service import get_service

router = APIRouter()


@router.post("/samples", response_model=TrainingSampleResponse)
async def create_sample(
    request: Request,
    body: TrainingSampleCreate,
):
    """Upload an image and create a training sample."""
    cid = request.state.correlation_id
    service = get_service()

    try:
        sample = service.upload_sample(
            image_b64=body.image,
            source=body.source,
            notes=body.notes,
            cid=cid,
        )
        return sample
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/samples", response_model=TrainingSampleListResponse)
async def list_samples(
    request: Request,
    status: Optional[SampleStatus] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """List training samples with optional filtering and pagination."""
    cid = request.state.correlation_id
    service = get_service()

    samples, total, total_pages = service.list_samples(
        status=status,
        page=page,
        page_size=page_size,
        cid=cid,
    )

    return TrainingSampleListResponse(
        samples=samples,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/samples/{sample_id}", response_model=TrainingSampleResponse)
async def get_sample(
    request: Request,
    sample_id: UUID,
):
    """Get a single training sample by ID."""
    cid = request.state.correlation_id
    service = get_service()

    sample = service.get_sample(sample_id, cid=cid)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return sample


@router.get("/samples/{sample_id}/image")
async def get_sample_image(
    request: Request,
    sample_id: UUID,
):
    """Get the image for a training sample."""
    cid = request.state.correlation_id
    service = get_service()

    sample = service.get_sample(sample_id, cid=cid)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    from pathlib import Path

    image_path = Path(sample.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    with open(image_path, "rb") as f:
        image_data = f.read()

    return Response(
        content=image_data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.patch("/samples/{sample_id}", response_model=TrainingSampleResponse)
async def update_sample(
    request: Request,
    sample_id: UUID,
    body: TrainingSampleUpdate,
):
    """Update a training sample's labels, notes, or status."""
    cid = request.state.correlation_id
    service = get_service()

    sample = service.update_sample(sample_id, body, cid=cid)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return sample


@router.delete("/samples/{sample_id}")
async def delete_sample(
    request: Request,
    sample_id: UUID,
):
    """Delete a training sample."""
    cid = request.state.correlation_id
    service = get_service()

    deleted = service.delete_sample(sample_id, cid=cid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sample not found")

    return {"success": True}


@router.post("/samples/{sample_id}/predict", response_model=TrainingSampleResponse)
async def repredict_sample(
    request: Request,
    sample_id: UUID,
):
    """Re-run detection on a sample."""
    cid = request.state.correlation_id
    service = get_service()

    sample = service.repredict_sample(sample_id, cid=cid)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return sample


@router.get("/stats", response_model=AdminStats)
async def get_stats(request: Request):
    """Get admin statistics."""
    cid = request.state.correlation_id
    service = get_service()

    return service.get_stats(cid=cid)


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: Request,
    body: TrainRequest = TrainRequest(),
):
    """Train the model using labeled samples."""
    cid = request.state.correlation_id
    service = get_service()

    return service.train_model(
        validation_split=body.validation_split,
        min_samples=body.min_samples,
        cid=cid,
    )
