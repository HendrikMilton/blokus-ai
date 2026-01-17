"""Admin schemas for training data management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .ml import BoardCell


class SampleStatus(str, Enum):
    """Status of a training sample."""
    PENDING = "pending"
    LABELED = "labeled"
    VERIFIED = "verified"


class SampleSource(str, Enum):
    """Source of a training sample."""
    UPLOAD = "upload"
    CAMERA = "camera"
    SYNTHETIC = "synthetic"


class TrainingSampleCreate(BaseModel):
    """Request to create a training sample via image upload."""
    image: str = Field(..., description="Base64 encoded image")
    source: SampleSource = SampleSource.UPLOAD
    notes: Optional[str] = None


class TrainingSampleUpdate(BaseModel):
    """Request to update a training sample."""
    labeled_state: Optional[list[list[BoardCell]]] = None
    notes: Optional[str] = None
    status: Optional[SampleStatus] = None


class TrainingSampleResponse(BaseModel):
    """Response for a training sample."""
    id: UUID
    image_path: str
    image_width: int
    image_height: int
    detected_state: list[list[BoardCell]]
    labeled_state: Optional[list[list[BoardCell]]] = None
    source: SampleSource
    notes: Optional[str] = None
    status: SampleStatus
    labeled_at: Optional[datetime] = None
    features_extracted: bool = False
    created_at: datetime
    updated_at: datetime


class TrainingSampleListResponse(BaseModel):
    """Paginated list of training samples."""
    samples: list[TrainingSampleResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class AdminStats(BaseModel):
    """Admin dashboard statistics."""
    total_samples: int = 0
    pending_samples: int = 0
    labeled_samples: int = 0
    verified_samples: int = 0
    total_cells_labeled: int = 0
    color_distribution: dict[str, int] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    """Request to train the model."""
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)
    min_samples: int = Field(default=10, ge=1)


class TrainResponse(BaseModel):
    """Response from training the model."""
    success: bool
    message: str
    samples_used: int = 0
    validation_accuracy: Optional[float] = None
    train_accuracy: Optional[float] = None
    color_accuracies: Optional[dict[str, float]] = None
    model_path: Optional[str] = None


class PredictResponse(BaseModel):
    """Response from re-running prediction on a sample."""
    detected_state: list[list[BoardCell]]
