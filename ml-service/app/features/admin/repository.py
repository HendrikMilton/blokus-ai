"""Repository for training sample storage using PostgreSQL."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import psycopg2
from psycopg2.extras import RealDictCursor

from app.schemas.admin import (
    AdminStats,
    SampleSource,
    SampleStatus,
    TrainingSampleResponse,
    TrainingSampleUpdate,
)
from app.schemas.ml import BoardCell


def get_db_connection():
    """Get a PostgreSQL database connection."""
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ.get("POSTGRES_USER", "c_connect_user"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database=os.environ.get("POSTGRES_DATABASE", "blokus_ai"),
    )


class TrainingSampleRepository:
    """PostgreSQL-based repository for training samples."""

    def __init__(self, base_path: Path | None = None):
        """Initialize repository with base path for image storage."""
        self.base_path = base_path or Path(__file__).parent.parent.parent.parent / "data"
        self.images_path = self.base_path / "training_images"
        self.feature_cache_path = self.base_path / "feature_cache"
        self.models_path = self.base_path / "models"

        # Ensure directories exist
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.feature_cache_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

    def _to_response(self, row: dict) -> TrainingSampleResponse:
        """Convert database row to response model."""
        detected_state = [
            [BoardCell(**cell) for cell in r]
            for r in row["detected_state"]
        ]
        labeled_state = None
        if row.get("labeled_state"):
            labeled_state = [
                [BoardCell(**cell) for cell in r]
                for r in row["labeled_state"]
            ]

        return TrainingSampleResponse(
            id=row["id"],
            image_path=row["image_path"],
            image_width=row["image_width"],
            image_height=row["image_height"],
            detected_state=detected_state,
            labeled_state=labeled_state,
            source=SampleSource(row["source"]),
            notes=row.get("notes"),
            status=SampleStatus(row["status"]),
            labeled_at=row.get("labeled_at"),
            features_extracted=row.get("features_extracted", False),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
        detected_state: list[list[BoardCell]],
        source: SampleSource = SampleSource.UPLOAD,
        notes: str | None = None,
    ) -> TrainingSampleResponse:
        """Create a new training sample."""
        sample_id = uuid4()
        now = datetime.now()

        # Save image to filesystem
        image_filename = f"{sample_id}.jpg"
        image_path = self.images_path / image_filename
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Convert board state to JSON-serializable format
        detected_state_json = [[cell.model_dump() for cell in row] for row in detected_state]

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO training_samples (
                        id, image_path, image_width, image_height,
                        detected_state, source, notes, status,
                        features_extracted, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING *
                    """,
                    (
                        str(sample_id),
                        str(image_path),
                        image_width,
                        image_height,
                        psycopg2.extras.Json(detected_state_json),
                        source.value,
                        notes,
                        SampleStatus.PENDING.value,
                        False,
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()
                conn.commit()

        return self._to_response(row)

    def get(self, sample_id: UUID) -> TrainingSampleResponse | None:
        """Get a training sample by ID."""
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM training_samples WHERE id = %s",
                    (str(sample_id),),
                )
                row = cur.fetchone()

        if row is None:
            return None
        return self._to_response(row)

    def list(
        self,
        status: SampleStatus | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[TrainingSampleResponse], int]:
        """List training samples with optional filtering and pagination."""
        offset = (page - 1) * page_size

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get total count
                if status:
                    cur.execute(
                        "SELECT COUNT(*) as count FROM training_samples WHERE status = %s",
                        (status.value,),
                    )
                else:
                    cur.execute("SELECT COUNT(*) as count FROM training_samples")
                total = cur.fetchone()["count"]

                # Get paginated results
                if status:
                    cur.execute(
                        """
                        SELECT * FROM training_samples
                        WHERE status = %s
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (status.value, page_size, offset),
                    )
                else:
                    cur.execute(
                        """
                        SELECT * FROM training_samples
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (page_size, offset),
                    )
                rows = cur.fetchall()

        samples = [self._to_response(row) for row in rows]
        return samples, total

    def update(
        self,
        sample_id: UUID,
        update: TrainingSampleUpdate,
    ) -> TrainingSampleResponse | None:
        """Update a training sample."""
        now = datetime.now()

        # Build update query dynamically
        updates = ["updated_at = %s"]
        values = [now]

        if update.labeled_state is not None:
            labeled_state_json = [[cell.model_dump() for cell in row] for row in update.labeled_state]
            updates.append("labeled_state = %s")
            values.append(psycopg2.extras.Json(labeled_state_json))
            updates.append("labeled_at = %s")
            values.append(now)

        if update.notes is not None:
            updates.append("notes = %s")
            values.append(update.notes)

        if update.status is not None:
            updates.append("status = %s")
            values.append(update.status.value)
        elif update.labeled_state is not None:
            # Auto-update status to labeled if labeled_state was set
            updates.append("status = CASE WHEN status = 'pending' THEN 'labeled' ELSE status END")

        values.append(str(sample_id))

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    UPDATE training_samples
                    SET {", ".join(updates)}
                    WHERE id = %s
                    RETURNING *
                    """,
                    values,
                )
                row = cur.fetchone()
                conn.commit()

        if row is None:
            return None
        return self._to_response(row)

    def delete(self, sample_id: UUID) -> bool:
        """Delete a training sample."""
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get image path first
                cur.execute(
                    "SELECT image_path FROM training_samples WHERE id = %s",
                    (str(sample_id),),
                )
                row = cur.fetchone()

                if row is None:
                    return False

                # Delete from database
                cur.execute(
                    "DELETE FROM training_samples WHERE id = %s",
                    (str(sample_id),),
                )
                conn.commit()

        # Delete image file
        image_path = Path(row["image_path"])
        if image_path.exists():
            image_path.unlink()

        # Delete feature cache if exists
        feature_cache = self.feature_cache_path / f"{sample_id}.npy"
        if feature_cache.exists():
            feature_cache.unlink()

        return True

    def get_stats(self) -> AdminStats:
        """Get statistics about training samples."""
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get counts by status
                cur.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'pending') as pending,
                        COUNT(*) FILTER (WHERE status = 'labeled') as labeled,
                        COUNT(*) FILTER (WHERE status = 'verified') as verified
                    FROM training_samples
                """)
                counts = cur.fetchone()

                # Get color distribution from labeled samples
                cur.execute("""
                    SELECT labeled_state
                    FROM training_samples
                    WHERE labeled_state IS NOT NULL
                """)
                rows = cur.fetchall()

        total_cells = 0
        color_distribution: dict[str, int] = {
            "blue": 0,
            "yellow": 0,
            "red": 0,
            "green": 0,
            "empty": 0,
        }

        for row in rows:
            if row["labeled_state"]:
                for board_row in row["labeled_state"]:
                    for cell in board_row:
                        total_cells += 1
                        color = cell.get("color")
                        if color:
                            color_distribution[color] = color_distribution.get(color, 0) + 1
                        else:
                            color_distribution["empty"] += 1

        return AdminStats(
            total_samples=counts["total"],
            pending_samples=counts["pending"],
            labeled_samples=counts["labeled"],
            verified_samples=counts["verified"],
            total_cells_labeled=total_cells,
            color_distribution=color_distribution,
        )

    def get_labeled_samples(self) -> list[TrainingSampleResponse]:
        """Get all labeled and verified samples for training."""
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM training_samples
                    WHERE status IN ('labeled', 'verified')
                    AND labeled_state IS NOT NULL
                    ORDER BY created_at DESC
                """)
                rows = cur.fetchall()

        return [self._to_response(row) for row in rows]

    def get_image_path(self, sample_id: UUID) -> Path | None:
        """Get the image path for a sample."""
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT image_path FROM training_samples WHERE id = %s",
                    (str(sample_id),),
                )
                row = cur.fetchone()

        if row is None:
            return None
        return Path(row["image_path"])

    def update_detected_state(
        self,
        sample_id: UUID,
        detected_state: list[list[BoardCell]],
    ) -> TrainingSampleResponse | None:
        """Update the detected state of a sample (after re-prediction)."""
        now = datetime.now()
        detected_state_json = [[cell.model_dump() for cell in row] for row in detected_state]

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE training_samples
                    SET detected_state = %s, updated_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (psycopg2.extras.Json(detected_state_json), now, str(sample_id)),
                )
                row = cur.fetchone()
                conn.commit()

        if row is None:
            return None
        return self._to_response(row)


# Singleton instance
_repository: TrainingSampleRepository | None = None


def get_repository() -> TrainingSampleRepository:
    """Get the singleton repository instance."""
    global _repository
    if _repository is None:
        _repository = TrainingSampleRepository()
    return _repository
