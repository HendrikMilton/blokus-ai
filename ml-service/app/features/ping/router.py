from fastapi import APIRouter

router = APIRouter()


@router.get("/liveness")
async def liveness():
    """Basic liveness probe."""
    return {"status": "ok"}


@router.get("/readiness")
async def readiness():
    """Readiness probe - service is ready to accept traffic."""
    return {"status": "ok"}


@router.get("/health")
async def health():
    """Health check with service info."""
    return {
        "status": "healthy",
        "service": "blokus-ai-ml",
        "version": "0.1.0",
    }
