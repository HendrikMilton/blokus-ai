from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.features.admin import router as admin_router
from app.features.analysis import router as analysis_router
from app.features.detection import router as detection_router
from app.features.ping import router as ping_router
from app.middleware import CorrelationIdMiddleware

app = FastAPI(
    title="Blokus AI ML Service",
    version="0.1.0",
    description="Board detection and move analysis for Blokus",
)

# Middleware (order matters - first added is outermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationIdMiddleware)

# Mount feature routers
app.include_router(ping_router, prefix="/ping", tags=["health"])
app.include_router(detection_router, prefix="/detect-board", tags=["detection"])
app.include_router(analysis_router, prefix="/analyze", tags=["analysis"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])


@app.get("/")
async def root():
    return {"service": "blokus-ai-ml", "status": "running"}
