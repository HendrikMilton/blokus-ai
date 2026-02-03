from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.features.ping import router as ping_router
from app.features.upload import router as upload_router
from app.features.samples import router as samples_router
from app.features.detect import router as detect_router
from app.features.train import router as train_router
from app.features.models import router as models_router
from app.features.alphazero import router as alphazero_router
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
app.include_router(upload_router, prefix="/upload", tags=["upload"])
app.include_router(samples_router, prefix="/samples", tags=["samples"])
app.include_router(detect_router, prefix="/detect", tags=["detect"])
app.include_router(train_router, prefix="/train", tags=["train"])
app.include_router(models_router, prefix="/models", tags=["models"])
app.include_router(alphazero_router, prefix="/alphazero", tags=["alphazero"])


@app.get("/")
async def root():
    return {"service": "blokus-ai-ml", "status": "running"}
