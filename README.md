# Blokus AI

AI-powered Blokus board game assistant with board detection and move analysis.

## Setup

```bash
# Install Node dependencies
bun install

# Setup ML service
cd ml-service && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && cd ..
```

## Running the Application

### All Services (Development)

Run each service in a separate terminal:

```bash
# Terminal 1: ML Service (Python/FastAPI)
cd ml-service && source .venv/bin/activate && uvicorn app.main:app --reload --port 8017

# Terminal 2: API + Frontend (Node/Express)
cd api && bun dev:tunnel
```

### Individual Services

**ML Service** (port 8017):
```bash
cd ml-service
source .venv/bin/activate
uvicorn app.main:app --reload --port 8017
```

**API** (port 3000):
```bash
cd api
bun run dev
```

**Frontend** (port 5173):
```bash
cd frontend
bun run dev
```

## Environment Variables

### ML Service

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FOUNDATION_MODELS` | `true` | Use Grounding DINO + SAM2 + DINOv3 pipeline |
| `PORT` | `8000` | Server port |

When `USE_FOUNDATION_MODELS=true`, all three models (Grounding DINO, SAM2, DINOv3) must be available or the service will fail to start.

### API

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_SERVICE_URL` | `http://localhost:8017` | ML service URL |
| `PORT` | `3000` | Server port |

## Project Structure

```
blokus-ai/
├── api/            # Node.js/Express API server
├── frontend/       # React/Vite frontend
├── ml-service/     # Python/FastAPI ML service
├── shared/         # Shared TypeScript types
└── scripts/        # Utility scripts
```

## ML Admin Interface

Access the training data labeling interface at `http://localhost:5173/admin`.

Features:
- Upload board images for labeling
- Interactive board state editor
- Keyboard shortcuts (1-4 for colors, 0 for clear, Ctrl+Z/Y for undo/redo)
- Train KNN classifier with labeled samples
