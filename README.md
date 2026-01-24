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

Run each service in a separate terminal:

```bash
# Terminal 1: ML Service (Python/FastAPI)
cd ml-service && source .venv/bin/activate && uvicorn app.main:app --reload --port 8017

# Terminal 2: API + Frontend (Node/Express)
cd api && bun dev:tunnel
```
