# Blokus AI

```bash
bun install
cd ml-service && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && cd ..
```

```bash
bun run dev
cd ml-service && source .venv/bin/activate && uvicorn app.main:app --reload --port 8001
```
