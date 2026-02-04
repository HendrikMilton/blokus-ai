# Start ML Service

Start the ML service (Python/FastAPI) in a tmux session.

## Command

```bash
tmux new-session -d -s ml-service -c /home/hendrik/blokus-ai/ml-service \
  "source .venv/bin/activate && uvicorn app.main:app --reload --port 8017"
```

## Verify

Check if running:
```bash
tmux has-session -t ml-service 2>/dev/null && echo "ml-service is running" || echo "ml-service is not running"
```

View logs:
```bash
tmux capture-pane -t ml-service -p | tail -20
```
