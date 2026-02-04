# Start API and Frontend

Start the API and frontend dev servers in a tmux session with tunnel.

## Command

```bash
tmux new-session -d -s api-frontend -c /home/hendrik/blokus-ai \
  "bun run dev:tunnel"
```

## Verify

Check if running:
```bash
tmux has-session -t api-frontend 2>/dev/null && echo "api-frontend is running" || echo "api-frontend is not running"
```

View logs:
```bash
tmux capture-pane -t api-frontend -p | tail -30
```
