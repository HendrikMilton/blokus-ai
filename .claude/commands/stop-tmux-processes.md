# Stop Tmux Processes

Stop both ml-service and api-frontend tmux sessions.

## Command

```bash
tmux kill-session -t ml-service 2>/dev/null && echo "Stopped ml-service" || echo "ml-service was not running"
tmux kill-session -t api-frontend 2>/dev/null && echo "Stopped api-frontend" || echo "api-frontend was not running"
```

## Verify

```bash
tmux list-sessions 2>/dev/null || echo "No tmux sessions running"
```
