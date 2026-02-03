#!/bin/bash
# Stop all blokus-ai services running in tmux

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[+]${NC} $1"; }

echo "Stopping blokus-ai services..."

# Kill tmux sessions
for session in api frontend ml tunnel; do
    if tmux has-session -t "$session" 2>/dev/null; then
        tmux kill-session -t "$session"
        log "Stopped $session"
    fi
done

echo ""
echo "All services stopped."
echo ""
echo "Note: PostgreSQL container is still running."
echo "To stop it: docker stop postgres17"
