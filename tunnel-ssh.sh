#!/bin/bash

# Kill existing processes
pkill -f "ssh -R 8888" 2>/dev/null
pkill -f "bun run --cwd frontend dev" 2>/dev/null
pkill -f "bun run --cwd api dev" 2>/dev/null
sleep 1

cd /home/hendrik/blokus-ai

# Build shared first
echo "Building shared..."
bun run --cwd shared build

# Start API
echo "Starting API..."
bun run --cwd api dev &
API_PID=$!

# Start frontend dev server
# Proxy stays enabled for local access - ingress routes /api directly for tunnel access
echo "Starting frontend..."
bun run --cwd frontend dev &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

# Start SSH reverse tunnel (frontend on 8888, API on 3009)
echo "Starting SSH tunnel..."
ssh -R 8888:localhost:5177 -R 3009:localhost:3009 -N root@157.90.171.156 &
SSH_PID=$!

TUNNEL_URL="https://blokus.hycapital.io"

sleep 2

echo ""
echo "=========================================="
echo "Tunnel URL: $TUNNEL_URL"
echo "=========================================="
echo ""
echo "$TUNNEL_URL" | qrencode -t ANSIUTF8 -m 1
echo ""
echo "Access methods:"
echo "  - Local:  http://localhost:5177"
echo "  - LAN:    http://$(hostname -I | awk '{print $1}'):5177"
echo "  - Tunnel: $TUNNEL_URL"
echo "  - API:    http://localhost:3009"
echo ""

# Handle cleanup on exit
trap "kill $API_PID $FRONTEND_PID $SSH_PID 2>/dev/null; exit" INT TERM

# Keep script running
wait
