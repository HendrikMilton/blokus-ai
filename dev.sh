#!/bin/bash

# Kill existing processes
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
echo "Starting frontend..."
bun run --cwd frontend dev &
FRONTEND_PID=$!

sleep 2

echo ""
echo "=========================================="
echo "  Local:  http://localhost:5177"
echo "  LAN:    http://$(hostname -I | awk '{print $1}'):5177"
echo "  API:    http://localhost:3009"
echo "=========================================="
echo ""

# Handle cleanup on exit
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Keep script running
wait
