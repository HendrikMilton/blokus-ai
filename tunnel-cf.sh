#!/bin/bash

# Kill existing processes
pkill -f "cloudflared tunnel" 2>/dev/null
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

# Wait for frontend to start
sleep 3

# Start cloudflared tunnel with token
echo "Starting cloudflare tunnel..."
~/.local/bin/cloudflared tunnel run --token eyJhIjoiNmY2YjJkYWExY2U5Yzk2YjhmYWVjZmI1YjIxNWNmNWYiLCJ0IjoiN2ZmYTRiYzItZTBjMy00OWY1LTljODgtYTc4MDVlZTk5NTExIiwicyI6Ik5tSm1OVGczWVRZdE5qQm1aQzAwWkROakxUa3lPR0V0WTJNd1lqTXdZalE1WWpZeSJ9 &
CF_PID=$!

TUNNEL_URL="https://blokus.cliniki.co.ke"

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
trap "kill $API_PID $FRONTEND_PID $CF_PID 2>/dev/null; exit" INT TERM

# Keep script running
wait
