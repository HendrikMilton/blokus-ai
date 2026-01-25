#!/bin/bash

# Kill existing processes
pkill -f "cloudflared tunnel" 2>/dev/null
pkill -f "lt --port" 2>/dev/null
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

# Start localtunnel
echo "Starting localtunnel..."
npx localtunnel --port 5177 > /tmp/localtunnel.txt 2>&1 &
LT_PID=$!

# Wait for URL
TUNNEL_URL=""
for i in {1..15}; do
    sleep 1
    if [ -z "$TUNNEL_URL" ]; then
        TUNNEL_URL=$(grep -o 'https://[^[:space:]]*loca.lt' /tmp/localtunnel.txt 2>/dev/null | head -1)
    fi
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
done

if [ -n "$TUNNEL_URL" ]; then
    echo ""
    echo "=========================================="
    echo "Tunnel URL: $TUNNEL_URL"
    echo "=========================================="
    echo ""
    echo "$TUNNEL_URL" | qrencode -t ANSIUTF8 -m 1
    echo ""
    echo "NOTE: First visit shows a reminder page - click 'Click to Continue'"
    echo ""
    echo "Access methods:"
    echo "  - Local:  http://localhost:5177"
    echo "  - LAN:    http://$(hostname -I | awk '{print $1}'):5177"
    echo "  - Tunnel: $TUNNEL_URL"
    echo "  - API:    http://localhost:3009"
    echo ""

    # Handle cleanup on exit
    trap "kill $API_PID $FRONTEND_PID $LT_PID 2>/dev/null; exit" INT TERM

    # Keep script running
    wait
else
    echo "Failed to get tunnel URL."
    echo "Localtunnel output:"
    cat /tmp/localtunnel.txt
    kill $API_PID $FRONTEND_PID 2>/dev/null
    exit 1
fi
