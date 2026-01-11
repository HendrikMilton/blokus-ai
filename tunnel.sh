#!/bin/bash

# Kill existing cloudflared tunnel processes
pkill -f "cloudflared tunnel" 2>/dev/null
sleep 1

# Start API tunnel
~/.local/bin/cloudflared tunnel --url http://localhost:3009 > /tmp/cloudflared_api.txt 2>&1 &
API_PID=$!

# Start Frontend tunnel
~/.local/bin/cloudflared tunnel --url http://localhost:5177 > /tmp/cloudflared_frontend.txt 2>&1 &
FRONTEND_PID=$!

echo "Starting tunnels..."

# Wait for both URLs
API_URL=""
FRONTEND_URL=""
for i in {1..15}; do
    sleep 1
    if [ -z "$API_URL" ]; then
        API_URL=$(grep -o 'https://[^[:space:]]*trycloudflare.com' /tmp/cloudflared_api.txt 2>/dev/null | head -1)
    fi
    if [ -z "$FRONTEND_URL" ]; then
        FRONTEND_URL=$(grep -o 'https://[^[:space:]]*trycloudflare.com' /tmp/cloudflared_frontend.txt 2>/dev/null | head -1)
    fi
    if [ -n "$API_URL" ] && [ -n "$FRONTEND_URL" ]; then
        break
    fi
done

if [ -n "$API_URL" ] && [ -n "$FRONTEND_URL" ]; then
    echo ""
    echo "API Tunnel: $API_URL"
    echo "Frontend Tunnel: $FRONTEND_URL"
    echo ""

    # Update frontend .env with API URL
    echo "VITE_API_URL=$API_URL" > /home/hendrik/blokus-ai/frontend/.env
    echo "Updated frontend/.env with API URL"
    echo ""

    echo "$FRONTEND_URL" | qrencode -t ANSIUTF8 -m 1
    echo ""
    echo "Scan QR code with your phone"
    echo ""

    # Start frontend dev server now that .env is set
    echo "Starting frontend dev server..."
    cd /home/hendrik/blokus-ai && bun run --cwd frontend dev &
    FRONTEND_DEV_PID=$!

    # Keep script running
    wait $API_PID $FRONTEND_PID $FRONTEND_DEV_PID
else
    echo "Failed to get tunnel URLs."
    echo "API output:"
    cat /tmp/cloudflared_api.txt
    echo ""
    echo "Frontend output:"
    cat /tmp/cloudflared_frontend.txt
    exit 1
fi
