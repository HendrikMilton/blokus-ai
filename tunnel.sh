#!/bin/bash

# Kill existing cloudflared
pkill -f cloudflared 2>/dev/null

# Start cloudflared with output capture
~/.local/bin/cloudflared tunnel --url http://localhost:5177 > /tmp/cloudflared_output.txt 2>&1 &

# Wait for URL
echo "Starting tunnel..."
for i in {1..10}; do
    sleep 1
    URL=$(grep -o 'https://[^[:space:]]*trycloudflare.com' /tmp/cloudflared_output.txt 2>/dev/null | head -1)
    if [ -n "$URL" ]; then
        break
    fi
done

if [ -n "$URL" ]; then
    echo ""
    echo "Tunnel URL: $URL"
    echo ""
    echo "$URL" | qrencode -t ANSI
    echo ""
    echo "Scan QR code with your phone"
    echo ""
    # Keep script running to maintain tunnel
    tail -f /tmp/cloudflared_output.txt
else
    echo "Failed to get tunnel URL. Check /tmp/cloudflared_output.txt"
    cat /tmp/cloudflared_output.txt
    exit 1
fi
