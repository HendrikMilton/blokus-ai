#!/bin/bash

# Get old sample IDs (before 2026-01-25 20:00:00)
IDS=$(docker exec postgres17 psql -U c_connect_user -d blokus_ai -t -c \
  "SELECT sample_id FROM board_detections WHERE created_at < '2026-01-25 20:00:00';")

COUNT=$(echo "$IDS" | grep -c '[a-f0-9]')
echo "Regenerating $COUNT thumbnails..."

i=0
for id in $IDS; do
  id=$(echo "$id" | tr -d ' ')
  [ -z "$id" ] && continue

  ((i++))
  echo "[$i/$COUNT] $id"

  curl -s -X POST http://localhost:3009/api/ml/upload/complete \
    -H "Content-Type: application/json" \
    -d "{\"id\":\"$id\"}" > /dev/null
done

echo ""
echo "Done regenerating thumbnails. Invalidating CloudFront cache..."

aws cloudfront create-invalidation \
  --distribution-id E17GGPAI4TT7FP \
  --paths "/thumbnails/*"

echo "Complete!"
