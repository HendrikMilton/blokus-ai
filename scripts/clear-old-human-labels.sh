#!/bin/bash

# Clear human labels from old samples (before 2026-01-25 20:00:00)
# These labels were made on incorrectly rotated thumbnails

echo "Clearing human labels from old samples..."

docker exec postgres17 psql -U c_connect_user -d blokus_ai -c "
UPDATE board_detections
SET
  human_has_board = NULL,
  human_is_entire_visible = NULL,
  human_bbox_x = NULL,
  human_bbox_y = NULL,
  human_bbox_width = NULL,
  human_bbox_height = NULL,
  verified_at = NULL,
  human_labeled_at = NULL
WHERE created_at < '2026-01-25 20:00:00';
"

echo "Done!"
