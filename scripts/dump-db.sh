#!/bin/bash
# Dump PostgreSQL database for transfer to Lambda.ai
# Run this locally before spinning up a Lambda instance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DUMP_FILE="$PROJECT_DIR/blokus_ai.dump"

# Database config (matches api/.env defaults)
DB_CONTAINER="postgres17"
DB_NAME="blokus_ai"
DB_USER="c_connect_user"

echo "Dumping database from Docker container '$DB_CONTAINER'..."

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${DB_CONTAINER}$"; then
    echo "Error: Container '$DB_CONTAINER' is not running"
    echo "Start it with: docker start $DB_CONTAINER"
    exit 1
fi

# Create dump using pg_dump custom format (fast restore, compressed)
docker exec "$DB_CONTAINER" pg_dump \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -Fc \
    --no-owner \
    --no-privileges \
    > "$DUMP_FILE"

# Get file size
SIZE=$(du -h "$DUMP_FILE" | cut -f1)

echo ""
echo "Database dump created: $DUMP_FILE ($SIZE)"
echo ""
echo "Next steps:"
echo "  1. Git commit/push to make dump available on Lambda"
echo "  2. On Lambda: git clone && cd blokus-ai"
echo "  3. On Lambda: scp local:~/.ssh/id_rsa .ssh-key  (for tunnel, optional)"
echo "  4. On Lambda: ./setup-lambda.sh"
