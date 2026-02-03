#!/bin/bash
# Lambda.ai GPU Instance Setup Script
# Provisions a fresh Lambda instance with all blokus-ai services running
#
# Usage:
#   ./setup-lambda.sh           # Full setup + start services
#   ./setup-lambda.sh --tunnel  # Also start SSH tunnel to blokus.hycapital.io
#
# Prerequisites (run locally before Lambda):
#   ./scripts/dump-db.sh        # Creates blokus_ai.dump
#   git push                    # Push dump to repo
#
# On Lambda (after git clone):
#   scp local:~/.ssh/id_rsa .ssh-key   # Copy SSH key for tunnel (optional)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[x]${NC} $1"; exit 1; }

# Parse arguments
START_TUNNEL=false
for arg in "$@"; do
    case $arg in
        --tunnel) START_TUNNEL=true ;;
    esac
done

# =============================================================================
# Step 1: Install Bun (if not installed)
# =============================================================================
log "Checking Bun installation..."
if ! command -v bun &> /dev/null; then
    log "Installing Bun..."
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="$HOME/.bun"
    export PATH="$BUN_INSTALL/bin:$PATH"
    # Add to shell profile
    echo 'export BUN_INSTALL="$HOME/.bun"' >> ~/.bashrc
    echo 'export PATH="$BUN_INSTALL/bin:$PATH"' >> ~/.bashrc
else
    log "Bun already installed: $(bun --version)"
fi

# =============================================================================
# Step 2: Install Claude Code CLI (if not installed)
# =============================================================================
log "Checking Claude Code CLI installation..."
if ! command -v claude &> /dev/null; then
    log "Installing Claude Code CLI..."
    curl -fsSL https://claude.ai/install.sh | bash
    export PATH="$HOME/.claude/bin:$PATH"
    echo 'export PATH="$HOME/.claude/bin:$PATH"' >> ~/.bashrc
else
    log "Claude Code CLI already installed"
fi

# =============================================================================
# Step 3: Start PostgreSQL Container
# =============================================================================
log "Setting up PostgreSQL..."
CONTAINER_NAME="postgres17"
DB_NAME="blokus_ai"
DB_USER="c_connect_user"
DB_PASS="postgres"

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Start if not running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log "Starting existing PostgreSQL container..."
        docker start "$CONTAINER_NAME"
    else
        log "PostgreSQL container already running"
    fi
else
    log "Creating PostgreSQL container..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        -e POSTGRES_USER="$DB_USER" \
        -e POSTGRES_PASSWORD="$DB_PASS" \
        -e POSTGRES_DB="$DB_NAME" \
        -p 5432:5432 \
        --restart unless-stopped \
        postgres:17
fi

# Wait for PostgreSQL to be ready
log "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker exec "$CONTAINER_NAME" pg_isready -U "$DB_USER" &> /dev/null; then
        break
    fi
    sleep 1
done

# =============================================================================
# Step 4: Restore Database (if dump exists)
# =============================================================================
if [ -f "blokus_ai.dump" ]; then
    log "Restoring database from dump..."
    # Drop and recreate database to ensure clean slate
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -c "DROP DATABASE IF EXISTS ${DB_NAME};" postgres 2>/dev/null || true
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -c "CREATE DATABASE ${DB_NAME};" postgres
    # Restore from dump
    docker exec -i "$CONTAINER_NAME" pg_restore \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --no-owner \
        --no-privileges \
        --clean \
        --if-exists \
        < blokus_ai.dump || warn "Some restore warnings (likely OK)"
    log "Database restored successfully"
else
    warn "No database dump found (blokus_ai.dump)"
    warn "Database will be empty - migrations will create tables on API start"
fi

# =============================================================================
# Step 5: Install Node/Bun Dependencies
# =============================================================================
log "Installing Bun packages..."
bun install

log "Building shared package..."
bun run --cwd shared build

# =============================================================================
# Step 6: Setup Python Environment
# =============================================================================
log "Setting up Python environment..."
cd ml-service

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    log "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate and install deps
source .venv/bin/activate
log "Installing Python dependencies..."
pip install -e ".[dev]" --quiet

# Verify GPU
log "Verifying GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" || warn "GPU check failed"

cd "$SCRIPT_DIR"

# =============================================================================
# Step 7: Create .env files if missing
# =============================================================================
if [ ! -f "api/.env" ]; then
    log "Creating api/.env..."
    cat > api/.env << 'EOF'
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=c_connect_user
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=blokus_ai

# Server
NODE_ENV=dev
PORT=3009

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:5177

# ML Service
ML_SERVICE_URL=http://localhost:8017
EOF
fi

# =============================================================================
# Step 8: Start Services in tmux
# =============================================================================
log "Starting services in tmux sessions..."

# Kill existing sessions (if any)
tmux kill-session -t api 2>/dev/null || true
tmux kill-session -t frontend 2>/dev/null || true
tmux kill-session -t ml 2>/dev/null || true
tmux kill-session -t tunnel 2>/dev/null || true

# Start API
tmux new-session -d -s api -c "$SCRIPT_DIR"
tmux send-keys -t api "bun run --cwd api dev" C-m

# Start Frontend (with --host for external access)
tmux new-session -d -s frontend -c "$SCRIPT_DIR"
tmux send-keys -t frontend "bun run --cwd frontend dev -- --host" C-m

# Start ML Service
tmux new-session -d -s ml -c "$SCRIPT_DIR/ml-service"
tmux send-keys -t ml "source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8017 --reload" C-m

# =============================================================================
# Step 9: Start SSH Tunnel (optional)
# =============================================================================
if [ "$START_TUNNEL" = true ]; then
    if [ -f ".ssh-key" ]; then
        log "Starting SSH tunnel..."
        chmod 600 .ssh-key
        tmux new-session -d -s tunnel -c "$SCRIPT_DIR"
        tmux send-keys -t tunnel "ssh -i .ssh-key -o StrictHostKeyChecking=no -R 8888:localhost:5177 -R 3009:localhost:3009 -N root@157.90.171.156" C-m
        log "Tunnel started - accessible at https://blokus.hycapital.io"
    else
        warn "SSH key not found (.ssh-key)"
        warn "Copy your SSH key: cp ~/.ssh/id_rsa .ssh-key"
        warn "Then run: ./setup-lambda.sh --tunnel"
    fi
fi

# =============================================================================
# Done - Print Summary
# =============================================================================
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Services running in tmux:"
echo "  tmux attach -t api       # API (port 3009)"
echo "  tmux attach -t frontend  # Frontend (port 5177)"
echo "  tmux attach -t ml        # ML Service (port 8017)"
if [ "$START_TUNNEL" = true ] && [ -f ".ssh-key" ]; then
    echo "  tmux attach -t tunnel    # SSH Tunnel"
fi
echo ""
echo "Verify services:"
echo "  curl http://localhost:3009/ping/liveness"
echo "  curl http://localhost:5177"
echo "  curl http://localhost:8017/health"
echo ""
echo "Stop all services:"
echo "  ./stop-services.sh"
echo ""
if [ "$START_TUNNEL" = false ]; then
    echo "To start SSH tunnel later:"
    echo "  ./setup-lambda.sh --tunnel"
    echo ""
fi
