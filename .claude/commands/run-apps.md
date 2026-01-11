---
description: Start the blokus-ai API and frontend for development.
allowed-tools: [Bash, Read, Glob, Grep, TaskOutput]
---

# Run Apps Command

Start the blokus-ai development environment.

## Project Structure

- **API:** Express + TypeScript (port 3000)
- **Frontend:** React + Vite (port 5173/5174)
- **Database:** PostgreSQL (port 5432)

## Step 1: Start PostgreSQL Container

```bash
# Try to start existing container, or create new one
docker start blokus-ai-db 2>/dev/null || \
docker run -d --name blokus-ai-db \
  -e POSTGRES_USER=dev \
  -e POSTGRES_PASSWORD=dev \
  -e POSTGRES_DB=blokus_ai \
  -p 5432:5432 \
  postgres:17
```

Wait for PostgreSQL to be ready:

```bash
echo "Waiting for PostgreSQL..."
for i in {1..30}; do
  docker exec blokus-ai-db pg_isready -U dev && break
  sleep 1
done
```

## Step 2: Kill Existing App Processes

```bash
# Kill by common ports
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
lsof -ti:5174 | xargs kill -9 2>/dev/null || true

# Kill by process patterns
pkill -f "bun.*dev" 2>/dev/null || true
pkill -f "bun.*start" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
```

## Step 3: Build Shared Package

```bash
cd /home/hendrik/Documents/blokus-ai/shared && bun run build
```

## Step 4: Start Applications

Start both apps using `run_in_background: true`:

### Option A: Using Root Start Script

```bash
cd /home/hendrik/Documents/blokus-ai && bun run start
```

### Option B: Start Individually

```bash
# Start API (run in background)
cd /home/hendrik/Documents/blokus-ai/api && bun run dev

# Start Frontend (run in background)
cd /home/hendrik/Documents/blokus-ai/frontend && bun run dev
```

## Step 5: Health Checks

Wait for apps to initialize, then verify:

```bash
sleep 5

# Database health
docker exec blokus-ai-db pg_isready -U dev && echo "PostgreSQL: Healthy" || echo "PostgreSQL: Unhealthy"

# API health
curl -sf http://localhost:3000/ping/liveness && echo "API: Healthy" || echo "API: Unhealthy"
curl -sf http://localhost:3000/ping/readiness && echo "API DB: Connected" || echo "API DB: Not connected"

# Frontend health
curl -sf -o /dev/null http://localhost:5174 && echo "Frontend: Healthy" || echo "Frontend: Unhealthy"
```

## Step 6: Report Status

```markdown
## Apps Running

### Containers
| Container | Port | Status |
|-----------|------|--------|
| PostgreSQL (blokus-ai-db) | 5432 | Healthy |

### Applications
| App | Port | Status |
|-----|------|--------|
| API | 3000 | Healthy |
| Frontend | 5174 | Healthy |

### URLs
- Frontend: http://localhost:5174
- API: http://localhost:3000
- API Health: http://localhost:3000/ping/liveness

### Commands

# View app logs
Use TaskOutput tool with the task IDs above

# Stop apps
pkill -f "bun run"

# Stop database
docker stop blokus-ai-db

# Restart everything
Run /run-apps again
```

## Troubleshooting

**Database won't start:**
```bash
docker logs blokus-ai-db --tail 50
lsof -i :5432
```

**API won't start:**
- Check if database is ready
- Check API logs via TaskOutput
- Verify environment variables in `api/.env`

**Frontend won't start:**
- Check if port 5174 is in use
- Check frontend logs via TaskOutput
