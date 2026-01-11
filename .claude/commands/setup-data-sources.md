---
description: Set up Playwright and Postgres MCP data sources for blokus-ai.
allowed-tools: [Bash, Read, Write, Edit, Glob, AskUserQuestion]
---

# Setup Data Sources

Configure MCP (Model Context Protocol) servers for Playwright browser automation and Postgres database access for blokus-ai.

## Step 1: Check Current Configuration

```bash
cat /home/hendrik/Documents/blokus-ai/.claude/settings.json 2>/dev/null || echo "No settings.json found"
cat /home/hendrik/Documents/blokus-ai/.claude/settings.local.json 2>/dev/null || echo "No settings.local.json found"
```

## Step 2: Database Configuration

The blokus-ai project uses PostgreSQL with these default credentials:

From `api/.env`:
- Host: localhost
- Port: 5432
- Database: blokus_ai
- User: dev
- Password: dev

## Step 3: Create MCP Configuration

Create or update `.claude/settings.local.json`:

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-playwright"],
      "env": {
        "PLAYWRIGHT_BROWSER": "chromium",
        "PLAYWRIGHT_HEADLESS": "true"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://dev:dev@localhost:5432/blokus_ai"
      }
    }
  }
}
```

## Step 4: Verify Installation

```bash
# Test Playwright MCP
npx -y @anthropic-ai/mcp-server-playwright --help 2>/dev/null && echo "Playwright MCP: OK" || echo "Playwright MCP: Install needed"

# Test Postgres MCP
npx -y @anthropic-ai/mcp-server-postgres --help 2>/dev/null && echo "Postgres MCP: OK" || echo "Postgres MCP: Install needed"
```

## Step 5: Ensure .gitignore

```bash
grep -q ".claude/settings.local.json" /home/hendrik/Documents/blokus-ai/.gitignore 2>/dev/null || \
echo ".claude/settings.local.json" >> /home/hendrik/Documents/blokus-ai/.gitignore
```

## Step 6: Report Configuration

```markdown
## MCP Data Sources Configured

### Playwright
- Status: Configured
- Browser: chromium
- Headless: true
- Usage: `browser_navigate`, `browser_click`, `browser_screenshot`, etc.

### Postgres
- Status: Configured
- Host: localhost
- Port: 5432
- Database: blokus_ai
- Usage: `postgres_query`, `postgres_execute`, etc.

### Files Created/Modified
- `.claude/settings.local.json` - MCP server configuration
- `.gitignore` - Added settings.local.json

### Next Steps
1. Restart Claude Code to load new MCP servers
2. Test with: "Take a screenshot of http://localhost:5174"
3. Test with: "Show me the tables in the database"
```

## Troubleshooting

**Postgres connection failed:**
```bash
# Verify database is running
docker exec blokus-ai-db pg_isready -U dev

# Check credentials in api/.env
cat /home/hendrik/Documents/blokus-ai/api/.env | grep -i postgres
```

**Playwright browser issues:**
```bash
# Install browsers
npx playwright install chromium
```
