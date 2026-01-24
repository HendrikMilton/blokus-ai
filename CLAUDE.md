# Blokus AI

Blokus board game assistant with AI-powered move recommendations.

## Stack

- **Runtime**: Bun (not npm/yarn)
- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **API**: Express + TypeScript + Knex (PostgreSQL)
- **ML Service**: Python
- **Shared**: TypeScript types/utils shared across packages

## Project Structure

```
blokus-ai/
├── frontend/        # React PWA
│   └── src/
│       ├── pages/           # Page components (one folder per page)
│       ├── components/      # Reusable components
│       ├── contexts/        # React contexts
│       ├── hooks/           # Custom hooks
│       ├── api/             # API client
│       └── types/           # TypeScript types
├── api/             # Express backend
├── shared/          # Shared types/utilities
├── ml-service/      # Python ML service
└── infrastructure/  # Deployment configs
```

## Commands

```bash
# Install dependencies
bun install

# Run all services (frontend + api)
bun run start

# Build all
bun run build

# Frontend only
bun run --cwd frontend dev
bun run --cwd frontend build
bun run --cwd frontend lint

# API only
bun run --cwd api dev
bun run --cwd api build
bun run --cwd api test
bun run --cwd api lint
bun run --cwd api migrate

# Shared (build before other packages)
bun run --cwd shared build
```

## Conventions

- Pages go in `frontend/src/pages/<page-name>/<PageName>.tsx`
- Export pages from `frontend/src/pages/index.ts`
- Use `@blokus-ai/shared` for shared types between frontend and api
