FROM oven/bun:1 AS builder

WORKDIR /app

# Copy workspace files
COPY package.json bun.lockb ./
COPY shared/package.json ./shared/
COPY api/package.json ./api/

# Install dependencies
RUN bun install --frozen-lockfile

# Copy source
COPY shared/ ./shared/
COPY api/ ./api/

# Build shared first, then API
WORKDIR /app/shared
RUN bun run build

WORKDIR /app/api
RUN bun run build

# Production image
FROM oven/bun:1-slim

WORKDIR /app

COPY --from=builder /app/api/dist ./dist
COPY --from=builder /app/api/package.json ./

ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

CMD ["bun", "dist/index.js"]
