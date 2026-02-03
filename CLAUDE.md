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

## Database

PostgreSQL runs in Docker container `postgres17`.

```bash
# Connect to database
docker exec postgres17 psql -U c_connect_user -d blokus_ai

# Run query
docker exec postgres17 psql -U c_connect_user -d blokus_ai -c "SELECT * FROM users;"

# List tables
docker exec postgres17 psql -U c_connect_user -d blokus_ai -c "\dt"

# Describe table
docker exec postgres17 psql -U c_connect_user -d blokus_ai -c "\d table_name"
```

Migrations are in `api/src/migrations/`. Migrations run automatically on API startup - no need to apply manually.

```bash
# Create new migration (ALWAYS use this command, not manual file creation)
bun run --cwd api migrate:make <migration_name>

# This uses scripts/create-migration.js which:
# - Generates proper timestamp prefix
# - Creates empty migration file
# - Adds file to git
# Then you edit the generated file to add the up/down functions
```

## Notes

- Glob/search from frontend folder won't find files in other packages. Use `ls` or search from project root `/home/hendrik/blokus-ai/`.


Implement the actual training data flow for the Blokus AI model training feature.
Currently the training is a stub that returns mock metrics.

### Current State (already implemented)
- Frontend "Train Model" button calls POST /api/ml/train
- API creates a training_run record in PostgreSQL
- API calls ML service POST /train with only { run_id }
- ML service returns random mock metrics (no actual training)
- Database `board_detections` table has verified samples with bounding boxes (bbox_x,
  bbox_y, bbox_width, bbox_height, verified_at)
- Images stored in S3 (CloudFront URL pattern:
  https://d26azxov2byjpl.cloudfront.net/thumbnails/{sample_id}.jpg)

### What Needs to Be Implemented

1. **API: Create endpoint to get training data**
    - Add `GET /api/ml/training-data` endpoint in `api/src/features/ml/router.ts`
    - Query `board_detections` WHERE `verified_at IS NOT NULL`
    - Return array of: `{ sample_id, image_url, bbox: { x, y, width, height } }`

2. **ML Service: Implement real YOLO training**
    - Update `ml-service/app/features/train/__init__.py`
    - Fetch training data from API endpoint
    - Download images from CloudFront URLs
    - Convert bounding boxes to YOLO format (center_x, center_y, width, height - all
      0-1 normalized)
    - Create YOLO dataset structure (images/ and labels/ folders)
    - Train YOLOv8 model using ultralytics library
    - Save model weights to local path: `ml-service/models/{run_id}/best.pt`
    - Return actual training metrics

3. **Update training run record**
    - Store the model path in training_runs table (may need migration to add
      `model_path` column)
    - Or store model filename convention based on run_id

4. **Update detection to use trained models**
    - The detection endpoint POST /api/ml/detect/:id currently receives `{ model_id }`
    - ML service should load the model from `ml-service/models/{model_id}/best.pt`
    - Update `ml-service/app/features/detect/__init__.py` to accept model_id and load
      correct weights

### Key Files to Modify
- `api/src/features/ml/router.ts` - Add training data endpoint
- `api/src/features/ml/database.ts` - Add function to get verified samples with labels
- `ml-service/app/features/train/__init__.py` - Implement real training
- `ml-service/app/features/detect/__init__.py` - Load model by ID
- Possibly add migration for model_path column

### YOLO Training Notes
- Use ultralytics library (already likely installed)
- Single class: "board" (class_id = 0)
- YOLO label format per line: `class_id center_x center_y width height`
- Center coordinates: center_x = bbox_x + bbox_width/2, center_y = bbox_y +
  bbox_height/2
- All values normalized 0-1 (which they already are in the database)

### Database Schema Reference
```sql
-- board_detections table
sample_id VARCHAR(36)
bbox_x FLOAT  -- 0-1 normalized
bbox_y FLOAT  -- 0-1 normalized
bbox_width FLOAT  -- 0-1 normalized
bbox_height FLOAT  -- 0-1 normalized
verified_at TIMESTAMP  -- NOT NULL means verified/ready for training

-- training_runs table
id UUID
status VARCHAR(20)  -- pending, running, completed, failed
image_count INTEGER
accuracy FLOAT
loss FLOAT
-- may need to add: model_path VARCHAR

Commands

- Build shared: bun run --cwd shared build
- Build API: bun run --cwd api build
- Build frontend: bun run --cwd frontend build
- Run all: bun run start
- API port: 3009
- ML service port: 8017
