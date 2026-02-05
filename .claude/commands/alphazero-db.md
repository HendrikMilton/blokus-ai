# AlphaZero Database Tables

Quick reference for querying AlphaZero training data in PostgreSQL.

## Connection

```bash
docker exec postgres17 psql -U c_connect_user -d blokus_ai -c "YOUR_QUERY"
```

## Tables Overview

### 1. `alphazero_runs` - Training Run Metadata
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| name | varchar(100) | Run name (e.g., "M NEW") |
| status | varchar(20) | pending/running/completed/failed/stopped |
| config | jsonb | Full training config |
| current_generation | int | Current generation |
| games_generated | int | Total games played |
| training_steps | int | Total training steps |
| policy_loss | real | Latest policy loss |
| value_loss | real | Latest value loss |
| best_win_rate | real | Best win rate achieved |
| started_at | timestamp | When started |
| completed_at | timestamp | When finished |
| error_message | text | Error if failed |

### 2. `alphazero_checkpoints` - Model Snapshots
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| run_id | uuid | FK to runs |
| generation | int | Generation number |
| step | int | Training step |
| policy_loss | real | Policy loss |
| value_loss | real | Value loss |
| eval_win_rate | real | Win rate vs previous gens (ELO) |
| eval_games | real | Eval games played |
| heuristic_win_rate | real | Win rate vs heuristic AI |
| is_best | boolean | Best model flag |
| model_path | varchar(255) | Path to .pt file |

### 3. `alphazero_metrics` - Per-Step Training Metrics
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| run_id | uuid | FK to runs |
| generation | int | Generation |
| step | int | Training step |
| policy_loss | real | Policy loss |
| value_loss | real | Value loss |
| total_loss | real | Combined loss |
| learning_rate | real | Current LR |
| grad_norm | real | Gradient norm |
| policy_entropy | real | Policy entropy |
| policy_top1 | real | Top-1 probability |
| policy_top5 | real | Top-5 probability |
| avg_corners | real | Avg corners |
| avg_mobility | real | Avg legal moves |
| center_coverage | real | Center coverage |
| games_per_sec | real | Self-play speed |
| moves_per_sec | real | Moves/sec |
| replay_buffer_size | int | Buffer size |
| step_time_ms | real | Step duration |

## Common Queries

### List all runs
```sql
SELECT id, name, status, current_generation, games_generated, best_win_rate
FROM alphazero_runs ORDER BY created_at DESC;
```

### Get running training status
```sql
SELECT name, current_generation, games_generated, training_steps, policy_loss, value_loss
FROM alphazero_runs WHERE status = 'running';
```

### List checkpoints for a run
```sql
SELECT generation, step, policy_loss, value_loss, eval_win_rate, heuristic_win_rate, is_best
FROM alphazero_checkpoints
WHERE run_id = 'RUN_ID' ORDER BY generation;
```

### Get metrics for a generation
```sql
SELECT step, policy_loss, value_loss, policy_entropy, grad_norm
FROM alphazero_metrics
WHERE run_id = 'RUN_ID' AND generation = 1
ORDER BY step;
```

### Compare heuristic win rates across generations
```sql
SELECT generation, heuristic_win_rate, policy_loss, value_loss
FROM alphazero_checkpoints
WHERE run_id = 'RUN_ID' ORDER BY generation;
```

### Get latest run's config
```sql
SELECT name, config FROM alphazero_runs ORDER BY created_at DESC LIMIT 1;
```

### Training progress over time
```sql
SELECT generation, AVG(policy_loss) as avg_policy, AVG(value_loss) as avg_value
FROM alphazero_metrics
WHERE run_id = 'RUN_ID'
GROUP BY generation ORDER BY generation;
```
