---
description: Run build, lint (with auto-fix), and tests for the blokus-ai project.
allowed-tools: [Bash, Read, Glob, Grep, Edit]
---

# Verify Command

Run verification checks (build, lint, typecheck, test) with auto-fix for the blokus-ai project.

## Project Structure

- **Package Manager:** Bun
- **Monorepo:** Yes (workspaces: shared, api, frontend)

| Package | Build | Lint | TypeCheck | Test |
|---------|-------|------|-----------|------|
| shared | `bun run build` | - | - | - |
| api | `bun run build` | `bun run lint` | `bunx tsc --noEmit` | `bun run test` |
| frontend | `bun run build` | `bun run lint` | `bunx tsc --noEmit` | - |

## Step 1: Install Dependencies

```bash
cd /home/hendrik/blokus-ai && bun install
```

## Step 2: Build Shared Package

Shared package must be built first (dependency for api and frontend):

```bash
cd /home/hendrik/blokus-ai/shared && bun run build
```

## Step 3: Build & Verify API

```bash
# Build
cd /home/hendrik/blokus-ai/api && bun run build

# TypeScript type check (catches errors not caught by build)
cd /home/hendrik/blokus-ai/api && bunx tsc --noEmit

# Lint with auto-fix
cd /home/hendrik/blokus-ai/api && bun run lint -- --fix

# Run tests
cd /home/hendrik/blokus-ai/api && bun run test
```

## Step 4: Build & Verify Frontend

```bash
# Build
cd /home/hendrik/blokus-ai/frontend && bun run build

# TypeScript type check (catches errors not caught by build)
cd /home/hendrik/blokus-ai/frontend && bunx tsc --noEmit

# Lint with auto-fix
cd /home/hendrik/blokus-ai/frontend && bun run lint -- --fix
```

## Step 5: Fix Errors Iteratively

If any command fails:

1. Read and understand the error
2. Fix the error in the source file
3. Re-run the failed command
4. Repeat until no errors remain

### Common Error Fixes

**TypeScript errors:**
- Unused variables: Remove or prefix with `_`
- Missing types: Add proper type annotations
- Import errors: Fix import paths

**ESLint errors:**
- Follow the suggested fix or rule documentation
- Auto-fix handles most formatting issues

**Test failures:**
- Read the test file to understand expected behavior
- Fix the implementation or update the test

## Step 6: Final Verification

Run all checks one more time to confirm everything passes:

```bash
# Shared
cd /home/hendrik/blokus-ai/shared && bun run build

# API
cd /home/hendrik/blokus-ai/api && bun run build && bunx tsc --noEmit && bun run lint && bun run test

# Frontend
cd /home/hendrik/blokus-ai/frontend && bun run build && bunx tsc --noEmit && bun run lint
```

## Step 7: Report Results

```
## Verification Complete

### Shared
- Build: [PASS/FAIL]

### API
- Build: [PASS/FAIL]
- TypeCheck: [PASS/FAIL]
- Lint: [PASS/FAIL] (X issues fixed)
- Tests: [PASS/FAIL] (X passed, Y failed)

### Frontend
- Build: [PASS/FAIL]
- TypeCheck: [PASS/FAIL]
- Lint: [PASS/FAIL] (X issues fixed)

### Summary
[All checks passed / X issues remaining]
```

## Success Criteria

All of the following must pass with no errors:
- Shared build exits with code 0
- API build, typecheck, lint, and tests exit with code 0
- Frontend build, typecheck, and lint exit with code 0
