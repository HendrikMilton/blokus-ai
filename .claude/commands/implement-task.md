---
description: Implement a feature or fix for blokus-ai. Plans, implements, then runs verify and code-review.
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep, Task, TodoWrite, AskUserQuestion, EnterPlanMode]
argument-hint: "<task description>"
---

# Implement Task

**Input:** $ARGUMENTS

## Phase 1: Understand & Refine the Task

### Parse the Input

1. **Identify task type**: New feature, enhancement, or bug fix?
2. **Clarify scope**: What parts of the codebase will be affected?
3. **Extract requirements**: What specifically needs to be done?

### Explore the Codebase

```bash
# Frontend files
find /home/hendrik/Documents/blokus-ai/frontend/src -name "*.tsx" -o -name "*.ts" | head -20

# API files
find /home/hendrik/Documents/blokus-ai/api/src -name "*.ts" | head -20

# Shared types
ls /home/hendrik/Documents/blokus-ai/shared/src/
```

Read existing code to understand:
- Current patterns and conventions
- Related functionality
- Where new code should go

### Present Refined Understanding

```markdown
## Task Understanding

**Type:** [Feature / Enhancement / Bug Fix]
**Summary:** [1-2 sentence refined description]
**Affected areas:** [frontend, api, shared]

**Key Requirements:**
- [requirement 1]
- [requirement 2]

**Questions/Assumptions:**
- [any clarifications needed]
```

## Phase 2: Clarify Requirements

If the task description is ambiguous, use `AskUserQuestion` to clarify.

**Do NOT proceed with implementation if key requirements are unclear.**

## Phase 3: Plan the Implementation

Use `EnterPlanMode` to design the implementation approach.

### Plan Should Include

```markdown
## Implementation Plan

### Files to Create
- `frontend/src/components/NewComponent.tsx` - Description

### Files to Modify
- `frontend/src/App.tsx` - Add route
- `api/src/routes/endpoint.ts` - Add handler

### Shared Types (if any)
- `shared/src/types.ts` - Add new interfaces

### Steps
1. [First step]
2. [Second step]
```

## Phase 4: Implement

After plan is approved, implement the changes.

### Implementation Guidelines

**General:**
- Follow existing code patterns
- Keep changes minimal and focused
- Handle errors appropriately
- Add TypeScript types

**Frontend (React + Tailwind):**
- Reuse existing components from `frontend/src/components/`
- Follow Tailwind styling patterns
- Add proper loading/error states

**API (Express + TypeScript):**
- Follow existing router patterns in `api/src/features/`
- Use Typia for request validation
- Handle errors with appropriate status codes

**Shared:**
- Add shared types to `shared/src/`
- Rebuild shared after changes: `cd shared && bun run build`

**Database Migrations:**
- Create new migration: `bun run --cwd api migrate:make <migration_name>`
- Migrations run automatically on API startup - no need to run migrate command manually
- Migration files go in `api/src/migrations/`

### Track Progress

Use `TodoWrite` to track implementation:

```
- Create new component - in_progress
- Add API endpoint - pending
- Update routing - pending
```

## Phase 5: Quality Assurance

### Run /verify

```
/verify
```

This runs:
1. `bun install`
2. `cd shared && bun run build`
3. `cd api && bun run build && bun run lint -- --fix && bun run test`
4. `cd frontend && bun run build && bun run lint -- --fix`

Fix any errors until all checks pass.

### Run /code-review (for significant changes)

For changes that touch 5+ files or add new features:

```
/code-review
```

If issues found, fix them and re-run verify.

### Run /test-ui (if UI changed)

For frontend changes:

```
/test-ui
```

## Phase 6: Summary

After all quality checks pass:

```markdown
## Implementation Complete

### Changes Made
- [file1.tsx] - Added new component
- [file2.ts] - Added API endpoint

### Quality Checks
- Build: PASS
- Lint: PASS
- Tests: PASS (X tests)
- Code Review: PASS

### Next Steps (if any)
- [Any follow-up tasks]
```

## Error Recovery

If any phase fails:

1. **Build fails:** Fix syntax/type errors
2. **Tests fail:** Fix failing tests or implementation
3. **Lint fails:** Run `bun run lint -- --fix` to auto-fix

**Always re-run /verify after fixes.**
