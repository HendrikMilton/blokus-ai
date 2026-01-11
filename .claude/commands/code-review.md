---
description: Review uncommitted and unpushed blokus-ai code for clean code principles, then fix issues if requested.
allowed-tools: [Bash, Read, Grep, Glob]
---

# Code Review Command

Review uncommitted and committed-but-not-pushed code for clean code principles.

**IMPORTANT**: This command runs in TWO phases:
1. **Review Phase** - Analyze code and generate review report (read-only)
2. **Fix Phase** - Create implementation plan to fix issues (after user says "fix")

## Phase 1: Code Review

### Step 1: Identify Changed Files

```bash
# Uncommitted changes (staged and unstaged)
git diff --name-only
git diff --cached --name-only

# Committed but not pushed
git log origin/main..HEAD --name-only --pretty=format:"" 2>/dev/null
```

Get diff content:

```bash
git diff
git diff --cached
git diff origin/main..HEAD 2>/dev/null
```

### Step 2: Project-Specific Patterns

The blokus-ai project uses:

**Frontend (React + TypeScript + Tailwind):**
- Components in `frontend/src/components/`
- Pages in `frontend/src/pages/`
- Hooks in `frontend/src/hooks/`
- Context in `frontend/src/contexts/`
- Utilities in `frontend/src/utils/`

**API (Express + TypeScript + Typia):**
- Features in `api/src/features/`
- Middleware in `api/src/middleware/`
- Use Typia `createValidateEquals` for request validation

**Shared:**
- Types in `shared/src/`

### Step 3: Clean Code Review

Review all changed code for:

#### DRY (Don't Repeat Yourself)
- Duplicated code blocks (3+ lines repeated)
- Similar logic that could be extracted

#### Single Responsibility
- Functions should do one thing
- Components should have clear purpose

#### Naming
- Descriptive variable/function names
- Boolean variables: `is_`, `has_`, `can_`, `should_`

#### Function Length & Complexity
- Functions over 50 lines should be flagged
- Deeply nested code (4+ levels) should be refactored

#### Error Handling
- No empty catch blocks
- Errors should be logged with context

#### Type Safety
- No `any` types without justification
- Proper TypeScript interfaces for all data structures

### Step 4: Security Review

1. **No Secrets**: No hardcoded passwords, API keys, tokens
2. **Input Validation**: Use Typia for API request validation
3. **XSS**: Sanitize user input in frontend
4. **Logging**: Don't log sensitive data

### Step 5: Generate Review Report

```markdown
# Code Review Report

**Date**: [current date]
**Files Changed**: X files
**Lines Changed**: +Y / -Z

## Critical Issues
- **[file:line]** Description

## Warnings
- **[file:line]** Description

## Suggestions
- **[file:line]** Description

## Checklist

### Clean Code
- [ ] No DRY violations
- [ ] Functions are focused
- [ ] Good naming conventions
- [ ] Proper error handling

### Security
- [ ] No hardcoded secrets
- [ ] Input properly validated
```

### Step 6: Prompt for Fix

```
---
CODE REVIEW COMPLETE

To fix the issues, respond with "fix" and I will:
1. Create a detailed implementation plan for each fix
2. Ask for your approval before making changes
3. Apply the fixes
---
```

Then STOP and wait for user response.

---

## Phase 2: Fix Implementation (After User Says "fix")

### Step 1: Prioritize Issues

1. **Critical** - Security issues, broken functionality
2. **Warnings** - DRY violations, type issues
3. **Suggestions** - Naming improvements, minor refactoring

### Step 2: Create Implementation Plan

For each issue:
- **File**: Path to file
- **Line**: Line number(s)
- **Issue**: What's wrong
- **Fix**: Exact changes to make

### Step 3: Apply Fixes

After user approves:
1. Apply fixes one at a time
2. Run `bun run lint -- --fix` after each fix
3. Report progress

### Step 4: Verify

Run `/verify` to ensure all changes pass build, lint, and tests.
