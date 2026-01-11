---
description: Test blokus-ai UI flows using Playwright MCP browser automation.
allowed-tools: [Bash, Read, Glob, Grep, Task, TodoWrite, TaskOutput]
argument-hint: "[commits:N] [viewport:desktop|mobile|both]"
---

# UI Flow Testing

Test blokus-ai UI flows using Playwright MCP browser automation.

**Arguments:** $ARGUMENTS
- `commits:N` - Number of commits to analyze (default: 5)
- `viewport:desktop|mobile|both` - Which viewport(s) to test (default: both)

## Step 0: Ensure Dev Servers Are Running

Check if the apps are already running:

```bash
curl -sf http://localhost:3000/ping/liveness && echo "API: Running" || echo "API: Not running"
curl -sf -o /dev/null http://localhost:5174 && echo "Frontend: Running" || echo "Frontend: Not running"
```

If not running, start them:

```bash
# Kill any existing processes
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:5174 | xargs kill -9 2>/dev/null || true

# Start apps (run in background)
cd /home/hendrik/Documents/blokus-ai && bun run start
```

Wait for startup:
```bash
sleep 5
curl -sf http://localhost:5174 && echo "Frontend ready"
```

## Step 1: Analyze Changes

Get changed frontend files:

```bash
git diff HEAD~$COMMITS --name-only | grep -E '\.(tsx|ts)$' | grep -v '\.test\.' | grep 'frontend/src'
```

**Read each changed file** to understand:
- What UI components/pages were modified
- What user interactions are involved
- What API calls are made
- What validation logic exists

## Step 2: Create Test Plan

Use TodoWrite to create a test checklist for blokus-ai flows:

Example flows to test:
- Camera capture flow - pending
- Board confirmation flow - pending
- Color selection flow - pending
- Results display - pending

For each flow, document:
1. **Flow name**: e.g., "Camera Capture"
2. **Steps**: Exact sequence of actions
3. **Verification**: How to confirm it worked

## Step 3: Run Browser Tests

Use Playwright MCP tools to test each identified flow.

### Viewport Testing

**Desktop viewport** (default ~1280x720):
- No resize needed

**Mobile viewport** (390x844):
```
browser_resize: { width: 390, height: 844 }
```

### Visual Inspection

After each navigation, **take a screenshot** with `browser_take_screenshot` (fullPage: true).

**Check for:**

**Layout Issues:**
- Elements overlapping or cut off
- Inconsistent spacing
- Content not properly aligned

**Mobile-Specific:**
- Buttons too small to tap (min 44x44px)
- Horizontal scrolling
- Elements not stacking properly

### Bug Discovery Protocol

**If you discover a bug during testing:**
1. STOP testing immediately
2. Fix the bug in the source code
3. Close the browser with `browser_close`
4. Restart testing from Step 0

### Common Test Patterns

**Navigation Flow:**
1. Navigate to http://localhost:5174
2. Verify page loads correctly
3. Test navigation between pages
4. Press Escape to dismiss any dialogs

**Camera Flow:**
1. Navigate to camera capture page
2. Test camera permissions handling
3. Test capture button
4. Verify preview works

**Board Confirmation:**
1. After capture, verify board preview
2. Test confirm/retry buttons
3. Verify navigation to next step

## Step 4: Report Results

```markdown
## Test Results

### Desktop Viewport (1280x720)

#### Passed
- [Flow name] - [what was tested]

#### Failed
- [Flow name] - [what failed, error message]

#### Visual Issues
- [Page/component] - [description]

### Mobile Viewport (390x844)

#### Passed
- [Flow name] - [what was tested]

#### Failed
- [Flow name] - [what failed]

#### Visual Issues
- [Page/component] - [description]

### Summary
- Viewports: Desktop [x] Mobile [x]
- Flows tested: [x]
- Issues found: [x]
```

## Notes

- App must be running at http://localhost:5174
- Playwright MCP must be configured
- Use `browser_console_messages` to check for JS errors
- Take screenshots for documentation
- Fix bugs immediately when found, then restart testing
