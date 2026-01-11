---
name: frontend-design
description: Create frontend pages, components, and UI for blokus-ai following project patterns.
---

# Frontend Design Skill

Create production-ready frontend pages and components for blokus-ai.

## Project Stack

- **Framework:** React 18
- **Build Tool:** Vite
- **Styling:** Tailwind CSS
- **State:** React Context (GameContext)
- **Routing:** React Router DOM
- **Notifications:** react-hot-toast
- **Real-time:** socket.io-client

## Project Structure

```
frontend/src/
├── components/     # Reusable UI components
│   ├── BoardOverlay.tsx
│   ├── ColorButton.tsx
│   ├── PieceIcon.tsx
│   └── StatsCard.tsx
├── contexts/       # React Context providers
│   └── GameContext.tsx
├── hooks/          # Custom React hooks
│   ├── useCamera.ts
│   └── index.ts
├── pages/          # Route pages
│   ├── BoardConfirmPage.tsx
│   ├── CameraCapturePage.tsx
│   ├── ColorSelectionPage.tsx
│   ├── ResultsPage.tsx
│   └── index.ts
├── types/          # TypeScript type definitions
├── utils/          # Utility functions
├── api/            # API client functions
│   └── index.ts
└── App.tsx         # Main app with routing
```

## Step 1: Understand Existing Patterns

Before creating new components, read existing files:

```bash
# Read existing components
cat /home/hendrik/Documents/blokus-ai/frontend/src/components/*.tsx

# Read existing pages
cat /home/hendrik/Documents/blokus-ai/frontend/src/pages/*.tsx

# Read existing hooks
cat /home/hendrik/Documents/blokus-ai/frontend/src/hooks/*.ts

# Read App.tsx for routing
cat /home/hendrik/Documents/blokus-ai/frontend/src/App.tsx
```

## Step 2: Component Guidelines

### Page Component Pattern

```tsx
import { useNavigate } from 'react-router-dom';
import { useGame } from '../contexts/GameContext';

export function MyPage() {
  const navigate = useNavigate();
  const { gameState, updateGame } = useGame();

  const handleAction = () => {
    updateGame({ /* ... */ });
    navigate('/next-page');
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">Page Title</h1>
      </header>

      <main>
        {/* Content */}
      </main>

      <footer className="fixed bottom-0 left-0 right-0 p-4">
        <button
          onClick={handleAction}
          className="w-full py-3 bg-blue-600 rounded-lg font-medium"
        >
          Continue
        </button>
      </footer>
    </div>
  );
}
```

### Reusable Component Pattern

```tsx
interface MyComponentProps {
  title: string;
  isActive?: boolean;
  onClick?: () => void;
}

export function MyComponent({ title, isActive = false, onClick }: MyComponentProps) {
  return (
    <div
      onClick={onClick}
      className={`p-4 rounded-lg cursor-pointer ${
        isActive ? 'bg-blue-600' : 'bg-gray-800'
      }`}
    >
      {title}
    </div>
  );
}
```

## Step 3: Tailwind Styling

### Color Palette (from tailwind.config.js)

- Background: `bg-gray-900`, `bg-gray-800`
- Text: `text-white`, `text-gray-400`
- Primary: `bg-blue-600`, `text-blue-500`
- Success: `bg-green-600`
- Error: `bg-red-600`

### Common Patterns

```tsx
// Full-screen page
<div className="min-h-screen bg-gray-900 text-white p-4">

// Card
<div className="bg-gray-800 rounded-lg p-4">

// Button (primary)
<button className="w-full py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium">

// Button (secondary)
<button className="w-full py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium">

// Fixed bottom action bar
<div className="fixed bottom-0 left-0 right-0 p-4 bg-gray-900">
```

## Step 4: Routing

Add new routes in `App.tsx`:

```tsx
import { MyPage } from './pages/MyPage';

// In Routes component
<Route path="/my-page" element={<MyPage />} />
```

## Step 5: API Calls

Use the API client in `api/index.ts`:

```tsx
import { api } from '../api';

const handleSubmit = async () => {
  try {
    const result = await api.post('/endpoint', data);
    toast.success('Success!');
  } catch (error) {
    toast.error('Failed to submit');
  }
};
```

## Step 6: Hooks

Create custom hooks in `hooks/`:

```tsx
// hooks/useMyHook.ts
import { useState, useCallback } from 'react';

export function useMyHook() {
  const [state, setState] = useState<string | null>(null);

  const doSomething = useCallback(() => {
    // Logic here
  }, []);

  return { state, doSomething };
}

// Export from hooks/index.ts
export { useMyHook } from './useMyHook';
```

## Step 7: Context

Use GameContext for shared state:

```tsx
import { useGame } from '../contexts/GameContext';

function MyComponent() {
  const { gameState, updateGame, resetGame } = useGame();

  // Access state
  const { selectedColor, boardImage } = gameState;

  // Update state
  updateGame({ selectedColor: 'blue' });
}
```

## Checklist

Before completing:

- [ ] Followed existing component patterns
- [ ] Used TypeScript interfaces for props
- [ ] Applied Tailwind classes consistently
- [ ] Handled loading states
- [ ] Handled error states
- [ ] Added route if new page
- [ ] Exported from index.ts files
- [ ] Mobile-responsive (min-h-screen, full-width buttons)
