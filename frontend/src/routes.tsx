import { ReactElement } from 'react';
import { ColorSelectionPage } from './pages/color-selection/ColorSelectionPage';
import { CameraCapturePage } from './pages/camera-capture/CameraCapturePage';
import { BoardConfirmPage } from './pages/board-confirm/BoardConfirmPage';
import { ResultsPage } from './pages/results/ResultsPage';
import { AdminPage } from './pages/admin/AdminPage';
import { ImagesPage } from './pages/admin/images/ImagesPage';
import { ImageDetailPage } from './pages/admin/images/[id]/ImageDetailPage';
import { LabelingPage } from './pages/admin/images/label/LabelingPage';
import { AdminProvider } from './contexts/AdminContext';

// Wrapper for admin pages that provides AdminContext
const withAdmin = (element: ReactElement) => (
    <AdminProvider>{element}</AdminProvider>
);

export interface RouteConfig {
    path: string;
    element: ReactElement;
    layout?: boolean;
}

export const routes: RouteConfig[] = [
    {
        path: '/',
        element: <ColorSelectionPage />,
        layout: true,
    },
    {
        path: '/capture',
        element: <CameraCapturePage />,
        layout: false,
    },
    {
        path: '/confirm',
        element: <BoardConfirmPage />,
        layout: false,
    },
    {
        path: '/results',
        element: <ResultsPage />,
        layout: false,
    },
    {
        path: '/admin',
        element: withAdmin(<AdminPage />),
        layout: true,
    },
    {
        path: '/admin/images',
        element: withAdmin(<ImagesPage />),
        layout: true,
    },
    {
        path: '/admin/images/label',
        element: withAdmin(<LabelingPage />),
        layout: true,
    },
    {
        path: '/admin/images/label/:id',
        element: withAdmin(<LabelingPage />),
        layout: true,
    },
    {
        path: '/admin/images/:id',
        element: withAdmin(<ImageDetailPage />),
        layout: true,
    },
];
