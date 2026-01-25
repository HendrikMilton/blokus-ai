import { ReactElement } from 'react';
import { ColorSelectionPage } from './pages/color-selection/ColorSelectionPage';
import { CameraCapturePage } from './pages/camera-capture/CameraCapturePage';
import { BoardConfirmPage } from './pages/board-confirm/BoardConfirmPage';
import { ResultsPage } from './pages/results/ResultsPage';
import { AdminPage } from './pages/admin/AdminPage';
import { ImagesPage } from './pages/admin/images/ImagesPage';
import { ImageDetailPage } from './pages/admin/images/[id]/ImageDetailPage';
import { LabelingPage } from './pages/admin/images/label/LabelingPage';

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
        element: <AdminPage />,
        layout: true,
    },
    {
        path: '/admin/images',
        element: <ImagesPage />,
        layout: true,
    },
    {
        path: '/admin/images/label',
        element: <LabelingPage />,
        layout: true,
    },
    {
        path: '/admin/images/label/:id',
        element: <LabelingPage />,
        layout: true,
    },
    {
        path: '/admin/images/:id',
        element: <ImageDetailPage />,
        layout: true,
    },
];
