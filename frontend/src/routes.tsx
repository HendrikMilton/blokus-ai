import { ReactElement } from 'react';
import { Navigate } from 'react-router-dom';
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
        element: <Navigate to="/admin" replace />,
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
