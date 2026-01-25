import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
    plugins: [
        react(),
        VitePWA({
            registerType: 'prompt',
            includeAssets: ['favicon.svg', 'apple-touch-icon.png', 'icon-192.png', 'icon-512.png'],
            manifest: {
                name: 'blokus-ai',
                short_name: 'blokus-ai',
                description: 'AI-powered Blokus game assistant',
                theme_color: '#3b82f6',
                background_color: '#ffffff',
                display: 'standalone',
                orientation: 'portrait-primary',
                start_url: '/',
                icons: [
                    {
                        src: '/icon-192.png',
                        sizes: '192x192',
                        type: 'image/png',
                    },
                    {
                        src: '/icon-512.png',
                        sizes: '512x512',
                        type: 'image/png',
                        purpose: 'any',
                    },
                    {
                        src: '/apple-touch-icon.png',
                        sizes: '180x180',
                        type: 'image/png',
                        purpose: 'apple touch icon',
                    },
                ],
            },
            workbox: {
                globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
                cleanupOutdatedCaches: true,
                runtimeCaching: [
                    {
                        urlPattern: /^https?:\/\/.*\/api\/.*/i,
                        handler: 'NetworkFirst',
                        options: {
                            cacheName: 'api-cache',
                            expiration: {
                                maxEntries: 50,
                                maxAgeSeconds: 60 * 2,
                            },
                            networkTimeoutSeconds: 5,
                        },
                    },
                    {
                        urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp)$/,
                        handler: 'CacheFirst',
                        options: {
                            cacheName: 'images-cache',
                            expiration: {
                                maxEntries: 60,
                                maxAgeSeconds: 60 * 60 * 24 * 7,
                            },
                        },
                    },
                ],
                navigateFallback: '/index.html',
                skipWaiting: true,
                clientsClaim: true,
            },
            devOptions: {
                enabled: false,
            },
        }),
    ],
    server: {
        host: '0.0.0.0',
        port: 5177,
        hmr: true,
        watch: {
            usePolling: true,
            interval: 100,
        },
        allowedHosts: ['.trycloudflare.com', '.ngrok-free.app', '.ngrok.io', '.loca.lt', '.hycapital.io'],
        // Proxy only needed for local dev - tunnel ingress handles /api and /socket.io directly
        proxy: process.env.TUNNEL_MODE
            ? {}
            : {
                  '/api': {
                      target: 'http://127.0.0.1:3009',
                      changeOrigin: true,
                      secure: false,
                  },
                  '/ping': {
                      target: 'http://127.0.0.1:3009',
                      changeOrigin: true,
                      secure: false,
                  },
                  '/socket.io': {
                      target: 'http://127.0.0.1:3009',
                      changeOrigin: true,
                      secure: false,
                      ws: true,
                  },
              },
    },
    build: {
        chunkSizeWarningLimit: 1000,
    },
});
