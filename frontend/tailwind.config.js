/** @type {import('tailwindcss').Config} */
export default {
    content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Plus Jakarta Sans', 'sans-serif'],
            },
            colors: {
                background: 'rgb(var(--background) / <alpha-value>)',
                foreground: 'rgb(var(--foreground) / <alpha-value>)',
                surface: 'rgb(var(--surface) / <alpha-value>)',
                border: 'rgb(var(--border) / <alpha-value>)',
                input: 'rgb(var(--input) / <alpha-value>)',
                ring: 'rgb(var(--ring) / <alpha-value>)',
                primary: {
                    DEFAULT: 'rgb(var(--primary) / <alpha-value>)',
                    foreground: 'rgb(var(--primary-foreground) / <alpha-value>)',
                },
                secondary: {
                    DEFAULT: 'rgb(var(--secondary) / <alpha-value>)',
                    foreground: 'rgb(var(--secondary-foreground) / <alpha-value>)',
                },
                muted: {
                    DEFAULT: 'rgb(var(--muted) / <alpha-value>)',
                    foreground: 'rgb(var(--muted-foreground) / <alpha-value>)',
                },
                destructive: {
                    DEFAULT: 'rgb(var(--destructive) / <alpha-value>)',
                    foreground: 'rgb(var(--destructive-foreground) / <alpha-value>)',
                },
            },
            keyframes: {
                shimmer: {
                    '0%': { transform: 'translateX(-100%)' },
                    '100%': { transform: 'translateX(100%)' },
                },
            },
            animation: {
                shimmer: 'shimmer 1.8s ease-in-out infinite',
            },
        },
    },
    darkMode: 'class',
    plugins: [
        function ({ addBase }) {
            addBase({
                ':root': {
                    '--background': '255 255 255',
                    '--foreground': '15 23 42',
                    '--surface': '248 250 252',
                    '--border': '226 232 240',
                    '--input': '226 232 240',
                    '--ring': '59 130 246',
                    '--primary': '59 130 246',
                    '--primary-foreground': '255 255 255',
                    '--secondary': '241 245 249',
                    '--secondary-foreground': '15 23 42',
                    '--muted': '241 245 249',
                    '--muted-foreground': '100 116 139',
                    '--destructive': '239 68 68',
                    '--destructive-foreground': '255 255 255',
                },
                '.dark': {
                    '--background': '15 23 42',
                    '--foreground': '241 245 249',
                    '--surface': '30 41 59',
                    '--border': '51 65 85',
                    '--input': '51 65 85',
                    '--ring': '59 130 246',
                    '--primary': '59 130 246',
                    '--primary-foreground': '255 255 255',
                    '--secondary': '30 41 59',
                    '--secondary-foreground': '241 245 249',
                    '--muted': '30 41 59',
                    '--muted-foreground': '148 163 184',
                    '--destructive': '239 68 68',
                    '--destructive-foreground': '255 255 255',
                },
            });
        },
    ],
};
