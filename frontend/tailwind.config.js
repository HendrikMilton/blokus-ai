/** @type {import('tailwindcss').Config} */
export default {
    content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Google Sans', 'Roboto', 'system-ui', 'sans-serif'],
                mono: ['Roboto Mono', 'monospace'],
            },
            colors: {
                background: 'rgb(var(--background) / <alpha-value>)',
                foreground: 'rgb(var(--foreground) / <alpha-value>)',
                surface: 'rgb(var(--surface) / <alpha-value>)',
                'surface-dim': 'rgb(var(--surface-dim) / <alpha-value>)',
                border: 'rgb(var(--border) / <alpha-value>)',
                primary: 'rgb(var(--primary) / <alpha-value>)',
                secondary: 'rgb(var(--secondary) / <alpha-value>)',
                muted: 'rgb(var(--muted) / <alpha-value>)',
                error: 'rgb(var(--error) / <alpha-value>)',
                success: 'rgb(var(--success) / <alpha-value>)',
                warning: 'rgb(var(--warning) / <alpha-value>)',
                info: 'rgb(var(--info) / <alpha-value>)',
            },
            boxShadow: {
                'gcp': '0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15)',
                'gcp-lg': '0 1px 2px 0 rgba(60,64,67,0.3), 0 2px 6px 2px rgba(60,64,67,0.15)',
                'gcp-hover': '0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15)',
            },
            keyframes: {
                'fade-slide-in': {
                    '0%': { opacity: '0', transform: 'translateY(-8px)', backgroundColor: 'rgba(26, 115, 232, 0.08)' },
                    '100%': { opacity: '1', transform: 'translateY(0)', backgroundColor: 'transparent' },
                },
                'pulse-subtle': {
                    '0%, 100%': { opacity: '1' },
                    '50%': { opacity: '0.7' },
                },
                shimmer: {
                    '0%': { transform: 'translateX(-100%)' },
                    '100%': { transform: 'translateX(100%)' },
                },
            },
            animation: {
                'fade-slide-in': 'fade-slide-in 0.4s ease-out forwards',
                'pulse-subtle': 'pulse-subtle 2s ease-in-out infinite',
                shimmer: 'shimmer 1.8s ease-in-out infinite',
            },
        },
    },
    darkMode: 'class',
    plugins: [
        function({ addBase }) {
            addBase({
                ':root': {
                    // GCP Light Theme - Clean whites with blue accents
                    '--background': '248 249 250',      // #f8f9fa - GCP background gray
                    '--foreground': '32 33 36',         // #202124 - GCP text
                    '--surface': '255 255 255',         // #ffffff
                    '--surface-dim': '241 243 244',     // #f1f3f4 - Subtle gray
                    '--border': '218 220 224',          // #dadce0 - GCP border
                    '--primary': '26 115 232',          // #1a73e8 - Google Blue
                    '--secondary': '117 117 117',       // #757575 - Lighter secondary text
                    '--muted': '158 158 158',           // #9e9e9e - Lighter muted
                    '--error': '217 48 37',             // #d93025 - Google Red
                    '--success': '24 128 56',           // #188038 - Google Green
                    '--warning': '249 171 0',           // #f9ab00 - Google Yellow
                    '--info': '142 68 173',             // #8e44ad - Purple
                },
                '.dark': {
                    '--background': '32 33 36',         // #202124
                    '--foreground': '232 234 237',      // #e8eaed
                    '--surface': '41 42 45',            // #292a2d
                    '--surface-dim': '53 54 58',        // #35363a
                    '--border': '95 99 104',            // #5f6368
                    '--primary': '66 133 244',          // #4285f4 - Google Blue (better contrast)
                    '--secondary': '154 160 166',       // #9aa0a6
                    '--muted': '154 160 166',           // #9aa0a6
                    '--error': '242 139 130',           // #f28b82
                    '--success': '129 201 149',         // #81c995
                    '--warning': '253 214 99',          // #fdd663
                    '--info': '187 143 206',            // #bb8fce - Light purple
                },
            });
        },
    ],
};
