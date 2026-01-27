import { ButtonHTMLAttributes, ReactNode } from 'react';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'destructive' | 'success' | 'danger';
    size?: 'xs' | 'sm' | 'md' | 'lg';
    fullWidth?: boolean;
    loading?: boolean;
    children: ReactNode;
    leftIcon?: ReactNode;
    rightIcon?: ReactNode;
}

const variants = {
    primary: 'bg-primary text-white hover:bg-primary/90 disabled:bg-muted disabled:text-white/60 disabled:opacity-60',
    secondary: 'bg-surface-dim text-foreground hover:bg-border/50 disabled:bg-surface-dim disabled:text-muted disabled:opacity-60',
    ghost: 'hover:bg-surface-dim text-foreground disabled:text-muted disabled:opacity-50',
    destructive: 'bg-red-600 dark:bg-red-500 text-white hover:bg-red-700 dark:hover:bg-red-600 disabled:bg-muted disabled:text-white/60 disabled:opacity-60',
    success: 'bg-success text-white hover:bg-success/90 disabled:bg-muted disabled:text-white/60 disabled:opacity-60',
    danger: 'bg-red-600 dark:bg-red-500 text-white hover:bg-red-700 dark:hover:bg-red-600 disabled:bg-muted disabled:text-white/60 disabled:opacity-60',
};

const sizes = {
    xs: 'px-2 py-1 text-xs',
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
};

export const Button = ({
    variant = 'primary',
    size = 'md',
    fullWidth = false,
    loading = false,
    children,
    leftIcon,
    rightIcon,
    className = '',
    disabled,
    ...props
}: ButtonProps) => {
    return (
        <button
            className={`
                inline-flex items-center justify-center gap-2 rounded-md font-medium
                transition-colors focus-visible:outline-none focus-visible:ring-2
                focus-visible:ring-primary/30 disabled:pointer-events-none
                ${variants[variant]}
                ${sizes[size]}
                ${fullWidth ? 'w-full' : ''}
                ${className}
            `}
            disabled={disabled || loading}
            {...props}
        >
            {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
                leftIcon
            )}
            {children}
            {!loading && rightIcon}
        </button>
    );
};
