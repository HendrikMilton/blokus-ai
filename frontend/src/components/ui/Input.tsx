import { InputHTMLAttributes, ReactNode, forwardRef } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    touched?: boolean;
    icon?: ReactNode;
    rightIcon?: ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
    ({ label, error, touched, icon, rightIcon, className = '', ...props }, ref) => {
        const showError = touched && error;

        return (
            <div className="w-full">
                {label && (
                    <label className="block text-sm font-medium text-foreground mb-1">
                        {label}
                    </label>
                )}
                <div className="relative">
                    {icon && (
                        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
                            {icon}
                        </div>
                    )}
                    <input
                        ref={ref}
                        className={`
                            w-full rounded-md border bg-background px-3 py-2 text-sm
                            placeholder:text-muted-foreground
                            focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent
                            disabled:cursor-not-allowed disabled:opacity-50
                            ${icon ? 'pl-10' : ''}
                            ${rightIcon ? 'pr-10' : ''}
                            ${showError ? 'border-destructive' : 'border-input'}
                            ${className}
                        `}
                        {...props}
                    />
                    {rightIcon && (
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground">
                            {rightIcon}
                        </div>
                    )}
                </div>
                {showError && (
                    <p className="mt-1 text-xs text-destructive">{error}</p>
                )}
            </div>
        );
    }
);

Input.displayName = 'Input';
