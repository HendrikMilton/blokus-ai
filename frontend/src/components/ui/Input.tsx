import { InputHTMLAttributes, ReactNode, forwardRef } from 'react';

interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'className'> {
    label?: string;
    error?: string;
    touched?: boolean;
    icon?: ReactNode;
    rightIcon?: ReactNode;
    /** Container className - use for width control */
    className?: string;
    /** Input element className */
    inputClassName?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
    ({ label, error, touched, icon, rightIcon, className, inputClassName = '', ...props }, ref) => {
        const showError = touched && error;

        return (
            <div className={className || 'w-full'}>
                {label && (
                    <label className="block text-sm font-medium text-foreground mb-1">
                        {label}
                    </label>
                )}
                <div className="relative">
                    {icon && (
                        <div className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted">
                            {icon}
                        </div>
                    )}
                    <input
                        ref={ref}
                        className={`
                            w-full h-9 rounded-md border bg-surface px-3 text-sm text-foreground
                            placeholder:text-muted
                            focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary
                            disabled:cursor-not-allowed disabled:opacity-50
                            ${icon ? 'pl-8' : ''}
                            ${rightIcon ? 'pr-8' : ''}
                            ${showError ? 'border-error' : 'border-border'}
                            ${inputClassName}
                        `}
                        {...props}
                    />
                    {rightIcon && (
                        <div className="absolute right-2.5 top-1/2 -translate-y-1/2">
                            {rightIcon}
                        </div>
                    )}
                </div>
                {showError && (
                    <p className="mt-1 text-xs text-error">{error}</p>
                )}
            </div>
        );
    }
);

Input.displayName = 'Input';
