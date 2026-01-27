import { AlertTriangle } from 'lucide-react';
import { Modal } from './Modal';
import { Button } from './Button';

interface ConfirmModalProps {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: () => void;
    title: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    loading?: boolean;
    variant?: 'danger' | 'warning' | 'info';
}

export const ConfirmModal = ({
    isOpen,
    onClose,
    onConfirm,
    title,
    message,
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    loading = false,
    variant = 'danger',
}: ConfirmModalProps) => {
    const iconColors = {
        danger: 'bg-red-500/10 text-red-500',
        warning: 'bg-warning/10 text-warning',
        info: 'bg-primary/10 text-primary',
    };

    const buttonVariants = {
        danger: 'danger' as const,
        warning: 'primary' as const,
        info: 'primary' as const,
    };

    return (
        <Modal isOpen={isOpen} onClose={onClose} title={title} size="sm">
            <div className="space-y-4">
                <div className="flex items-start gap-4">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${iconColors[variant]}`}>
                        <AlertTriangle className="w-5 h-5" />
                    </div>
                    <p className="text-sm text-secondary flex-1">{message}</p>
                </div>
                <div className="flex justify-end gap-3 pt-2">
                    <Button
                        variant="ghost"
                        onClick={onClose}
                        disabled={loading}
                    >
                        {cancelText}
                    </Button>
                    <Button
                        variant={buttonVariants[variant]}
                        onClick={onConfirm}
                        loading={loading}
                    >
                        {confirmText}
                    </Button>
                </div>
            </div>
        </Modal>
    );
};
