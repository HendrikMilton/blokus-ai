import { useState, useEffect } from 'react';
import { Plus } from 'lucide-react';
import { Modal } from '../../../components/ui/Modal';
import { Button } from '../../../components/ui/Button';
import { Input } from '../../../components/ui/Input';

interface Props {
    isOpen: boolean;
    onClose: () => void;
    onSave: (name: string) => Promise<void>;
}

export const CreateTagModal = ({ isOpen, onClose, onSave }: Props) => {
    const [name, setName] = useState('');
    const [isSaving, setIsSaving] = useState(false);

    useEffect(() => {
        if (isOpen) {
            setName('');
        }
    }, [isOpen]);

    const handleSave = async () => {
        if (!name.trim()) {
            return;
        }

        setIsSaving(true);
        try {
            await onSave(name.trim());
            onClose();
        } finally {
            setIsSaving(false);
        }
    };

    const handleClose = () => {
        setName('');
        onClose();
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && name.trim()) {
            handleSave();
        }
    };

    return (
        <Modal
            isOpen={isOpen}
            onClose={handleClose}
            title="Create Tag"
            size="sm"
        >
            <div className="space-y-4">
                <Input
                    label="Tag Name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="e.g., training-set, needs-review"
                    autoFocus
                />

                <div className="flex justify-end gap-2 pt-2">
                    <Button variant="secondary" onClick={handleClose}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleSave}
                        disabled={!name.trim()}
                        loading={isSaving}
                        leftIcon={<Plus className="w-4 h-4" />}
                    >
                        Create Tag
                    </Button>
                </div>
            </div>
        </Modal>
    );
};
