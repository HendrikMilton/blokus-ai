import { useState, useEffect, useCallback } from 'react';
import { Button, Input, Select, Table, Modal, ConfirmModal, TableSkeleton } from '../components/ui';
import type { Column } from '../components/ui';
import { USERS_API } from '../api/users';
import { useSocketEvent } from '../hooks/useSocketEvent';
import type { SafeUser, CreateUserRequest, UpdateUserRequest, UserRole, UserStatus } from '@blokus-ai/shared';
import { toast } from 'react-hot-toast';

const roleOptions = [
    { value: 'user', label: 'User' },
    { value: 'admin', label: 'Admin' },
];

const statusOptions = [
    { value: 'active', label: 'Active' },
    { value: 'inactive', label: 'Inactive' },
];

export const UsersPage = () => {
    const [users, setUsers] = useState<SafeUser[]>([]);
    const [loading, setLoading] = useState(true);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [editingUser, setEditingUser] = useState<SafeUser | null>(null);
    const [formData, setFormData] = useState<CreateUserRequest & { status?: UserStatus }>({
        email: '',
        name: '',
        password: '',
        role: 'user',
    });
    const [saving, setSaving] = useState(false);
    const [deletingUser, setDeletingUser] = useState<SafeUser | null>(null);
    const [deleting, setDeleting] = useState(false);

    // Fetch users
    const fetchUsers = useCallback(async () => {
        try {
            const data = await USERS_API.getAll();
            setUsers(data);
        } catch {
            toast.error('Failed to fetch users');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchUsers();
    }, [fetchUsers]);

    // Listen for real-time updates via WebSocket
    // These handle updates from OTHER clients - local actions already update state
    useSocketEvent('user-created', (data) => {
        setUsers((prev) => {
            // Only add if not already in list (prevents duplicates from own actions)
            if (prev.some((u) => u.id === data.user.id)) {
                return prev;
            }
            return [data.user, ...prev];
        });
    });

    useSocketEvent('user-updated', (data) => {
        setUsers((prev) =>
            prev.map((u) => (u.id === data.user.id ? data.user : u))
        );
    });

    useSocketEvent('user-deleted', (data) => {
        setUsers((prev) => prev.filter((u) => u.id !== data.userId));
    });

    // Open modal for create
    const handleCreate = () => {
        setEditingUser(null);
        setFormData({ email: '', name: '', password: '', role: 'user' });
        setIsModalOpen(true);
    };

    // Open modal for edit
    const handleEdit = (user: SafeUser) => {
        setEditingUser(user);
        setFormData({
            email: user.email,
            name: user.name,
            password: '',
            role: user.role,
            status: user.status,
        });
        setIsModalOpen(true);
    };

    // Delete user - open confirmation modal
    const handleDelete = (user: SafeUser) => {
        setDeletingUser(user);
    };

    // Confirm delete
    const confirmDelete = async () => {
        if (!deletingUser) {
            return;
        }

        setDeleting(true);
        try {
            await USERS_API.delete(deletingUser.id);
            setUsers((prev) => prev.filter((u) => u.id !== deletingUser.id));
            toast.success('User deleted');
            setDeletingUser(null);
        } catch {
            toast.error('Failed to delete user');
        } finally {
            setDeleting(false);
        }
    };

    // Save user (create or update)
    const handleSave = async () => {
        setSaving(true);
        try {
            if (editingUser) {
                const updateData: UpdateUserRequest = {
                    email: formData.email,
                    name: formData.name,
                    role: formData.role,
                    status: formData.status,
                };
                if (formData.password) {
                    updateData.password = formData.password;
                }
                const updated = await USERS_API.update(editingUser.id, updateData);
                setUsers((prev) =>
                    prev.map((u) => (u.id === updated.id ? updated : u))
                );
                toast.success('User updated');
            } else {
                const created = await USERS_API.create({
                    email: formData.email,
                    name: formData.name,
                    password: formData.password,
                    role: formData.role,
                });
                setUsers((prev) => [created, ...prev]);
                toast.success('User created');
            }
            setIsModalOpen(false);
        } catch {
            toast.error(editingUser ? 'Failed to update user' : 'Failed to create user');
        } finally {
            setSaving(false);
        }
    };

    // Table columns
    const columns: Column<SafeUser>[] = [
        { key: 'id', header: 'ID', className: 'w-16' },
        { key: 'name', header: 'Name' },
        { key: 'email', header: 'Email' },
        {
            key: 'role',
            header: 'Role',
            render: (user) => (
                <span
                    className={`px-2 py-1 rounded text-xs font-medium ${
                        user.role === 'admin'
                            ? 'bg-primary/10 text-primary'
                            : 'bg-muted text-muted-foreground'
                    }`}
                >
                    {user.role}
                </span>
            ),
        },
        {
            key: 'status',
            header: 'Status',
            render: (user) => (
                <span
                    className={`px-2 py-1 rounded text-xs font-medium ${
                        user.status === 'active'
                            ? 'bg-green-100 text-green-700'
                            : 'bg-red-100 text-red-700'
                    }`}
                >
                    {user.status}
                </span>
            ),
        },
        {
            key: 'created_at',
            header: 'Created',
            render: (user) => new Date(user.created_at).toLocaleDateString(),
        },
        {
            key: 'actions',
            header: 'Actions',
            render: (user) => (
                <div className="flex gap-2">
                    <Button size="xs" variant="ghost" onClick={() => handleEdit(user)}>
                        Edit
                    </Button>
                    <Button
                        size="xs"
                        variant="destructive"
                        onClick={() => handleDelete(user)}
                    >
                        Delete
                    </Button>
                </div>
            ),
        },
    ];

    if (loading) {
        return (
            <div>
                <div className="flex items-center justify-between mb-6">
                    <h1 className="text-2xl font-semibold text-foreground">Users</h1>
                    <Button disabled>Add User</Button>
                </div>
                <TableSkeleton rows={5} />
            </div>
        );
    }

    return (
        <>
            <div className="flex items-center justify-between mb-6">
                <h1 className="text-2xl font-semibold text-foreground">Users</h1>
                <Button onClick={handleCreate}>Add User</Button>
            </div>

            <Table
                columns={columns}
                data={users}
                keyField="id"
                emptyMessage="No users found"
            />

            <Modal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                title={editingUser ? 'Edit User' : 'Create User'}
                size="md"
            >
                <div className="space-y-4">
                    <Input
                        label="Name"
                        value={formData.name}
                        onChange={(e) =>
                            setFormData((prev) => ({ ...prev, name: e.target.value }))
                        }
                        placeholder="John Doe"
                    />
                    <Input
                        label="Email"
                        type="email"
                        value={formData.email}
                        onChange={(e) =>
                            setFormData((prev) => ({ ...prev, email: e.target.value }))
                        }
                        placeholder="john@example.com"
                    />
                    <Input
                        label={editingUser ? 'Password (leave blank to keep current)' : 'Password'}
                        type="password"
                        value={formData.password}
                        onChange={(e) =>
                            setFormData((prev) => ({ ...prev, password: e.target.value }))
                        }
                        placeholder="********"
                    />
                    <Select
                        label="Role"
                        value={formData.role}
                        options={roleOptions}
                        onChange={(value) =>
                            setFormData((prev) => ({ ...prev, role: value as UserRole }))
                        }
                    />
                    {editingUser && (
                        <Select
                            label="Status"
                            value={formData.status}
                            options={statusOptions}
                            onChange={(value) =>
                                setFormData((prev) => ({
                                    ...prev,
                                    status: value as UserStatus,
                                }))
                            }
                        />
                    )}
                    <div className="flex justify-end gap-3 pt-4">
                        <Button
                            variant="ghost"
                            onClick={() => setIsModalOpen(false)}
                            disabled={saving}
                        >
                            Cancel
                        </Button>
                        <Button onClick={handleSave} loading={saving}>
                            {editingUser ? 'Update' : 'Create'}
                        </Button>
                    </div>
                </div>
            </Modal>

            <ConfirmModal
                isOpen={!!deletingUser}
                onClose={() => setDeletingUser(null)}
                onConfirm={confirmDelete}
                title="Delete User"
                message={`Are you sure you want to delete "${deletingUser?.name}"? This action cannot be undone.`}
                confirmText="Delete"
                loading={deleting}
                variant="danger"
            />
        </>
    );
};
