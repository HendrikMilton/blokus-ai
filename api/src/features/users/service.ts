import { knex } from '../../knex';
import type { CreateUserRequest, UpdateUserRequest, SafeUser, User } from '@blokus-ai/shared';

// Convert DB user to safe user (remove password)
const _toSafeUser = (user: User & { password?: string }): SafeUser => {
    const { password: _, ...safeUser } = user as User & { password?: string };
    return safeUser;
};

export const getAllUsers = async (): Promise<SafeUser[]> => {
    const users = await knex('users')
        .select('id', 'email', 'name', 'role', 'status', 'created_at', 'updated_at')
        .orderBy('created_at', 'desc');
    return users;
};

export const getUserById = async (id: number): Promise<SafeUser | null> => {
    const user = await knex('users')
        .select('id', 'email', 'name', 'role', 'status', 'created_at', 'updated_at')
        .where('id', id)
        .first();
    return user || null;
};

export const createUser = async (data: CreateUserRequest): Promise<SafeUser> => {
    // In production, hash the password with bcrypt
    const [user] = await knex('users')
        .insert({
            email: data.email,
            name: data.name,
            password: data.password, // TODO: Hash with bcrypt
            role: data.role || 'user',
            status: 'active',
        })
        .returning(['id', 'email', 'name', 'role', 'status', 'created_at', 'updated_at']);

    return user;
};

export const updateUser = async (
    id: number,
    data: UpdateUserRequest
): Promise<SafeUser | null> => {
    const updateData: Record<string, unknown> = {
        ...data,
        updated_at: new Date(),
    };

    // If password is provided, hash it
    if (data.password) {
        // TODO: Hash with bcrypt
        updateData.password = data.password;
    }

    const [user] = await knex('users')
        .where('id', id)
        .update(updateData)
        .returning(['id', 'email', 'name', 'role', 'status', 'created_at', 'updated_at']);

    return user || null;
};

export const deleteUser = async (id: number): Promise<boolean> => {
    const deleted = await knex('users').where('id', id).delete();
    return deleted > 0;
};
