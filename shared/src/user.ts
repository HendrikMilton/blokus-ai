// User types

export type UserRole = 'admin' | 'user';
export type UserStatus = 'active' | 'inactive';

export interface User {
    id: number;
    email: string;
    name: string;
    role: UserRole;
    status: UserStatus;
    created_at: Date;
    updated_at: Date;
}

// Safe user type (without password)
export type SafeUser = Omit<User, 'password'>;

export interface CreateUserRequest {
    email: string;
    name: string;
    password: string;
    role?: UserRole;
}

export interface UpdateUserRequest {
    email?: string;
    name?: string;
    password?: string;
    role?: UserRole;
    status?: UserStatus;
}

// API path
export const usersPath = '/users';
