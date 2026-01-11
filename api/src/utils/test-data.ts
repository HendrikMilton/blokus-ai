import type { CreateUserRequest } from '@blokus-ai/shared';

/**
 * Test user data fixtures
 */
export const TEST_ADMIN_USER: CreateUserRequest = {
    email: 'admin@test.com',
    name: 'Test Admin',
    password: 'password123',
    role: 'admin',
};

export const TEST_REGULAR_USER: CreateUserRequest = {
    email: 'user@test.com',
    name: 'Test User',
    password: 'password123',
    role: 'user',
};

/**
 * Generate unique test user data
 */
export const createTestUser = (suffix: string): CreateUserRequest => ({
    email: `user-${suffix}@test.com`,
    name: `Test User ${suffix}`,
    password: 'password123',
    role: 'user',
});

/**
 * Generate unique admin user data
 */
export const createTestAdmin = (suffix: string): CreateUserRequest => ({
    email: `admin-${suffix}@test.com`,
    name: `Test Admin ${suffix}`,
    password: 'password123',
    role: 'admin',
});
