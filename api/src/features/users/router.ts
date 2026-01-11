import { Router } from 'express';
import typia from 'typia';
import { validateBody } from '../../lib/validate';
import { asyncHandler } from '../../lib/async-handler';
import { NotFoundError } from '../../error';
import {
    getAllUsers,
    getUserById,
    createUser,
    updateUser,
    deleteUser,
} from './service';
import type { CreateUserRequest, UpdateUserRequest, SafeUser } from '@blokus-ai/shared';

export const usersRouter = Router();

// GET /users - List all users
usersRouter.get('/', asyncHandler(async (_req, res) => {
    const users = await getAllUsers();
    res.json(users satisfies SafeUser[]);
}));

// GET /users/:id - Get user by ID
usersRouter.get('/:id', asyncHandler(async (req, res) => {
    const id = parseInt(req.params.id, 10);
    const user = await getUserById(id);

    if (!user) {
        throw new NotFoundError('User not found');
    }

    res.json(user satisfies SafeUser);
}));

// POST /users - Create user
usersRouter.post(
    '/',
    validateBody(typia.createValidateEquals<CreateUserRequest>()),
    asyncHandler(async (req, res) => {
        const user = await createUser(req.body);
        res.status(201).json(user satisfies SafeUser);
    })
);

// PUT /users/:id - Update user
usersRouter.put(
    '/:id',
    validateBody(typia.createValidateEquals<UpdateUserRequest>()),
    asyncHandler(async (req, res) => {
        const id = parseInt(req.params.id, 10);
        const user = await updateUser(id, req.body);

        if (!user) {
            throw new NotFoundError('User not found');
        }

        res.json(user satisfies SafeUser);
    })
);

// DELETE /users/:id - Delete user
usersRouter.delete('/:id', asyncHandler(async (req, res) => {
    const id = parseInt(req.params.id, 10);
    const deleted = await deleteUser(id);

    if (!deleted) {
        throw new NotFoundError('User not found');
    }

    res.status(204).send();
}));
