import { http } from './config';
import type {
    SafeUser,
    CreateUserRequest,
    UpdateUserRequest,
} from '@blokus-ai/shared';

const path = '/users';

export const USERS_API = {
    getAll: () => http.get<SafeUser[]>(path),

    getById: (id: number) => http.get<SafeUser>(`${path}/${id}`),

    create: (data: CreateUserRequest) => http.post<SafeUser>(path, data),

    update: (id: number, data: UpdateUserRequest) =>
        http.put<SafeUser>(`${path}/${id}`, data),

    delete: (id: number) => http.delete<void>(`${path}/${id}`),
};
