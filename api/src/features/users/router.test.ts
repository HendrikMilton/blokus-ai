import { describe, it, before, beforeEach, after } from 'mocha';
import { validateEquals } from 'typia';
import {
    setupTestDatabase,
    truncateTables,
    teardownTestDatabase,
    getTestKnex,
    expect,
    request,
} from '../../utils/test-helper';
import { testType, Serialize } from '../../utils/test-utils';
import { TEST_ADMIN_USER, TEST_REGULAR_USER, createTestUser } from '../../utils/test-data';
import type { SafeUser } from '@blokus-ai/shared';

// Import the Express app (not the http server)
import express from 'express';
import cors from 'cors';
import { usersRouter } from './router';
import { errorHandler } from '../../error';
import { cidMiddleware } from '../../middleware/cid';

// Create a test app instance
const app = express();
app.use(express.json());
app.use(cors());
app.use(cidMiddleware);
app.use('/users', usersRouter);
app.use(errorHandler);

describe('Users API', () => {
    let testCounter = 0;

    before(async () => {
        await setupTestDatabase();
    });

    beforeEach(async () => {
        await truncateTables();
        testCounter++;
    });

    after(async () => {
        await teardownTestDatabase();
    });

    describe('GET /users', () => {
        it('should return empty array when no users exist', async () => {
            const resp = await request(app).get('/users').expect(200);

            expect(resp.body).to.be.an('array');
            expect(resp.body).to.have.length(0);
        });

        it('should return all users', async () => {
            const knex = getTestKnex();

            // Create test users
            await knex('users').insert([
                { ...TEST_ADMIN_USER, password: 'hashed' },
                { ...TEST_REGULAR_USER, password: 'hashed' },
            ]);

            const resp = await request(app).get('/users').expect(200);

            expect(resp.body).to.be.an('array');
            expect(resp.body).to.have.length(2);

            // Validate each user matches SafeUser type
            resp.body.forEach((user: unknown) => {
                testType(() => validateEquals<Serialize<SafeUser>>(user));
            });
        });
    });

    describe('GET /users/:id', () => {
        it('should return user by id', async () => {
            const knex = getTestKnex();

            const [user] = await knex('users')
                .insert({ ...TEST_ADMIN_USER, password: 'hashed' })
                .returning('*');

            const resp = await request(app).get(`/users/${user.id}`).expect(200);

            // Validate response type
            testType(() => validateEquals<Serialize<SafeUser>>(resp.body));

            expect(resp.body.id).to.equal(user.id);
            expect(resp.body.email).to.equal(TEST_ADMIN_USER.email);
            expect(resp.body.name).to.equal(TEST_ADMIN_USER.name);
            expect(resp.body.role).to.equal(TEST_ADMIN_USER.role);
            // Password should NOT be returned
            expect(resp.body).to.not.have.property('password');
        });

        it('should return 404 for non-existent user', async () => {
            const resp = await request(app).get('/users/99999').expect(404);

            expect(resp.body.message).to.equal('User not found');
        });
    });

    describe('POST /users', () => {
        it('should create a new user', async () => {
            const userData = createTestUser(`${testCounter}`);

            const resp = await request(app)
                .post('/users')
                .send(userData)
                .expect(201);

            // Validate response type
            testType(() => validateEquals<Serialize<SafeUser>>(resp.body));

            expect(resp.body.email).to.equal(userData.email);
            expect(resp.body.name).to.equal(userData.name);
            expect(resp.body.role).to.equal(userData.role);
            expect(resp.body.status).to.equal('active');
            expect(resp.body).to.not.have.property('password');
        });

        it('should reject invalid request body', async () => {
            const resp = await request(app)
                .post('/users')
                .send({ email: 'test@test.com' }) // Missing required fields
                .expect(400);

            expect(resp.body.message).to.include('Validation failed');
        });

        it('should reject extra properties', async () => {
            const userData = {
                ...createTestUser(`${testCounter}`),
                extraField: 'should be rejected',
            };

            const resp = await request(app)
                .post('/users')
                .send(userData)
                .expect(400);

            expect(resp.body.message).to.include('Validation failed');
        });
    });

    describe('PUT /users/:id', () => {
        it('should update an existing user', async () => {
            const knex = getTestKnex();

            const [user] = await knex('users')
                .insert({ ...TEST_REGULAR_USER, password: 'hashed' })
                .returning('*');

            const updateData = {
                name: 'Updated Name',
                role: 'admin' as const,
            };

            const resp = await request(app)
                .put(`/users/${user.id}`)
                .send(updateData)
                .expect(200);

            // Validate response type
            testType(() => validateEquals<Serialize<SafeUser>>(resp.body));

            expect(resp.body.id).to.equal(user.id);
            expect(resp.body.name).to.equal(updateData.name);
            expect(resp.body.role).to.equal(updateData.role);
            expect(resp.body.email).to.equal(TEST_REGULAR_USER.email);
        });

        it('should return 404 for non-existent user', async () => {
            const resp = await request(app)
                .put('/users/99999')
                .send({ name: 'Updated' })
                .expect(404);

            expect(resp.body.message).to.equal('User not found');
        });

        it('should update user status', async () => {
            const knex = getTestKnex();

            const [user] = await knex('users')
                .insert({ ...TEST_REGULAR_USER, password: 'hashed' })
                .returning('*');

            const resp = await request(app)
                .put(`/users/${user.id}`)
                .send({ status: 'inactive' })
                .expect(200);

            testType(() => validateEquals<Serialize<SafeUser>>(resp.body));

            expect(resp.body.status).to.equal('inactive');
        });
    });

    describe('DELETE /users/:id', () => {
        it('should delete an existing user', async () => {
            const knex = getTestKnex();

            const [user] = await knex('users')
                .insert({ ...TEST_REGULAR_USER, password: 'hashed' })
                .returning('*');

            await request(app).delete(`/users/${user.id}`).expect(204);

            // Verify user is deleted
            const deletedUser = await knex('users').where('id', user.id).first();
            expect(deletedUser).to.be.undefined;
        });

        it('should return 404 for non-existent user', async () => {
            const resp = await request(app).delete('/users/99999').expect(404);

            expect(resp.body.message).to.equal('User not found');
        });
    });
});
