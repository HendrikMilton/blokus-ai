import { expect } from 'chai';
import request from 'supertest';
import express from 'express';
import { pingRouter } from './router';

describe('Ping Router', () => {
    const app = express();
    app.use('/ping', pingRouter);

    describe('GET /ping/liveness', () => {
        it('should return pong message', async () => {
            const res = await request(app).get('/ping/liveness');
            expect(res.status).to.equal(200);
            expect(res.body.message).to.equal('pong');
        });
    });
});
