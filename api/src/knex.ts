import Knex from 'knex';
import config from '../knexfile';

export const knex = Knex(config);
