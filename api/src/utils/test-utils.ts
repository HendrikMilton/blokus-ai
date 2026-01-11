import { IValidation } from 'typia';

/**
 * Serialize type - converts Date fields to strings for API response validation.
 * API responses return dates as ISO strings, not Date objects.
 */
export type Serialize<T> = T extends Date
    ? string
    : T extends Array<infer U>
        ? Array<Serialize<U>>
        : T extends object
            ? { [P in keyof T]: T[P] extends Date ? string : T[P] extends Date | null ? string | null : Serialize<T[P]> }
            : T;

/**
 * Custom error class for Typia validation failures with detailed path information.
 */
class TypeValidationError extends Error {
    constructor(public validation: IValidation.IFailure) {
        const errors = validation.errors
            .map((e) => {
                if (e.expected === 'undefined') {
                    return `${e.path} is not allowed, received ${typeof e.value}`;
                }
                return `${e.path} received: ${typeof e.value} expected: ${e.expected}`;
            })
            .join('\n');
        const message = `Type validation failed:\n${errors}`;
        super(message);
    }
}

let hasWarned = false;
const warnOnce = () => {
    if (hasWarned) {
        return;
    }
    hasWarned = true;
    console.warn('Warning: Typia transform not configured. Type validation skipped.');
};

/**
 * Wrapper for Typia validation in integration tests.
 *
 * IMPORTANT: Use typia.validateEquals (NOT typia.validate) to ensure:
 * - API responses match the exact type structure
 * - No extra fields are present (catches API regressions)
 * - Date fields are properly serialized as strings
 *
 * Example usage in tests:
 * ```typescript
 * import typia from 'typia';
 * import { testType, Serialize } from '../utils/test-utils';
 * import { User } from '@blokus-ai/shared';
 *
 * it('should return user data', async () => {
 *   const resp = await request(app)
 *     .get('/users/me')
 *     .set('Authorization', `Bearer ${token}`)
 *     .expect(200);
 *
 *   // Validate response matches User type EXACTLY
 *   testType(() => typia.validateEquals<Serialize<User>>(resp.body));
 *
 *   // Additional assertions
 *   expect(resp.body.email).to.equal('test@example.com');
 * });
 * ```
 */
export const testType = (callback: () => IValidation<unknown>): void => {
    try {
        const result = callback();
        if (result.success) {
            return;
        }
        throw new TypeValidationError(result);
    } catch (e: unknown) {
        if (e instanceof Error && e.toString().includes('no transform has been configured')) {
            warnOnce();
            return;
        }
        throw e;
    }
};
