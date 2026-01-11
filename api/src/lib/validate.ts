import { Request, Response, NextFunction } from 'express';
import { IValidation } from 'typia';
import { BadRequestError } from '../error';

/**
 * Typia validation middleware for Express routes.
 *
 * IMPORTANT: Use typia.createValidateEquals for request validation.
 * - validateEquals is STRICT: rejects extra properties (security!)
 * - validate is LOOSE: allows extra properties (dangerous for APIs)
 *
 * Example usage:
 * ```typescript
 * import typia from 'typia';
 *
 * interface CreateUserRequest {
 *   email: string;
 *   password: string;
 * }
 *
 * router.post('/users',
 *   validateBody(typia.createValidateEquals<CreateUserRequest>()),
 *   async (req, res) => {
 *     // req.body is guaranteed to be exactly CreateUserRequest
 *     // No extra fields allowed - prevents mass assignment attacks
 *   }
 * );
 * ```
 */
export const validateBody = <T>(validator: (input: unknown) => IValidation<T>) => {
    return (req: Request, res: Response, next: NextFunction) => {
        const result = validator(req.body);

        if (!result.success) {
            const errors = result.errors
                .map(e => `${e.path}: ${e.expected}`)
                .join(', ');
            throw new BadRequestError(`Validation failed: ${errors}`);
        }

        // Replace body with validated & typed version
        req.body = result.data;
        next();
    };
};

/**
 * Validate query parameters
 */
export const validateQuery = <T>(validator: (input: unknown) => IValidation<T>) => {
    return (req: Request, res: Response, next: NextFunction) => {
        const result = validator(req.query);

        if (!result.success) {
            const errors = result.errors
                .map(e => `${e.path}: ${e.expected}`)
                .join(', ');
            throw new BadRequestError(`Query validation failed: ${errors}`);
        }

        req.query = result.data as any;
        next();
    };
};
