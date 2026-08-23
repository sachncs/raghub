/**
 * @raghub/api — public surface.
 */

export { createApp } from './app.js';
export type { AppDeps } from './app.js';
export { errorMiddleware } from './middleware/error.js';
export { jwtAuthMiddleware, getClaims } from './middleware/auth.js';
export type { AuthVars } from './middleware/auth.js';
export { authRoutes } from './routes/auth.js';
export { documentsRoutes } from './routes/documents.js';
export { meRoutes } from './routes/me.js';
export { queryRoutes } from './routes/query.js';