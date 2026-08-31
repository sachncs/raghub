/**
 * Telemetry factory + barrel.
 *
 * Picks a provider based on `Settings.telemetry.provider`. Unknown
 * providers fall back to NoOp; missing credentials fall back to
 * NoOp but log a one-shot warning so production deploys surface
 * the misconfiguration.
 */

import { NoOpTelemetry } from './noop.js';
import type { Telemetry } from './types.js';
import type { Settings } from '../settings/index.js';

export type { Telemetry, TelemetrySpan } from './types.js';
export { NoOpTelemetry } from './noop.js';
export { LangfuseTelemetry } from './langfuse.js';
export type { LangfuseTelemetryOptions } from './langfuse.js';
export { OtelTelemetry } from './otel.js';
export { createLogger, setSink } from './logger.js';
export type { LogLevel, LogRecord, LogSink, Logger } from './logger.js';
export { SECRET_KEY_RE, scrubSecrets, RedactingTelemetry } from './redaction.js';

export const createTelemetry = async (settings: Settings): Promise<Telemetry> => {
  switch (settings.telemetry.provider) {
    case 'noop':
      return new NoOpTelemetry();
    case 'langfuse': {
      if (!settings.telemetry.langfusePublicKey || !settings.telemetry.langfuseSecretKey) {
        console.warn('[revex] telemetry=langfuse but credentials missing; falling back to noop');
        return new NoOpTelemetry();
      }
      const { LangfuseTelemetry } = await import('./langfuse.js');
      const t = await LangfuseTelemetry.create({
        publicKey: settings.telemetry.langfusePublicKey,
        secretKey: settings.telemetry.langfuseSecretKey,
        ...(settings.telemetry.langfuseBaseUrl !== undefined
          ? { baseUrl: settings.telemetry.langfuseBaseUrl }
          : {}),
      });
      if (!t) {
        console.warn('[revex] langfuse package not installed; falling back to noop');
        return new NoOpTelemetry();
      }
      return t;
    }
    case 'otel': {
      const { OtelTelemetry } = await import('./otel.js');
      const t = await OtelTelemetry.create('revex');
      if (!t) {
        console.warn('[revex] @opentelemetry/api not installed; falling back to noop');
        return new NoOpTelemetry();
      }
      return t;
    }
    default: {
      const _exhaustive: never = settings.telemetry.provider;
      console.warn(`[revex] unknown telemetry provider ${String(_exhaustive)}; falling back to noop`);
      return new NoOpTelemetry();
    }
  }
};