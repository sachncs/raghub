/**
 * Workspace passphrase + AES-256-GCM encryption at rest.
 *
 * `Workspace.open(path)` keeps its existing shape for now and
 * accepts an optional `passphrase`. When provided:
 *   1. scrypt(passphrase, salt, N=2^15, r=8, p=1) -> 32-byte key
 *   2. Verify: AES-GCM decrypt the row in workspace_keycheck; if it
 *      matches "revex", proceed; else throw ConfigurationError.
 *   3. workspace_settings rows are stored as { nonce, ciphertext }.
 *      Reads decrypt on the fly; writes encrypt transparently.
 *
 * Without a passphrase the workspace runs in plaintext mode
 * (development convenience). The CLI server always passes the
 * passphrase; the Next.js API server reads it from
 * REVEX_WORKSPACE_PASSPHRASE.
 *
 * The KDF parameters + salt + verifier are stored in plaintext in
 * workspace_keycheck (the salt is not secret; the verifier is
 * AES-GCM("revex", key) — proving possession of the key without
 * revealing anything useful).
 */

import {
  createCipheriv,
  createDecipheriv,
  randomBytes,
  scryptSync,
} from 'node:crypto';

import { ConfigurationError } from './errors/index.js';
import type { Database, WorkspaceHandle, WorkspaceOptions } from './workspace.js';
import { openWorkspace } from './workspace.js';

const SCRYPT_N = 1 << 15;
const SCRYPT_R = 8;
const SCRYPT_P = 1;
const SCRYPT_KEYLEN = 32;
const VERIFIER_PLAINTEXT = 'revex';
const NONCE_LEN = 12;
const TAG_LEN = 16;

export interface EncryptedField {
  readonly nonce: Buffer;
  readonly ciphertext: Buffer;
}

export interface WorkspaceKeyParams {
  readonly n: number;
  readonly r: number;
  readonly p: number;
  readonly keylen: number;
}

const deriveKey = (passphrase: string, salt: Buffer, params: WorkspaceKeyParams): Buffer => {
  return scryptSync(Buffer.from(passphrase, 'utf8'), salt, params.keylen, {
    N: params.n,
    r: params.r,
    p: params.p,
    maxmem: 128 * params.n * params.r * 2,
  });
};

const encryptField = (plaintext: string, key: Buffer): EncryptedField => {
  const nonce = randomBytes(NONCE_LEN);
  const cipher = createCipheriv('aes-256-gcm', key, nonce);
  const ct = Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  return { nonce, ciphertext: Buffer.concat([ct, tag]) };
};

const decryptField = (field: EncryptedField, key: Buffer): string => {
  if (field.ciphertext.length < TAG_LEN) {
    throw new ConfigurationError('encrypted field too short (corrupted)');
  }
  const ct = field.ciphertext.subarray(0, field.ciphertext.length - TAG_LEN);
  const tag = field.ciphertext.subarray(field.ciphertext.length - TAG_LEN);
  const decipher = createDecipheriv('aes-256-gcm', key, field.nonce);
  decipher.setAuthTag(tag);
  const pt = Buffer.concat([decipher.update(ct), decipher.final()]);
  return pt.toString('utf8');
};

const DEFAULT_KDF_PARAMS: WorkspaceKeyParams = {
  n: SCRYPT_N,
  r: SCRYPT_R,
  p: SCRYPT_P,
  keylen: SCRYPT_KEYLEN,
};

interface KeycheckRow {
  salt: Buffer;
  kdf_params: string;
  verifier_nonce: Buffer;
  verifier_ciphertext: Buffer;
  created_at: number;
}

const readKeycheck = (db: Database): KeycheckRow | null => {
  const row = db
    .prepare('SELECT salt, kdf_params, verifier_nonce, verifier_ciphertext, created_at FROM workspace_keycheck WHERE id = 1')
    .get() as Record<string, unknown> | undefined;
  if (!row) return null;
  return {
    salt: row['salt'] as Buffer,
    kdf_params: String(row['kdf_params']),
    verifier_nonce: row['verifier_nonce'] as Buffer,
    verifier_ciphertext: row['verifier_ciphertext'] as Buffer,
    created_at: Number(row['created_at']),
  };
};

const writeKeycheck = (
  db: Database,
  salt: Buffer,
  kdfParams: WorkspaceKeyParams,
  verifier: EncryptedField,
): void => {
  db.prepare(
    `INSERT OR REPLACE INTO workspace_keycheck (id, salt, kdf_params, verifier_nonce, verifier_ciphertext, created_at)
     VALUES (1, ?, ?, ?, ?, ?)`,
  ).run(salt, JSON.stringify(kdfParams), verifier.nonce, verifier.ciphertext, Date.now());
};

const unlockKey = (passphrase: string, kc: KeycheckRow): Buffer => {
  const params: WorkspaceKeyParams =
    kc.kdf_params === '' ? DEFAULT_KDF_PARAMS : (JSON.parse(kc.kdf_params) as WorkspaceKeyParams);
  const key = deriveKey(passphrase, kc.salt, params);
  try {
    const decoded = decryptField(
      { nonce: kc.verifier_nonce, ciphertext: kc.verifier_ciphertext },
      key,
    );
    if (decoded !== VERIFIER_PLAINTEXT) {
      throw new ConfigurationError('workspace passphrase mismatch');
    }
  } catch (e) {
    if (e instanceof ConfigurationError) throw e;
    throw new ConfigurationError('workspace passphrase mismatch', { cause: e });
  }
  return key;
};

export interface WorkspaceSettingsStore {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T): Promise<void>;
  all(): Promise<Readonly<Record<string, unknown>>>;
  delete(key: string): Promise<void>;
}

const noPassphraseSettingsStore = (db: Database): WorkspaceSettingsStore => {
  return {
    async get<T>(key: string): Promise<T | null> {
      const row = db
        .prepare('SELECT ciphertext FROM workspace_settings WHERE key = ?')
        .get(key) as Record<string, unknown> | undefined;
      if (!row) return null;
      try {
        return JSON.parse(String(row['ciphertext'])) as T;
      } catch {
        return null;
      }
    },
    async set<T>(key: string, value: T): Promise<void> {
      db.prepare(
        `INSERT INTO workspace_settings (key, nonce, ciphertext, updated_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(key) DO UPDATE SET nonce = excluded.nonce, ciphertext = excluded.ciphertext, updated_at = excluded.updated_at`,
      ).run(key, Buffer.alloc(0), JSON.stringify(value), Date.now());
    },
    async all(): Promise<Readonly<Record<string, unknown>>> {
      const rows = db
        .prepare('SELECT key, ciphertext FROM workspace_settings')
        .all() as Record<string, unknown>[];
      const out: Record<string, unknown> = {};
      for (const r of rows) {
        try {
          out[String(r['key'])] = JSON.parse(String(r['ciphertext']));
        } catch {
          /* skip */
        }
      }
      return out;
    },
    async delete(key: string): Promise<void> {
      db.prepare('DELETE FROM workspace_settings WHERE key = ?').run(key);
    },
  };
};

const encryptedSettingsStore = (db: Database, key: Buffer): WorkspaceSettingsStore => {
  return {
    async get<T>(keyName: string): Promise<T | null> {
      const row = db
        .prepare('SELECT nonce, ciphertext FROM workspace_settings WHERE key = ?')
        .get(keyName) as Record<string, unknown> | undefined;
      if (!row) return null;
      const plaintext = decryptField(
        {
          nonce: row['nonce'] as Buffer,
          ciphertext: row['ciphertext'] as Buffer,
        },
        key,
      );
      return JSON.parse(plaintext) as T;
    },
    async set<T>(keyName: string, value: T): Promise<void> {
      const field = encryptField(JSON.stringify(value), key);
      db.prepare(
        `INSERT INTO workspace_settings (key, nonce, ciphertext, updated_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(key) DO UPDATE SET nonce = excluded.nonce, ciphertext = excluded.ciphertext, updated_at = excluded.updated_at`,
      ).run(keyName, field.nonce, field.ciphertext, Date.now());
    },
    async all(): Promise<Readonly<Record<string, unknown>>> {
      const rows = db
        .prepare('SELECT key, nonce, ciphertext FROM workspace_settings')
        .all() as Record<string, unknown>[];
      const out: Record<string, unknown> = {};
      for (const r of rows) {
        try {
          out[String(r['key'])] = JSON.parse(
            decryptField(
              { nonce: r['nonce'] as Buffer, ciphertext: r['ciphertext'] as Buffer },
              key,
            ),
          );
        } catch {
          /* skip corrupt rows */
        }
      }
      return out;
    },
    async delete(keyName: string): Promise<void> {
      db.prepare('DELETE FROM workspace_settings WHERE key = ?').run(keyName);
    },
  };
};

export interface WorkspaceWithSettings extends WorkspaceHandle {
  readonly settings: WorkspaceSettingsStore;
  readonly encryption: 'plaintext' | 'passphrase-aes-256-gcm';
}

export interface OpenWorkspaceOptions extends WorkspaceOptions {
  readonly passphrase?: string;
}

export const openEncryptedWorkspace = async (
  opts: OpenWorkspaceOptions,
): Promise<WorkspaceWithSettings> => {
  const handle = await openWorkspace(opts);
  const kc = readKeycheck(handle.db);
  let settingsStore: WorkspaceSettingsStore;
  let encryption: 'plaintext' | 'passphrase-aes-256-gcm';
  if (opts.passphrase) {
    let key: Buffer;
    if (kc === null) {
      const params = DEFAULT_KDF_PARAMS;
      const salt = randomBytes(16);
      const candidate = deriveKey(opts.passphrase, salt, params);
      const verifier = encryptField(VERIFIER_PLAINTEXT, candidate);
      writeKeycheck(handle.db, salt, params, verifier);
      key = candidate;
    } else {
      key = unlockKey(opts.passphrase, kc);
    }
    settingsStore = encryptedSettingsStore(handle.db, key);
    encryption = 'passphrase-aes-256-gcm';
  } else {
    settingsStore = noPassphraseSettingsStore(handle.db);
    encryption = 'plaintext';
  }
  return {
    path: handle.path,
    db: handle.db,
    id: handle.id,
    settings: settingsStore,
    encryption,
    close: handle.close,
  };
};
