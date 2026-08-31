/**
 * Plugin registry + entry-point discovery.
 *
 * `Registry<T>` is a polymorphic base class — concrete plugins
 * (vector stores, embedders, LLMs, retrievers, feedback scorers,
 * etc.) register themselves via `@Registry.register("name")`
 * decorators or `Registry.get("name")` lookups.
 *
 * `PluginRegistry` walks the standard Node entry-point group
 * (`revex.plugins`) at startup and instantiates each discovered
 * plugin. Entry points are declared in consumer `package.json`:
 *
 * ```json
 * {
 *   "revex": {
 *     "plugins": [
 *       { "group": "vector_stores", "name": "qdrant", "module": "revex-qdrant" }
 *     ]
 *   }
 }
 * ```
 *
 * Or programmatically via `register("name", ctor)` for in-process
 * extensions.
 */

export interface PluginMeta {
  readonly group: string;
  readonly name: string;
  readonly module: string;
  readonly version?: string;
}

export interface RegistryCtor<T> {
  readonly name: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  new (...args: any[]): T;
}

export interface RegistryEntry<T> {
  readonly name: string;
  readonly ctor: RegistryCtor<T>;
  readonly meta?: PluginMeta;
}

const registryByGroup = new Map<string, Map<string, RegistryEntry<unknown>>>();

export class Registry<T extends { name: string } = { name: string }> {
  static byGroup<T>(group: string): Map<string, RegistryEntry<T>> {
    let m = registryByGroup.get(group) as Map<string, RegistryEntry<T>> | undefined;
    if (!m) {
      m = new Map();
      registryByGroup.set(group, m);
    }
    return m;
  }

  static groups(): readonly string[] {
    return Array.from(registryByGroup.keys());
  }

  static register<T extends { name: string }>(
    group: string,
    name: string,
    ctor: RegistryCtor<T>,
    meta?: PluginMeta,
  ): void {
    if (!name || !ctor) {
      throw new Error(`Registry: invalid registration for group "${group}"`);
    }
    const m = Registry.byGroup<T>(group);
    if (m.has(name)) {
      throw new Error(
        `Registry: duplicate registration for "${group}/${name}"`,
      );
    }
    const entry: RegistryEntry<T> = meta !== undefined
      ? { name, ctor, meta }
      : { name, ctor };
    m.set(name, entry);
  }

  static get<T extends { name: string }>(
    group: string,
    name: string,
  ): RegistryCtor<T> | null {
    const m = Registry.byGroup<T>(group);
    return m.get(name)?.ctor ?? null;
  }

  static create<T extends { name: string }>(
    group: string,
    name: string,
    ...args: readonly unknown[]
  ): T | null {
    const ctor = Registry.get<T>(group, name);
    if (!ctor) return null;
    return new ctor(...args);
  }

  static list<T extends { name: string }>(group: string): readonly RegistryEntry<T>[] {
    return Array.from(Registry.byGroup<T>(group).values());
  }

  static clear(group?: string): void {
    if (group === undefined) {
      registryByGroup.clear();
      return;
    }
    registryByGroup.delete(group);
  }
}

export class PluginRegistry {
  private readonly discovered = new Map<string, PluginMeta[]>();

  discover(packageJson: {
    readonly revex?: { readonly plugins?: readonly PluginMeta[] };
  }): readonly PluginMeta[] {
    const plugins = packageJson.revex?.plugins ?? [];
    this.discovered.set('local', [...plugins]);
    return plugins;
  }

  all(): readonly PluginMeta[] {
    const out: PluginMeta[] = [];
    for (const list of this.discovered.values()) out.push(...list);
    return out;
  }

  byGroup(group: string): readonly PluginMeta[] {
    return this.all().filter((p) => p.group === group);
  }
}

export const PLUGIN_GROUPS = {
  vectorStores: 'revex.vector_stores',
  embedders: 'revex.embedders',
  llms: 'revex.llms',
  retrievers: 'revex.retrievers',
  feedbackScorers: 'revex.feedback_scorers',
  rerankers: 'revex.rerankers',
  chunkers: 'revex.chunkers',
} as const;

export type PluginGroup = (typeof PLUGIN_GROUPS)[keyof typeof PLUGIN_GROUPS];