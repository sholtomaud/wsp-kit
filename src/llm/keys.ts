/**
 * API keys stay in this browser and go only to their provider. By default a
 * key lives in memory and is gone on refresh; the person can choose to keep
 * it for the tab's lifetime (sessionStorage) or on this device (localStorage).
 * Keys are never part of a checkpoint, an export, or the conversation.
 */

export type KeyPersistence = 'memory' | 'session' | 'local';

const PREFIX = 'wsp-kit:key:';
const memory = new Map<string, string>();

function storage(kind: KeyPersistence): Storage | undefined {
  try {
    if (kind === 'session') return globalThis.sessionStorage;
    if (kind === 'local') return globalThis.localStorage;
  } catch {
    /* storage blocked */
  }
  return undefined;
}

export const keys = {
  get(provider: string): string {
    return memory.get(provider) ?? storage('session')?.getItem(PREFIX + provider) ?? storage('local')?.getItem(PREFIX + provider) ?? '';
  },

  /** Where a stored key currently lives. */
  where(provider: string): KeyPersistence | undefined {
    if (memory.has(provider)) return 'memory';
    if (storage('session')?.getItem(PREFIX + provider)) return 'session';
    if (storage('local')?.getItem(PREFIX + provider)) return 'local';
    return undefined;
  },

  set(provider: string, key: string, persistence: KeyPersistence) {
    this.clear(provider);
    if (!key) return;
    if (persistence === 'memory') memory.set(provider, key);
    else storage(persistence)?.setItem(PREFIX + provider, key);
  },

  clear(provider: string) {
    memory.delete(provider);
    storage('session')?.removeItem(PREFIX + provider);
    storage('local')?.removeItem(PREFIX + provider);
  },
};
