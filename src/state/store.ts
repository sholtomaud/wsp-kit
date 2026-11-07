/**
 * Where checkpoints go: the browser's Origin Private File System (fast, local,
 * private to this origin, survives reloads), or memory where there is no OPFS
 * (Node, tests, some private windows).
 */

export interface CheckpointInfo {
  id: string;
  bytes: number;
  savedAt: number;
}

export interface CheckpointStore {
  readonly kind: 'opfs' | 'memory';
  save(id: string, data: ArrayBuffer): Promise<void>;
  load(id: string): Promise<ArrayBuffer>;
  list(): Promise<CheckpointInfo[]>;
  delete(id: string): Promise<void>;
}

/** Checkpoint ids become file names: keep them short and plain. */
export function checkId(id: string): string {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(id)) {
    throw new Error(`Checkpoint id "${id}" must be 1–64 letters, digits, dots, dashes or underscores`);
  }
  return id;
}

export class MemoryStore implements CheckpointStore {
  readonly kind = 'memory' as const;
  private items = new Map<string, { data: ArrayBuffer; savedAt: number }>();

  async save(id: string, data: ArrayBuffer) {
    this.items.set(checkId(id), { data: data.slice(0), savedAt: Date.now() });
  }
  async load(id: string) {
    const item = this.items.get(id);
    if (!item) throw new Error(`No checkpoint "${id}"`);
    return item.data.slice(0);
  }
  async list() {
    return [...this.items].map(([id, { data, savedAt }]) => ({ id, bytes: data.byteLength, savedAt }));
  }
  async delete(id: string) {
    this.items.delete(id);
  }
}


/**
 * Safari has long lacked FileSystemFileHandle.createWritable in the window;
 * every engine with OPFS has sync access handles in a worker, so writes fall
 * back to a tiny inline worker.
 */
const WRITER_WORKER = `
onmessage = async (e) => {
  const { dir, name, data } = e.data;
  try {
    let d = await navigator.storage.getDirectory();
    for (const part of dir) d = await d.getDirectoryHandle(part, { create: true });
    const file = await d.getFileHandle(name, { create: true });
    const handle = await file.createSyncAccessHandle();
    handle.truncate(0);
    handle.write(new Uint8Array(data), { at: 0 });
    handle.flush();
    handle.close();
    postMessage({ ok: true });
  } catch (err) {
    postMessage({ ok: false, error: String(err && err.message || err) });
  }
};`;

export class OpfsStore implements CheckpointStore {
  readonly kind = 'opfs' as const;
  private readonly path: string[];
  private readonly ext: string;

  constructor(path: string[] = ['wsp-kit', 'checkpoints'], ext = '.wspk') {
    this.path = path;
    this.ext = ext;
  }

  static available(): boolean {
    return typeof navigator !== 'undefined' && typeof navigator.storage?.getDirectory === 'function';
  }

  private async dir(): Promise<FileSystemDirectoryHandle> {
    let d = await navigator.storage.getDirectory();
    for (const part of this.path) d = await d.getDirectoryHandle(part, { create: true });
    return d;
  }

  async save(id: string, data: ArrayBuffer) {
    const name = checkId(id) + this.ext;
    const dir = await this.dir();
    const file = await dir.getFileHandle(name, { create: true });
    if ('createWritable' in file && typeof file.createWritable === 'function') {
      const w = await file.createWritable();
      await w.write(data);
      await w.close();
      return;
    }
    await writeInWorker(this.path, name, data);
  }

  async load(id: string) {
    const dir = await this.dir();
    let file: FileSystemFileHandle;
    try {
      file = await dir.getFileHandle(checkId(id) + this.ext);
    } catch {
      throw new Error(`No checkpoint "${id}"`);
    }
    return (await file.getFile()).arrayBuffer();
  }

  async list() {
    const dir = await this.dir();
    const out: CheckpointInfo[] = [];
    // entries() is async-iterable in every OPFS engine; the DOM typings lag.
    for await (const [name, handle] of (dir as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries()) {
      if (handle.kind !== 'file' || !name.endsWith(this.ext)) continue;
      const f = await (handle as FileSystemFileHandle).getFile();
      out.push({ id: name.slice(0, -this.ext.length), bytes: f.size, savedAt: f.lastModified });
    }
    return out.sort((a, b) => b.savedAt - a.savedAt);
  }

  async delete(id: string) {
    const dir = await this.dir();
    await dir.removeEntry(checkId(id) + this.ext).catch(() => {});
  }
}

function writeInWorker(dir: string[], name: string, data: ArrayBuffer): Promise<void> {
  const url = URL.createObjectURL(new Blob([WRITER_WORKER], { type: 'text/javascript' }));
  const worker = new Worker(url);
  return new Promise<void>((resolve, reject) => {
    worker.onmessage = (e) => {
      worker.terminate();
      URL.revokeObjectURL(url);
      if (e.data.ok) resolve();
      else reject(new Error(`OPFS write failed: ${e.data.error}`));
    };
    worker.onerror = (e) => {
      worker.terminate();
      URL.revokeObjectURL(url);
      reject(new Error(`OPFS writer worker failed: ${e.message}`));
    };
    const copy = data.slice(0);
    worker.postMessage({ dir, name, data: copy }, [copy]);
  });
}

export function defaultStore(path?: string[], ext?: string): CheckpointStore {
  return OpfsStore.available() ? new OpfsStore(path, ext) : new MemoryStore();
}
