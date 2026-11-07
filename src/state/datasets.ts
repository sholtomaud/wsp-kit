/**
 * Named text datasets: what the Architect writes as the teacher, and what
 * microTrain learns from. Kept in OPFS (as JSON) so they survive reloads and
 * are shared by every model.
 */
import { checkId, defaultStore, type CheckpointStore } from './store.ts';

export interface DatasetInfo {
  name: string;
  texts: number;
  characters: number;
}

const MAX_TEXTS = 100_000;

export class Datasets {
  private readonly store: CheckpointStore;
  private cache = new Map<string, string[]>();

  constructor(store: CheckpointStore = defaultStore(['wsp-kit', 'datasets'], '.json')) {
    this.store = store;
  }

  async write(name: string, texts: string[], append = false): Promise<DatasetInfo> {
    checkId(name);
    if (!Array.isArray(texts) || texts.some((t) => typeof t !== 'string')) throw new Error('texts must be an array of strings');
    const clean = texts.filter((t) => t.length > 0);
    const existing = append ? await this.get(name).catch(() => [] as string[]) : [];
    const all = [...existing, ...clean];
    if (all.length > MAX_TEXTS) throw new RangeError(`A dataset holds at most ${MAX_TEXTS} texts`);
    this.cache.set(name, all);
    await this.store.save(name, new TextEncoder().encode(JSON.stringify(all)).buffer as ArrayBuffer);
    return info(name, all);
  }

  async get(name: string): Promise<string[]> {
    const hit = this.cache.get(name);
    if (hit) return hit;
    let data: ArrayBuffer;
    try {
      data = await this.store.load(name);
    } catch {
      throw new Error(`No dataset "${name}". Write one with dataset_write first.`);
    }
    const texts = JSON.parse(new TextDecoder().decode(data)) as string[];
    this.cache.set(name, texts);
    return texts;
  }

  async list(): Promise<DatasetInfo[]> {
    const out: DatasetInfo[] = [];
    for (const { id } of await this.store.list()) out.push(info(id, await this.get(id)));
    return out.sort((a, b) => a.name.localeCompare(b.name));
  }

  async sample(name: string, start = 0, count = 5) {
    const texts = await this.get(name);
    return { name, total: texts.length, start, texts: texts.slice(start, start + Math.min(Math.max(1, count), 50)) };
  }

  async delete(name: string) {
    this.cache.delete(name);
    await this.store.delete(name);
    return { deleted: name };
  }
}

function info(name: string, texts: string[]): DatasetInfo {
  return { name, texts: texts.length, characters: texts.reduce((n, t) => n + t.length, 0) };
}
