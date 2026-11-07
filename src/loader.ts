/**
 * Starter models and how they reach the browser: downloaded once (from
 * Hugging Face and the llama2.c repository, both of which allow cross-origin
 * fetches), then kept in OPFS so later visits load from disk.
 */
import type { Backend } from './backend.ts';
import { CpuBackend } from './cpu/backend.ts';
import { WebGpuBackend } from './gpu/backend.ts';
import { isWspk, parseLlama2c, parseWspk, seedModel, type LoadedWeights, type SeedOptions } from './model/formats.ts';
import { Model } from './model/model.ts';
import { Tokenizer } from './model/tokenizer.ts';
import { defaultStore, type CheckpointStore } from './state/store.ts';
import { WSP } from './wsp.ts';

export interface StarterModel {
  id: string;
  label: string;
  description: string;
  model: string;
  tokenizer: string;
  vocabSize: number;
  approxBytes: number;
}

const HF = 'https://huggingface.co/karpathy/tinyllamas/resolve/main';
const LLAMA2C_TOKENIZER = 'https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin';

/** Karpathy's TinyStories models (MIT): Llama-architecture, trained on children's stories. */
export const STARTERS: StarterModel[] = [
  {
    id: 'stories260K',
    label: 'TinyStories 260K',
    description: '260K parameters, 5 layers, 512-token vocabulary. Small enough to read most of; loads in a second.',
    model: `${HF}/stories260K/stories260K.bin`,
    tokenizer: `${HF}/stories260K/tok512.bin`,
    vocabSize: 512,
    approxBytes: 1.1e6,
  },
  {
    id: 'stories15M',
    label: 'TinyStories 15M',
    description: '15M parameters, 6 layers, Llama 2 tokenizer (32K). Writes coherent short stories.',
    model: `${HF}/stories15M.bin`,
    tokenizer: LLAMA2C_TOKENIZER,
    vocabSize: 32000,
    approxBytes: 61e6,
  },
  {
    id: 'stories42M',
    label: 'TinyStories 42M',
    description: '42M parameters, 8 layers. Better stories, slower edits.',
    model: `${HF}/stories42M.bin`,
    tokenizer: LLAMA2C_TOKENIZER,
    vocabSize: 32000,
    approxBytes: 167e6,
  },
];

export interface LoadedModel extends LoadedWeights {
  id: string;
  tokenizer: Tokenizer;
}

export type Progress = (p: { file: string; loaded: number; total: number; cached: boolean }) => void;

async function opfsCacheDir(): Promise<FileSystemDirectoryHandle | undefined> {
  try {
    let d = await navigator.storage.getDirectory();
    for (const part of ['wsp-kit', 'models']) d = await d.getDirectoryHandle(part, { create: true });
    return d;
  } catch {
    return undefined;
  }
}

/** Fetches a file with progress, through the OPFS cache when there is one. */
export async function fetchCached(url: string, cacheName: string, onProgress?: Progress, signal?: AbortSignal): Promise<ArrayBuffer> {
  const dir = await opfsCacheDir();
  if (dir) {
    try {
      const file = await (await dir.getFileHandle(cacheName)).getFile();
      if (file.size > 0) {
        onProgress?.({ file: cacheName, loaded: file.size, total: file.size, cached: true });
        return file.arrayBuffer();
      }
    } catch {
      /* not cached */
    }
  }
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Downloading ${url} failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  let buffer: ArrayBuffer;
  if (res.body && onProgress) {
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress({ file: cacheName, loaded, total, cached: false });
    }
    const all = new Uint8Array(loaded);
    let o = 0;
    for (const c of chunks) {
      all.set(c, o);
      o += c.length;
    }
    buffer = all.buffer;
  } else buffer = await res.arrayBuffer();
  if (dir) {
    try {
      const handle = await dir.getFileHandle(cacheName, { create: true });
      if ('createWritable' in handle) {
        const w = await handle.createWritable();
        await w.write(buffer);
        await w.close();
      }
    } catch {
      /* caching is best-effort */
    }
  }
  return buffer;
}

export async function loadStarter(id: string, onProgress?: Progress, signal?: AbortSignal): Promise<LoadedModel> {
  const s = STARTERS.find((m) => m.id === id);
  if (!s) throw new Error(`No starter model "${id}"`);
  const [modelBuf, tokBuf] = await Promise.all([
    fetchCached(s.model, `${s.id}.bin`, onProgress, signal),
    fetchCached(s.tokenizer, `${s.id}.tokenizer.bin`, onProgress, signal),
  ]);
  const { config, weights } = parseLlama2c(modelBuf);
  return { id: s.id, config, weights, tokenizer: Tokenizer.fromLlama2c(tokBuf, config.vocabSize) };
}

/** A user's own file: a llama2.c checkpoint (with its tokenizer) or a .wspk export. */
export function loadFile(model: ArrayBuffer, tokenizer: ArrayBuffer | undefined, name: string): LoadedModel {
  if (isWspk(model)) {
    const { config, weights, meta } = parseWspk(model);
    const tok = tokenizer ? Tokenizer.fromLlama2c(tokenizer, config.vocabSize) : config.vocabSize === 259 ? Tokenizer.byteLevel() : undefined;
    if (!tok) throw new Error(`${name} needs its tokenizer file (vocabulary of ${config.vocabSize})`);
    return { id: meta.modelId, config, weights, tokenizer: tok };
  }
  const { config, weights } = parseLlama2c(model);
  if (!tokenizer) throw new Error(`${name} is a llama2.c checkpoint: choose its tokenizer.bin too`);
  return { id: name.replace(/\.[^.]*$/, ''), config, weights, tokenizer: Tokenizer.fromLlama2c(tokenizer, config.vocabSize) };
}

/** A fresh network with a byte-level vocabulary, to grow by grafting. */
export function loadSeed(options: SeedOptions = {}): LoadedModel {
  const tokenizer = Tokenizer.byteLevel();
  const { config, weights } = seedModel({ ...options, vocabSize: tokenizer.vocabSize });
  const id = `seed-${config.dim}d-${config.layers.length}L-${options.seed ?? 1}`;
  return { id, config, weights, tokenizer };
}

export interface CreateWspOptions {
  backend?: 'auto' | 'webgpu' | 'cpu' | Backend;
  store?: CheckpointStore;
}

export async function createBackend(kind: CreateWspOptions['backend'] = 'auto'): Promise<Backend> {
  if (kind && typeof kind === 'object') return kind;
  if (kind === 'cpu') return new CpuBackend();
  try {
    return await WebGpuBackend.create();
  } catch (e) {
    if (kind === 'webgpu') throw e;
    console.warn(`wsp-kit: ${(e as Error).message}; using the CPU reference backend`);
    return new CpuBackend();
  }
}

export async function createWsp(loaded: LoadedModel, options: CreateWspOptions = {}): Promise<WSP> {
  const backend = await createBackend(options.backend);
  const model = new Model(backend, loaded.tokenizer, loaded.config, loaded.weights, loaded.id);
  return new WSP(model, { store: options.store ?? defaultStore(['wsp-kit', 'checkpoints', loaded.id]) });
}
