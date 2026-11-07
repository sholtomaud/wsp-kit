/**
 * Model files: llama2.c checkpoints (what the starter models ship as), the
 * .wspk checkpoint format (any shape, after grafts), and seed models.
 */
import { defaultHeadKv, tensorSpecs, validateConfig, type ModelConfig } from './types.ts';
import { gaussian } from './random.ts';

export type Weights = Map<string, Float32Array>;

export interface LoadedWeights {
  config: ModelConfig;
  weights: Weights;
}

/**
 * llama2.c's run.c format: seven int32s (dim, hidden_dim, n_layers, n_heads,
 * n_kv_heads, vocab_size, seq_len; a negative vocab_size means a separate
 * classifier), then the tensors grouped by kind across layers.
 */
export function parseLlama2c(buffer: ArrayBuffer): LoadedWeights {
  if (buffer.byteLength < 28) throw new Error('Not a llama2.c checkpoint: file is too short');
  const h = new Int32Array(buffer, 0, 7);
  const [dim, hiddenDim, nLayers, nHeads, nKvHeads, rawVocab, seqLen] = h;
  const vocabSize = Math.abs(rawVocab);
  const sharedClassifier = rawVocab > 0;
  for (const [k, v] of Object.entries({ dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize, seqLen })) {
    if (!(v > 0 && v < 1 << 24)) throw new Error(`Not a llama2.c checkpoint: ${k} = ${v}`);
  }
  if (dim % nHeads) throw new Error('Not a llama2.c checkpoint: dim is not a multiple of n_heads');
  const headSize = dim / nHeads;
  const config: ModelConfig = {
    dim,
    headSize,
    nKvHeads,
    vocabSize,
    seqLen,
    sharedClassifier,
    normEps: 1e-5,
    ropeTheta: 10000,
    layers: Array.from({ length: nLayers }, () => ({
      nHeads,
      headKv: defaultHeadKv(nHeads, nKvHeads),
      hidden: hiddenDim,
    })),
  };
  validateConfig(config);
  const kv = nKvHeads * headSize;
  let offset = 28;
  const take = (n: number) => {
    const end = offset + n * 4;
    if (end > buffer.byteLength) throw new Error('llama2.c checkpoint is truncated');
    const out = new Float32Array(buffer.slice(offset, end));
    offset = end;
    return out;
  };
  const perLayer = (n: number) => {
    const all = take(n * nLayers);
    return Array.from({ length: nLayers }, (_, l) => all.subarray(l * n, (l + 1) * n));
  };
  const weights: Weights = new Map();
  weights.set('tok_embeddings', take(vocabSize * dim));
  const groups: [string, number][] = [
    ['attn_norm', dim],
    ['attn.wq', dim * dim],
    ['attn.wk', kv * dim],
    ['attn.wv', kv * dim],
    ['attn.wo', dim * dim],
    ['mlp_norm', dim],
    ['mlp.gate', hiddenDim * dim],
    ['mlp.down', dim * hiddenDim],
    ['mlp.up', hiddenDim * dim],
  ];
  for (const [suffix, n] of groups) {
    perLayer(n).forEach((t, l) => weights.set(`layer_${l}.${suffix}`, t));
  }
  weights.set('norm', take(dim));
  // Legacy precomputed RoPE tables (freq_cis_real, freq_cis_imag); run.c skips them.
  offset += seqLen * headSize * 4;
  if (!sharedClassifier) weights.set('lm_head', take(vocabSize * dim));
  checkWeights(config, weights);
  return { config, weights };
}

/** Writes a llama2.c checkpoint; only for models whose layers all share one shape. */
export function toLlama2c({ config, weights }: LoadedWeights): ArrayBuffer {
  const l0 = config.layers[0];
  const uniform = config.layers.every(
    (l) => l.nHeads === l0.nHeads && l.hidden === l0.hidden && l.headKv.every((kv, h) => kv === l0.headKv[h])
  );
  if (!uniform || !l0 || l0.nHeads * config.headSize !== config.dim) {
    throw new Error('Only models with identical, unmodified layers can be written as llama2.c checkpoints');
  }
  const parts: Float32Array[] = [weights.get('tok_embeddings')!];
  for (const suffix of ['attn_norm', 'attn.wq', 'attn.wk', 'attn.wv', 'attn.wo', 'mlp_norm', 'mlp.gate', 'mlp.down', 'mlp.up']) {
    config.layers.forEach((_, l) => parts.push(weights.get(`layer_${l}.${suffix}`)!));
  }
  parts.push(weights.get('norm')!, new Float32Array(config.seqLen * config.headSize));
  if (!config.sharedClassifier) parts.push(weights.get('lm_head')!);
  const total = 28 + parts.reduce((n, p) => n + p.byteLength, 0);
  const out = new ArrayBuffer(total);
  new Int32Array(out, 0, 7).set([
    config.dim,
    l0.hidden,
    config.layers.length,
    l0.nHeads,
    config.nKvHeads,
    config.sharedClassifier ? config.vocabSize : -config.vocabSize,
    config.seqLen,
  ]);
  let o = 28;
  for (const p of parts) {
    new Float32Array(out, o, p.length).set(p);
    o += p.byteLength;
  }
  return out;
}

const WSPK_MAGIC = 0x4b505357; // "WSPK"

export interface WspkMeta {
  /** Which model this came from (the tokenizer it needs). */
  modelId: string;
  createdAt: string;
  note?: string;
}

/** WSPK: magic, version, header length, JSON header (config, tensor table, meta), f32 data at 16-byte offsets. */
export function serializeWspk(config: ModelConfig, weights: Weights, meta: WspkMeta): ArrayBuffer {
  const table: { name: string; shape: [number, number]; offset: number }[] = [];
  let data = 0;
  for (const spec of tensorSpecs(config)) {
    const t = weights.get(spec.name);
    if (!t) throw new Error(`Missing tensor ${spec.name}`);
    table.push({ name: spec.name, shape: spec.shape, offset: data });
    data += Math.ceil(t.byteLength / 16) * 16;
  }
  const header = new TextEncoder().encode(JSON.stringify({ config, tensors: table, meta }));
  const headerEnd = Math.ceil((12 + header.length) / 16) * 16;
  const out = new ArrayBuffer(headerEnd + data);
  const view = new DataView(out);
  view.setUint32(0, WSPK_MAGIC, true);
  view.setUint32(4, 1, true);
  view.setUint32(8, header.length, true);
  new Uint8Array(out, 12, header.length).set(header);
  for (const entry of table) {
    const t = weights.get(entry.name)!;
    new Float32Array(out, headerEnd + entry.offset, t.length).set(t);
  }
  return out;
}

export function isWspk(buffer: ArrayBuffer): boolean {
  return buffer.byteLength >= 12 && new DataView(buffer).getUint32(0, true) === WSPK_MAGIC;
}

export function parseWspk(buffer: ArrayBuffer): LoadedWeights & { meta: WspkMeta } {
  if (!isWspk(buffer)) throw new Error('Not a .wspk checkpoint');
  const view = new DataView(buffer);
  const version = view.getUint32(4, true);
  if (version !== 1) throw new Error(`Unsupported .wspk version ${version}`);
  const len = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 12, len)));
  const headerEnd = Math.ceil((12 + len) / 16) * 16;
  const config = header.config as ModelConfig;
  validateConfig(config);
  const weights: Weights = new Map();
  for (const t of header.tensors as { name: string; shape: [number, number]; offset: number }[]) {
    const n = t.shape[0] * t.shape[1];
    weights.set(t.name, new Float32Array(buffer.slice(headerEnd + t.offset, headerEnd + t.offset + n * 4)));
  }
  checkWeights(config, weights);
  return { config, weights, meta: header.meta };
}

export function checkWeights(config: ModelConfig, weights: Weights) {
  for (const spec of tensorSpecs(config)) {
    const t = weights.get(spec.name);
    const n = spec.shape[0] * spec.shape[1];
    if (!t) throw new Error(`Missing tensor ${spec.name}`);
    if (t.length !== n) throw new Error(`${spec.name} has ${t.length} values, expected ${n}`);
  }
}

export interface SeedOptions {
  dim?: number;
  layers?: number;
  heads?: number;
  kvHeads?: number;
  hidden?: number;
  seqLen?: number;
  vocabSize?: number;
  /** Std of the random init; 0 gives all-zero matrices (norm gains stay 1). */
  std?: number;
  seed?: number;
}

/**
 * A fresh network to grow from: random (or zero) matrices, unit norm gains.
 * With heads = 0 and hidden = 0 it has no attention or MLP capacity at all,
 * only an embedding and a classifier; grafts add the rest.
 */
export function seedModel(o: SeedOptions = {}): LoadedWeights {
  const heads = o.heads ?? 4;
  const kvHeads = o.kvHeads ?? Math.max(1, heads);
  const dim = o.dim ?? 64;
  const config: ModelConfig = {
    dim,
    headSize: o.heads ? dim / heads : dim / 4,
    nKvHeads: kvHeads,
    vocabSize: o.vocabSize ?? 259,
    seqLen: o.seqLen ?? 256,
    sharedClassifier: true,
    normEps: 1e-5,
    ropeTheta: 10000,
    layers: Array.from({ length: o.layers ?? 2 }, () => ({
      nHeads: heads,
      headKv: defaultHeadKv(heads, kvHeads),
      hidden: o.hidden ?? 172,
    })),
  };
  validateConfig(config);
  const std = o.std ?? 0.02;
  let seed = o.seed ?? 1;
  const weights: Weights = new Map();
  for (const spec of tensorSpecs(config)) {
    const n = spec.shape[0] * spec.shape[1];
    const t = new Float32Array(n);
    if (spec.name.endsWith('norm')) t.fill(1);
    else if (std > 0) for (let i = 0; i < n; i++) t[i] = std * gaussian(seed, i);
    seed++;
    weights.set(spec.name, t);
  }
  return { config, weights };
}
