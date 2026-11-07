/**
 * The network wsp-kit edits: a Llama-style decoder (RMSNorm, rotary attention
 * with grouped KV heads, SwiGLU MLP), as in Karpathy's llama2.c.
 *
 * Unlike llama2.c, each layer keeps its own width: an LLM can graft attention
 * heads or MLP neurons onto one layer without touching the others, so heads
 * and hidden units are per-layer, and each query head names the KV head it reads.
 */

export interface LayerConfig {
  /** Query heads in this layer. */
  nHeads: number;
  /** For each query head, the KV head it attends with (grouped-query attention). */
  headKv: number[];
  /** MLP hidden units (SwiGLU width). */
  hidden: number;
}

export interface ModelConfig {
  /** Residual stream width. */
  dim: number;
  /** Per-head width, shared by every head in the model. */
  headSize: number;
  nKvHeads: number;
  vocabSize: number;
  /** Longest context the KV cache holds. */
  seqLen: number;
  /** The classifier reuses the token embeddings (no separate lm_head). */
  sharedClassifier: boolean;
  normEps: number;
  ropeTheta: number;
  layers: LayerConfig[];
}

/** [rows, cols]; a vector is [1, n]. Matrices are row-major, rows = outputs. */
export type Shape = [number, number];

export interface TensorSpec {
  name: string;
  shape: Shape;
  /** What it is, for an LLM reading the model overview. */
  role: string;
}

export const kvDim = (c: ModelConfig) => c.nKvHeads * c.headSize;
export const qDim = (c: ModelConfig, layer: number) => c.layers[layer].nHeads * c.headSize;

/** Every tensor of a model, in a fixed order (the checkpoint order). */
export function tensorSpecs(c: ModelConfig): TensorSpec[] {
  const specs: TensorSpec[] = [
    { name: 'tok_embeddings', shape: [c.vocabSize, c.dim], role: 'token embedding table (row = token id)' },
  ];
  c.layers.forEach((layer, l) => {
    const q = layer.nHeads * c.headSize;
    const p = `layer_${l}`;
    specs.push(
      { name: `${p}.attn_norm`, shape: [1, c.dim], role: 'RMSNorm gain before attention' },
      { name: `${p}.attn.wq`, shape: [q, c.dim], role: `query projection (${layer.nHeads} heads × ${c.headSize} rows each)` },
      { name: `${p}.attn.wk`, shape: [kvDim(c), c.dim], role: `key projection (${c.nKvHeads} KV heads × ${c.headSize} rows)` },
      { name: `${p}.attn.wv`, shape: [kvDim(c), c.dim], role: `value projection (${c.nKvHeads} KV heads × ${c.headSize} rows)` },
      { name: `${p}.attn.wo`, shape: [c.dim, q], role: 'attention output projection (column block per head)' },
      { name: `${p}.mlp_norm`, shape: [1, c.dim], role: 'RMSNorm gain before the MLP' },
      { name: `${p}.mlp.gate`, shape: [layer.hidden, c.dim], role: 'MLP gate (w1); row = hidden neuron' },
      { name: `${p}.mlp.up`, shape: [layer.hidden, c.dim], role: 'MLP up projection (w3); row = hidden neuron' },
      { name: `${p}.mlp.down`, shape: [c.dim, layer.hidden], role: 'MLP down projection (w2); column = hidden neuron' }
    );
  });
  specs.push({ name: 'norm', shape: [1, c.dim], role: 'final RMSNorm gain' });
  if (!c.sharedClassifier) {
    specs.push({ name: 'lm_head', shape: [c.vocabSize, c.dim], role: 'output classifier (row = token id)' });
  }
  return specs;
}

/** The tensor the classifier reads. */
export const classifierName = (c: ModelConfig) => (c.sharedClassifier ? 'tok_embeddings' : 'lm_head');

/** Even grouping of query heads onto KV heads, as llama2.c does it. */
export function defaultHeadKv(nHeads: number, nKvHeads: number): number[] {
  const group = Math.max(1, Math.floor(nHeads / nKvHeads));
  return Array.from({ length: nHeads }, (_, h) => Math.min(nKvHeads - 1, Math.floor(h / group)));
}

export function cloneConfig(c: ModelConfig): ModelConfig {
  return { ...c, layers: c.layers.map((l) => ({ ...l, headKv: [...l.headKv] })) };
}

export function parameterCount(c: ModelConfig): number {
  return tensorSpecs(c).reduce((n, s) => n + s.shape[0] * s.shape[1], 0);
}

export function validateConfig(c: ModelConfig): void {
  const ints: (keyof ModelConfig)[] = ['dim', 'headSize', 'nKvHeads', 'vocabSize', 'seqLen'];
  for (const k of ints) {
    const v = c[k] as number;
    if (!Number.isInteger(v) || v <= 0) throw new Error(`Model config: ${k} must be a positive integer, got ${v}`);
  }
  if (c.headSize % 2) throw new Error('Model config: headSize must be even (rotary embeddings rotate pairs)');
  c.layers.forEach((l, i) => {
    if (!Number.isInteger(l.nHeads) || l.nHeads < 0) throw new Error(`layer_${i}: bad head count ${l.nHeads}`);
    if (!Number.isInteger(l.hidden) || l.hidden < 0) throw new Error(`layer_${i}: bad hidden width ${l.hidden}`);
    if (l.headKv.length !== l.nHeads) throw new Error(`layer_${i}: headKv must list one KV head per query head`);
    for (const kv of l.headKv) {
      if (!Number.isInteger(kv) || kv < 0 || kv >= c.nKvHeads) throw new Error(`layer_${i}: KV head ${kv} out of range`);
    }
  });
}
