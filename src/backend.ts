/**
 * Where the tensors live and the operations wsp-kit needs on them. The WebGPU
 * backend keeps every tensor in VRAM and runs each operation as a compute
 * shader; the CPU backend is the reference the GPU is tested against (and
 * what runs in Node).
 */
import type { ModelConfig, Shape } from './model/types.ts';

export interface Region {
  rowStart: number;
  /** Exclusive. */
  rowEnd: number;
  colStart: number;
  /** Exclusive. */
  colEnd: number;
}

export type Fill =
  | { kind: 'constant'; value: number }
  | { kind: 'gaussian'; mean: number; std: number; seed: number }
  | { kind: 'add_gaussian'; std: number; seed: number }
  | { kind: 'scale'; factor: number };

export interface Stats {
  count: number;
  /** Over finite values. */
  sum: number;
  sumSq: number;
  min: number;
  max: number;
  zeros: number;
  nonFinite: number;
}

export interface Hits {
  /** Flat indices, in no particular order. */
  indices: Uint32Array;
  values: Float32Array;
  /** All matches, including those past the cap. */
  total: number;
}

export interface RunOptions {
  /** Next-token targets for each fed token (-1: none); their NLL is summed. */
  targets?: number[];
  /** Return the logits after the last token. */
  logits?: boolean;
  /** Accumulate |activation| of every MLP hidden unit. */
  track?: boolean;
}

export interface RunResult {
  logits?: Float32Array;
  nll?: { sum: number; count: number };
}

/** A forward pass bound to the current tensors, with its own KV cache. */
export interface Runner {
  /** Feeds tokens at positions start, start+1, … */
  run(tokens: number[], start: number, options?: RunOptions): Promise<RunResult>;
  /** Mean |activation| per MLP hidden unit, per layer, since the last reset. */
  activations(): Promise<{ tokens: number; layers: Float32Array[] }>;
  resetActivations(): void;
  destroy(): void;
}

/** A tensor, or a block of one, that training may change. */
export interface TrainTarget {
  name: string;
  region: Region;
}

export interface AdamOptions {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  /** Global gradient-norm clip. */
  clip: number;
}

/** One training window: input tokens and next-token targets (-1 = not scored), equal length. */
export interface Sequence {
  tokens: number[];
  targets: number[];
}

/**
 * Batched forward and backward passes over fixed-length windows, with AdamW
 * updates confined to the trainable regions. Everything else stays frozen.
 */
export interface Trainer {
  /** Window length every sequence is padded to. */
  readonly length: number;
  /** Summed NLL and scored-token count, forward only. */
  loss(sequences: Sequence[]): Promise<{ sum: number; count: number }>;
  /** Zeroes the gradients, then runs forward and backward over the batch (loss = mean over its scored tokens). */
  accumulate(sequences: Sequence[]): Promise<{ sum: number; count: number }>;
  /** The accumulated gradients of the trainable tensors (whole tensors). */
  gradients(): Promise<Map<string, Float32Array>>;
  /** One AdamW step on the trainable regions. Skipped when a gradient is not finite. */
  update(options: AdamOptions): Promise<{ gradNorm: number; skipped: boolean }>;
  /**
   * Runs one window forward and returns layer `layer`'s activations, each row-major
   * [tokens, width]: the MLP key (input to mlp.down), the normalized MLP input
   * (what mlp.gate and mlp.up read), and the residual stream after the layer.
   */
  capture(sequence: Sequence, layer: number): Promise<{ key: Float32Array; mlpInput: Float32Array; residual: Float32Array }>;
  /** After accumulate(): dLoss/d(output of the probe layer), [tokens, dim]. Needs the probeLayer option. */
  probe(): Promise<Float32Array>;
  destroy(): void;
}

export interface TrainerOptions {
  /** Keep the gradient at this layer's output (for computed edits). */
  probeLayer?: number;
}

export interface Backend {
  readonly kind: 'webgpu' | 'cpu';
  readonly description: string;
  has(name: string): boolean;
  shape(name: string): Shape;
  names(): string[];
  /** Creates (or replaces) a tensor; zero-filled without data. */
  alloc(name: string, shape: Shape, data?: Float32Array): void;
  free(name: string): void;
  /**
   * Copies src into dst. A dst of the same shape keeps its buffer (so a running
   * forward pass still sees it); otherwise dst is (re)allocated.
   */
  copy(src: string, dst: string): Promise<void>;
  rename(from: string, to: string): void;
  /** W · v for a tensor W and an uploaded vector v. */
  matvec(name: string, vector: Float32Array): Promise<Float32Array>;
  /** Copies a block of src into dst at (dstRow, dstCol). */
  copyRegion(src: string, dst: string, region: Region, dstRow: number, dstCol: number): Promise<void>;
  read(name: string, offset?: number, length?: number): Promise<Float32Array>;
  write(name: string, data: Float32Array): Promise<void>;
  stats(name: string): Promise<Stats>;
  rowNorms(name: string): Promise<Float32Array>;
  /** Elements with |x − mean| ≥ z·std, at most `cap` of them. */
  outliers(name: string, mean: number, std: number, z: number, cap: number): Promise<Hits>;
  /** Writes (or adds) values at flat indices; indices must be unique. */
  scatter(name: string, indices: Uint32Array, values: Float32Array, mode: 'set' | 'add'): Promise<void>;
  fill(name: string, region: Region, fill: Fill): Promise<void>;
  createRunner(config: ModelConfig): Runner;
  /** A trainer over windows of `length` tokens; with no targets it only evaluates losses. */
  createTrainer(config: ModelConfig, length: number, targets: TrainTarget[], options?: TrainerOptions): Trainer;
  /** W += scale · a ⊗ b (a rank-one update; a has one entry per row, b one per column). */
  addOuter(name: string, a: Float32Array, b: Float32Array, scale: number): Promise<void>;
  /** Resolves when all queued work has finished. */
  flush(): Promise<void>;
  /** Releases every tensor (and the GPU device). */
  destroy?(): void;
}

export const fullRegion = (shape: Shape): Region => ({ rowStart: 0, rowEnd: shape[0], colStart: 0, colEnd: shape[1] });

export function checkRegion(shape: Shape, r: Region, what = 'region'): void {
  const ok =
    Number.isInteger(r.rowStart) &&
    Number.isInteger(r.rowEnd) &&
    Number.isInteger(r.colStart) &&
    Number.isInteger(r.colEnd) &&
    r.rowStart >= 0 &&
    r.colStart >= 0 &&
    r.rowStart <= r.rowEnd &&
    r.colStart <= r.colEnd &&
    r.rowEnd <= shape[0] &&
    r.colEnd <= shape[1];
  if (!ok) {
    throw new RangeError(
      `${what} rows ${r.rowStart}..${r.rowEnd}, cols ${r.colStart}..${r.colEnd} is outside the tensor's ${shape[0]}×${shape[1]}`
    );
  }
}
