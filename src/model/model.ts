/**
 * A loaded network: its shape, its tensors (in a backend), its tokenizer, and
 * a forward pass that is rebuilt whenever the shape changes.
 */
import type { Backend, Runner } from '../backend.ts';
import { rng } from './random.ts';
import { BOS, EOS, type Tokenizer } from './tokenizer.ts';
import { checkWeights, type Weights } from './formats.ts';
import { cloneConfig, tensorSpecs, validateConfig, type ModelConfig, type TensorSpec } from './types.ts';

export interface GenerateOptions {
  maxTokens?: number;
  /** 0 = greedy. */
  temperature?: number;
  seed?: number;
  stopAtEos?: boolean;
}

export class Model {
  readonly backend: Backend;
  readonly tokenizer: Tokenizer;
  /** Identifies the tokenizer/vocabulary checkpoints must match. */
  readonly id: string;
  private _config: ModelConfig;
  private runner?: Runner;

  constructor(backend: Backend, tokenizer: Tokenizer, config: ModelConfig, weights: Weights, id: string) {
    if (tokenizer.vocabSize !== config.vocabSize) {
      throw new Error(`Tokenizer has ${tokenizer.vocabSize} pieces but the model's vocabulary is ${config.vocabSize}`);
    }
    this.backend = backend;
    this.tokenizer = tokenizer;
    this.id = id;
    this._config = cloneConfig(config);
    this.load(config, weights);
  }

  get config(): ModelConfig {
    return this._config;
  }

  /** Replaces every tensor (a restore, or a new model of the same vocabulary). */
  load(config: ModelConfig, weights: Weights) {
    validateConfig(config);
    checkWeights(config, weights);
    if (config.vocabSize !== this.tokenizer.vocabSize) throw new Error('Checkpoint vocabulary does not match the tokenizer');
    for (const name of this.backend.names()) this.backend.free(name);
    for (const spec of tensorSpecs(config)) this.backend.alloc(spec.name, spec.shape, weights.get(spec.name));
    this._config = cloneConfig(config);
    this.invalidate();
  }

  /** Call after a change of shape or of which buffers hold the tensors. */
  setConfig(config: ModelConfig) {
    validateConfig(config);
    this._config = cloneConfig(config);
    this.invalidate();
  }

  invalidate() {
    this.runner?.destroy();
    this.runner = undefined;
  }

  forward(): Runner {
    this.runner ??= this.backend.createRunner(this._config);
    return this.runner;
  }

  specs(): TensorSpec[] {
    return tensorSpecs(this._config);
  }

  spec(name: string): TensorSpec {
    const spec = this.specs().find((s) => s.name === name);
    if (!spec) {
      const close = this.specs()
        .map((s) => s.name)
        .filter((n) => n.includes(name) || name.includes(n.split('.').pop()!))
        .slice(0, 6);
      throw new Error(`No tensor named "${name}".${close.length ? ` Did you mean: ${close.join(', ')}?` : ''} Call model_overview for the list.`);
    }
    return spec;
  }

  /** Every tensor read back from the backend (for checkpoints). */
  async weights(): Promise<Weights> {
    const out: Weights = new Map();
    for (const spec of this.specs()) out.set(spec.name, await this.backend.read(spec.name));
    return out;
  }

  /** Mean next-token NLL of `continuation` given `context` (teacher forcing). */
  async nll(context: number[], continuation: number[]): Promise<{ loss: number; tokens: number }> {
    const all = [...context, ...continuation].slice(0, this._config.seqLen + 1);
    if (all.length < 2) return { loss: NaN, tokens: 0 };
    const inputs = all.slice(0, -1);
    const targets = inputs.map((_, i) => (i + 1 >= context.length ? all[i + 1] : -1));
    const { nll } = await this.forward().run(inputs, 0, { targets });
    return { loss: nll!.count ? nll!.sum / nll!.count : NaN, tokens: nll!.count };
  }

  /** Runs texts through the network with activation tracking on. */
  async track(texts: string[]): Promise<{ tokens: number; layers: Float32Array[] }> {
    const runner = this.forward();
    runner.resetActivations();
    for (const text of texts) {
      const tokens = this.tokenizer.encode(text).slice(0, this._config.seqLen);
      await runner.run(tokens, 0, { track: true });
    }
    const acts = await runner.activations();
    runner.resetActivations();
    return acts;
  }

  async generate(prompt: string, o: GenerateOptions = {}): Promise<{ text: string; tokens: number[] }> {
    const maxTokens = o.maxTokens ?? 64;
    const random = rng(o.seed ?? 42);
    const runner = this.forward();
    let tokens = this.tokenizer.encode(prompt);
    if (tokens.length >= this._config.seqLen) tokens = tokens.slice(0, this._config.seqLen - 1);
    const out: number[] = [];
    let pos = 0;
    let feed = tokens;
    while (out.length < maxTokens && pos + feed.length <= this._config.seqLen) {
      const { logits } = await runner.run(feed, pos, { logits: true });
      pos += feed.length;
      const next = sample(logits!, o.temperature ?? 0, random);
      if ((o.stopAtEos ?? true) && (next === EOS || next === BOS)) break;
      out.push(next);
      feed = [next];
    }
    return { text: this.tokenizer.decode(out, tokens[tokens.length - 1]), tokens: out };
  }

  /** The most likely next tokens after a prompt. */
  async topNext(prompt: string, k = 5): Promise<{ token: number; piece: string; p: number }[]> {
    const tokens = this.tokenizer.encode(prompt).slice(0, this._config.seqLen);
    const { logits } = await this.forward().run(tokens, 0, { logits: true });
    const probs = softmax(logits!);
    return [...probs.keys()]
      .sort((a, b) => probs[b] - probs[a])
      .slice(0, k)
      .map((token) => ({ token, piece: this.tokenizer.describe(token), p: probs[token] }));
  }
}

export function softmax(logits: Float32Array, temperature = 1): Float32Array {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp((logits[i] - max) / temperature);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

function sample(logits: Float32Array, temperature: number, random: () => number): number {
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const probs = softmax(logits, temperature);
  let r = random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}
