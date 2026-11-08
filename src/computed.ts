/**
 * Computed edits: the master states a behaviour, wsp-kit reads what the
 * student represents at that point and writes the weights that produce it.
 * No training loop: each edit is a closed-form rank-one change, scaled by a
 * short search over its strength.
 *
 * insertAssociation is ROME-style (Meng et al., 2022): the key is the input to
 * one layer's mlp.down at the last prompt token; the edit
 *     ΔW = δ ⊗ C⁻¹k / (kᵀC⁻¹k)
 * changes that layer's output by exactly δ for this key, and as little as
 * possible for the keys of the texts to preserve (their covariance is C).
 * δ points along the negative gradient of the target token's loss at that
 * layer's output, so it accounts for every later layer.
 */
import type { Sequence, Trainer } from './backend.ts';
import type { WSP } from './wsp.ts';

export interface AssociationOptions {
  /** The context, e.g. "The capital of France is". */
  prompt: string;
  /** The continuation it should produce, e.g. " Paris" (one edit per token). */
  target: string;
  /** Which layer's MLP holds the association. Default: the middle layer. */
  layer?: number;
  /** Texts whose behaviour should not change; their keys shape the edit (C). */
  preserve?: string[];
  /** Probability to reach for each target token. Default 0.9. */
  goal?: number;
}

export interface WireOptions {
  layer: number;
  /** Which hidden neuron to rewire; omitted, a new one is grafted. */
  neuron?: number;
  /** The context the neuron should fire on (its last token). */
  prompt: string;
  /** The next token it should promote there (the first token of this text). */
  token: string;
  preserve?: string[];
  goal?: number;
}

const r4 = (x: number) => (Number.isFinite(x) ? Number(x.toPrecision(4)) : x);
function norm(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}
/** Edit strengths tried, as multiples of the residual stream's norm. */
const STRENGTHS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8];

/** Splits prompt + continuation into context ids and continuation ids (BPE may merge across the seam). */
export function splitTokens(wsp: WSP, prompt: string, continuation: string) {
  const tok = wsp.model.tokenizer;
  const context = tok.encode(prompt);
  const all = tok.encode(prompt + continuation);
  let k = 0;
  while (k < context.length && k < all.length && context[k] === all[k]) k++;
  return { context: all.slice(0, k), continuation: all.slice(k) };
}

/** A window that scores only its last position, predicting `next`. */
const scoreLast = (ids: number[], next: number): Sequence => ({ tokens: ids, targets: ids.map((_, i) => (i === ids.length - 1 ? next : -1)) });

/** Solves (A + λI)x = b for a symmetric positive-definite A by Cholesky. */
export function solveSpd(A: Float64Array, b: Float64Array, n: number): Float64Array {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) L[i * n + i] = Math.sqrt(Math.max(s, 1e-12));
      else L[i * n + j] = s / L[j * n + j];
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

class Session {
  readonly wsp: WSP;
  readonly layer: number;
  readonly length: number;
  readonly scorer: Trainer;
  readonly prober: Trainer;

  constructor(wsp: WSP, layer: number, main: number[], preserve: number[][]) {
    const c = wsp.model.config;
    this.wsp = wsp;
    this.layer = layer;
    // Preserve texts are read in windows of at most 256 tokens.
    this.length = Math.min(c.seqLen, Math.max(2, main.length, ...preserve.map((t) => Math.min(256, t.length))));
    this.scorer = wsp.backend.createTrainer(c, this.length, []);
    this.prober = wsp.backend.createTrainer(c, this.length, [], { probeLayer: layer });
  }

  async prob(ids: number[], next: number): Promise<number> {
    const r = await this.scorer.loss([scoreLast(ids, next)]);
    return Math.exp(-r.sum);
  }

  /** The layer's key, MLP input and residual at the last position, and the target loss's gradient there. */
  async read(ids: number[], next: number) {
    const c = this.wsp.model.config;
    const seq = scoreLast(ids, next);
    const pos = ids.length - 1;
    const cap = await this.prober.capture(seq, this.layer);
    await this.prober.accumulate([seq]);
    const probe = await this.prober.probe();
    const h = c.layers[this.layer].hidden;
    return {
      key: cap.key.slice(pos * h, (pos + 1) * h),
      mlpInput: cap.mlpInput.slice(pos * c.dim, (pos + 1) * c.dim),
      residualNorm: norm(cap.residual.subarray(pos * c.dim, (pos + 1) * c.dim)),
      grad: probe.slice(pos * c.dim, (pos + 1) * c.dim),
    };
  }

  /** Mean loss over windows of the given texts. */
  async meanLoss(texts: number[][]): Promise<number> {
    const seqs = texts.filter((t) => t.length >= 2).map((t) => ({ tokens: t.slice(0, -1).slice(0, this.length), targets: t.slice(1).slice(0, this.length) }));
    if (!seqs.length) return NaN;
    const r = await this.scorer.loss(seqs);
    return r.sum / r.count;
  }

  /** Keys at every position of the texts (rows of length hidden). */
  async keys(texts: number[][]): Promise<Float32Array[]> {
    const h = this.wsp.model.config.layers[this.layer].hidden;
    const rows: Float32Array[] = [];
    for (const t of texts) {
      const ids = t.slice(0, this.length);
      const cap = await this.prober.capture({ tokens: ids, targets: ids.map(() => -1) }, this.layer);
      for (let p = 0; p < ids.length && rows.length < 4096; p++) rows.push(cap.key.slice(p * h, (p + 1) * h));
    }
    return rows;
  }

  destroy() {
    this.scorer.destroy();
    this.prober.destroy();
  }
}

/**
 * Tries each strength of `apply` from a backup and keeps the smallest that
 * reaches the goal (else the most probable); returns what it kept.
 */
async function search(
  wsp: WSP,
  tensor: string,
  base: number,
  goal: number,
  measure: () => Promise<number>,
  apply: (strength: number) => Promise<void>
): Promise<{ strength: number; p: number }> {
  const backup = `${tensor}#edit`;
  await wsp.backend.copy(tensor, backup);
  let best = { strength: 0, p: await measure() };
  try {
    for (const s of STRENGTHS) {
      await wsp.backend.copy(backup, tensor);
      await apply(base * s);
      const p = await measure();
      if (p > best.p) best = { strength: base * s, p };
      if (p >= goal) {
        best = { strength: base * s, p };
        break;
      }
    }
    await wsp.backend.copy(backup, tensor);
    if (best.strength) await apply(best.strength);
  } finally {
    wsp.backend.free(backup);
  }
  return best;
}

function defaultLayer(wsp: WSP): number {
  const layers = wsp.model.config.layers;
  const mid = Math.floor((layers.length - 1) / 2);
  // The middle layer, or the nearest one that has MLP neurons.
  const order = [...layers.keys()].sort((a, b) => Math.abs(a - mid) - Math.abs(b - mid));
  return order.find((l) => layers[l].hidden > 0) ?? mid;
}

export async function insertAssociation(wsp: WSP, o: AssociationOptions) {
  const c = wsp.model.config;
  const layer = o.layer ?? defaultLayer(wsp);
  const lc = c.layers[layer];
  if (!lc) throw new RangeError(`No layer ${layer}; the model has ${c.layers.length}`);
  if (!lc.hidden) throw new Error(`layer_${layer} has no MLP neurons to hold an association; graft some first (mutate_graft_node), or pick another layer`);
  const goal = Math.min(Math.max(o.goal ?? 0.9, 0.05), 0.999);
  const { context, continuation } = splitTokens(wsp, o.prompt, o.target);
  if (!continuation.length) throw new Error('The target adds no tokens to the prompt');
  if (context.length + continuation.length > c.seqLen) throw new RangeError('Prompt and target are longer than the context window');
  const tok = wsp.model.tokenizer;
  const preserve = (o.preserve ?? []).map((t) => tok.encode(t, { eos: true }));
  const s = new Session(wsp, layer, [...context, ...continuation], preserve);
  const down = `layer_${layer}.mlp.down`;
  const h = lc.hidden;
  try {
    // C = KᵀK/n + λI over the keys to preserve (identity without any).
    const C = new Float64Array(h * h);
    const keys = preserve.length ? await s.keys(preserve) : [];
    if (keys.length) {
      for (const k of keys) for (let i = 0; i < h; i++) for (let j = 0; j <= i; j++) C[i * h + j] += k[i] * k[j];
      let trace = 0;
      for (let i = 0; i < h; i++) {
        for (let j = 0; j <= i; j++) C[j * h + i] = C[i * h + j] /= keys.length;
        trace += C[i * h + i];
      }
      const lambda = 0.05 * (trace / h) + 1e-6;
      for (let i = 0; i < h; i++) C[i * h + i] += lambda;
    } else for (let i = 0; i < h; i++) C[i * h + i] = 1;
    const preserveBefore = await s.meanLoss(preserve);
    // One fixed protection weight per edited key, relative to C's original scale.
    let trace = 0;
    for (let i = 0; i < h; i++) trace += C[i * h + i];
    const traceScale = trace / h;
    const positions = continuation.map((want, i) => ({ ids: [...context, ...continuation.slice(0, i)], want }));
    const before = await Promise.all(positions.map((q) => s.prob(q.ids, q.want)));
    const rounds = positions.map(() => 0);
    let editNorm = 0;
    // Sweep the target tokens; each edited key joins C so later edits leave it alone (as MEMIT does).
    for (let sweep = 0; sweep < 3; sweep++) {
      let changed = false;
      for (let i = 0; i < positions.length; i++) {
        const { ids, want } = positions[i];
        let p = await s.prob(ids, want);
        for (let round = 0; round < 4 && p < goal; round++) {
          const r = await s.read(ids, want);
          const kn = norm(r.key);
          const gn = norm(r.grad);
          if (kn < 1e-8) throw new Error(`The key at layer_${layer} is zero for this prompt (dead neurons?); try another layer or wire a neuron instead`);
          if (gn < 1e-12) break;
          const u64 = solveSpd(C, Float64Array.from(r.key), h);
          let ku = 0;
          for (let j = 0; j < h; j++) ku += r.key[j] * u64[j];
          const u = Float32Array.from(u64, (x) => x / ku);
          const d = Float32Array.from(r.grad, (x) => -x / gn);
          const kept = await search(wsp, down, Math.max(r.residualNorm, 1e-3), goal, () => s.prob(ids, want), (m) => wsp.backend.addOuter(down, d, u, m));
          if (!kept.strength) break;
          p = kept.p;
          editNorm += kept.strength * norm(u);
          rounds[i]++;
          changed = true;
        }
        // Protect this key from the edits that follow.
        const k = (await s.read(ids, want)).key;
        const kk = k.reduce((a, x) => a + x * x, 0);
        if (kk > 0) {
          const w = (100 * traceScale) / kk;
          for (let a = 0; a < h; a++) for (let b = 0; b < h; b++) C[a * h + b] += w * k[a] * k[b];
        }
      }
      const now = await Promise.all(positions.map((q) => s.prob(q.ids, q.want)));
      if (!changed || now.every((p) => p >= goal)) break;
    }
    const after = await Promise.all(positions.map((q) => s.prob(q.ids, q.want)));
    const steps = positions.map((q, i) => ({ token: tok.describe(q.want), probabilityBefore: r4(before[i]), probabilityAfter: r4(after[i]), rounds: rounds[i] }));
    const preserveAfter = await s.meanLoss(preserve);
    return {
      method: 'rank-one edit of mlp.down (ROME-style); no training loop',
      layer,
      tensor: down,
      tokens: steps,
      reachedGoal: steps.every((t) => t.probabilityAfter >= goal),
      goal,
      editNorm: r4(editNorm),
      ...(preserve.length ? { preserveLossBefore: r4(preserveBefore), preserveLossAfter: r4(preserveAfter), preservedKeys: keys.length } : { note: 'No preserve texts: the edit is specific only to how distinct this key is. Pass preserve_texts to protect other behaviour.' }),
    };
  } finally {
    s.destroy();
  }
}

/**
 * One neuron, one association: its gate and up rows are set to the student's
 * own normalized MLP input at the end of `prompt` (so it fires there), and its
 * down column to the direction that most raises the wanted token.
 */
export async function wireNeuron(wsp: WSP, o: WireOptions) {
  const c = wsp.model.config;
  const lc = c.layers[o.layer];
  if (!lc) throw new RangeError(`No layer ${o.layer}`);
  const goal = Math.min(Math.max(o.goal ?? 0.9, 0.05), 0.999);
  let neuron = o.neuron;
  let grafted = false;
  if (neuron === undefined) {
    const g = await wsp.mutate.graftNode({ layer: o.layer, kind: 'mlp_neurons', count: 1, init: 'zeros', output: 'zeros' });
    neuron = g.newIndices[0];
    grafted = true;
  }
  const h = wsp.model.config.layers[o.layer].hidden;
  if (!(Number.isInteger(neuron) && neuron >= 0 && neuron < h)) throw new RangeError(`layer_${o.layer} has ${h} MLP neurons`);
  const { context, continuation } = splitTokens(wsp, o.prompt, o.token);
  if (!continuation.length) throw new Error('token adds nothing to the prompt');
  const want = continuation[0];
  const tok = wsp.model.tokenizer;
  const preserve = (o.preserve ?? []).map((t) => tok.encode(t, { eos: true }));
  const s = new Session(wsp, o.layer, context, preserve);
  const dim = c.dim;
  const rows = (start: number) => Uint32Array.from({ length: dim }, (_, j) => start + j);
  try {
    const pBefore = await s.prob(context, want);
    // Read: the neuron's input weights are the student's own representation here.
    const r = await s.read(context, want);
    const xn = norm(r.mlpInput);
    if (xn < 1e-8) throw new Error('The MLP input is zero here');
    // gate·x = up·x = 4 at the prompt, so the neuron fires at silu(4)·4 ≈ 16 there.
    const gain = 4 / (xn * xn);
    const read = Float32Array.from(r.mlpInput, (x) => x * gain);
    await wsp.backend.scatter(`layer_${o.layer}.mlp.gate`, rows(neuron * dim), read, 'set');
    await wsp.backend.scatter(`layer_${o.layer}.mlp.up`, rows(neuron * dim), read, 'set');
    // Clear the old output so the strength search starts from silence.
    const downIdx = Uint32Array.from({ length: dim }, (_, i) => i * h + neuron!);
    await wsp.backend.scatter(`layer_${o.layer}.mlp.down`, downIdx, new Float32Array(dim), 'set');
    const r2 = await s.read(context, want);
    const act = r2.key[neuron];
    const gn = norm(r2.grad);
    if (gn < 1e-12) throw new Error('No gradient reaches this layer for the wanted token');
    const d = Float32Array.from(r2.grad, (x) => -x / gn);
    const kept = await search(wsp, `layer_${o.layer}.mlp.down`, Math.max(r2.residualNorm, 1e-3) / Math.max(Math.abs(act), 1e-3), goal, () => s.prob(context, want), (m) =>
      wsp.backend.scatter(`layer_${o.layer}.mlp.down`, downIdx, Float32Array.from(d, (x) => x * m), 'set')
    );
    // How selective is it? Its activation on the preserve texts vs. on the prompt.
    let preserveActivation: number | undefined;
    if (preserve.length) {
      const keys = await s.keys(preserve);
      preserveActivation = keys.reduce((m, k) => Math.max(m, Math.abs(k[neuron!])), 0);
    }
    return {
      method: 'neuron wiring: gate/up read the student’s own MLP input at the prompt; down writes toward the token',
      layer: o.layer,
      neuron,
      grafted,
      token: tok.describe(want),
      activationOnPrompt: r4(act),
      ...(preserveActivation !== undefined ? { maxActivationOnPreserveTexts: r4(preserveActivation) } : {}),
      probabilityBefore: r4(pBefore),
      probabilityAfter: r4(kept.p),
      reachedGoal: kept.p >= goal,
    };
  } finally {
    s.destroy();
  }
}
