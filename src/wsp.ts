/**
 * The WSP API from the spec: wsp.state, wsp.inspect, wsp.mutate, wsp.eval.
 *
 * Every method returns small, JSON-able results an LLM can read: statistics,
 * a bounded grid, a ranked list of coordinates. Nothing hands back a whole
 * tensor, and every write goes through a shader in the backend.
 */
import { checkRegion, fullRegion, type Backend, type Region, type Sequence, type TrainTarget } from './backend.ts';
import { parseWspk, serializeWspk } from './model/formats.ts';
import { Model } from './model/model.ts';
import { checkId, defaultStore, type CheckpointStore } from './state/store.ts';
import { Datasets } from './state/datasets.ts';
import { insertAssociation, wireNeuron, type AssociationOptions, type WireOptions } from './computed.ts';
import { rng } from './model/random.ts';
import { classifierName, cloneConfig, parameterCount, type ModelConfig } from './model/types.ts';

export interface TensorSummary {
  name: string;
  role: string;
  shape: [number, number];
  count: number;
  mean: number;
  variance: number;
  std: number;
  min: number;
  max: number;
  /** Fraction of exact zeros. */
  sparsity: number;
  l2: number;
  nonFinite: number;
}

export type Edit = [number, number, number] | { row: number; col: number; value: number };

export interface GraftConfig {
  layer: number;
  kind: 'mlp_neurons' | 'attention_head';
  /** Neurons (or heads) to add. Default 1. */
  count?: number;
  /** How the new input weights start: random, a copy of `source` plus noise, or zeros. Default gaussian. */
  init?: 'gaussian' | 'clone' | 'zeros';
  /** Std for gaussian init; default the std of the tensor being grown (or 0.02). */
  std?: number;
  /** Neuron or head index to clone. */
  source?: number;
  /** Std of the noise added to cloned input weights; default 1% of the tensor's std. */
  noise?: number;
  /**
   * The new output weights: zeros (the network computes exactly what it did
   * until they are edited), gaussian, or split (Net2Net: the source's output is
   * shared with its clones, so the function is preserved). Default: split for
   * clones, zeros otherwise.
   */
  output?: 'zeros' | 'gaussian' | 'split';
  /** For attention heads: which KV head the new heads read. Default the source's, or 0. */
  kvHead?: number;
  seed?: number;
}

export interface PruneOptions {
  /** Texts the knowledge should be kept for; scores become trigger/contrast ratios. */
  contrastDataset?: string[];
  /** Restrict to these layers. */
  layers?: number[];
  /** At most this many neurons are zeroed. Default 16. */
  maxNeurons?: number;
  /** Report what would be pruned without changing anything. */
  dryRun?: boolean;
}

/**
 * What training may change: a tensor name, a layer prefix ("layer_2" for all
 * its tensors), "all", or a block of one tensor. Bounds are half-open and
 * default to the whole tensor.
 */
export type TargetSpec =
  | string
  | { tensor: string; row_start?: number; row_end?: number; col_start?: number; col_end?: number };

export interface MicroTrainOptions {
  /** AdamW learning rate. Default 3e-3. */
  lr?: number;
  /** Windows per step. Default 8. */
  batchSize?: number;
  weightDecay?: number;
  /** Global gradient-norm clip. Default 1. */
  clip?: number;
  /** Longest training window, in tokens. Default min(context, 256). */
  window?: number;
  seed?: number;
  signal?: AbortSignal;
  onProgress?: (text: string) => void;
}

export interface EvalOptions {
  maxTokens?: number;
  temperature?: number;
}

export interface WspOptions {
  store?: CheckpointStore;
  datasets?: Datasets;
}

const r4 = (x: number) => (Number.isFinite(x) ? Number(x.toPrecision(4)) : x);

export class WSP {
  readonly model: Model;
  readonly store: CheckpointStore;
  readonly datasets: Datasets;
  private lastEval = new Map<string, number>();
  private seedCounter = 1;

  constructor(model: Model, options: WspOptions = {}) {
    this.model = model;
    this.store = options.store ?? defaultStore();
    this.datasets = options.datasets ?? new Datasets();
  }

  get backend(): Backend {
    return this.model.backend;
  }

  private nextSeed(seed?: number) {
    return seed ?? (Date.now() ^ this.seedCounter++ * 2654435761) >>> 0;
  }

  // ---------------------------------------------------------------- state

  readonly state = {
    /** Serializes every tensor (VRAM → OPFS). */
    checkpoint: async (id: string, note?: string) => {
      checkId(id);
      const started = performance.now();
      const data = serializeWspk(this.model.config, await this.model.weights(), {
        modelId: this.model.id,
        createdAt: new Date().toISOString(),
        note,
      });
      await this.store.save(id, data);
      return { id, bytes: data.byteLength, store: this.store.kind, ms: Math.round(performance.now() - started) };
    },

    /** Loads a checkpoint back into the backend (OPFS → VRAM), shape included. */
    restore: async (id: string) => {
      const started = performance.now();
      const { config, weights, meta } = parseWspk(await this.store.load(id));
      if (meta.modelId !== this.model.id) {
        throw new Error(`Checkpoint "${id}" belongs to model ${meta.modelId}, not the loaded ${this.model.id}`);
      }
      this.model.load(config, weights);
      this.lastEval.clear();
      return { id, parameters: parameterCount(config), ms: Math.round(performance.now() - started) };
    },

    list: async () => (await this.store.list()).map((c) => ({ ...c, savedAt: new Date(c.savedAt).toISOString() })),

    delete: async (id: string) => {
      await this.store.delete(id);
      return { deleted: id };
    },
  };

  // ---------------------------------------------------------------- inspect

  readonly inspect = {
    /** The architecture, every tensor's name, shape and role, and the tokenizer. */
    overview: () => {
      const c = this.model.config;
      return {
        model: this.model.id,
        backend: this.backend.description,
        parameters: parameterCount(c),
        dim: c.dim,
        headSize: c.headSize,
        kvHeads: c.nKvHeads,
        vocabSize: c.vocabSize,
        contextLength: c.seqLen,
        classifier: classifierName(c),
        layers: c.layers.map((l, i) => ({ layer: i, heads: l.nHeads, headKv: l.headKv, mlpHidden: l.hidden })),
        tensors: this.model.specs().map((s) => ({ name: s.name, shape: s.shape, role: s.role })),
        architecture: {
          block: 'x += wo·Attention(RMSNorm(x)); x += down·(silu(gate·n) * (up·n)) with n = RMSNorm(x); logits = classifier·RMSNorm(x)',
          rmsNormEps: c.normEps,
          rotary: `applied to q and k after projection, on adjacent pairs (2i, 2i+1) within each head; angle = position / ${c.ropeTheta}^((2i mod headSize)/headSize)`,
          attention: 'causal softmax(q·k / sqrt(headSize)); query head h reads KV head headKv[h]',
          matrices: '[rows = outputs, cols = inputs]; y = W·x',
          classifier: c.sharedClassifier ? 'tied: logits use tok_embeddings' : 'separate lm_head',
        },
        tokenizer: this.tokenizerInfo(),
        training: 'mutate_micro_train runs real backpropagation with AdamW in WebGPU on the tensors or blocks you choose; everything else stays frozen.',
      };
    },

    /** Dimensions, sparsity, mean and variance of a tensor, of every tensor in a layer ("layer_3"), or the model ("model"). */
    summary: async (target: string): Promise<TensorSummary | { target: string; tensors: TensorSummary[] }> => {
      if (target === 'model' || /^layer_\d+$/.test(target)) {
        const names = this.model
          .specs()
          .map((s) => s.name)
          .filter((n) => target === 'model' || n.startsWith(target + '.'));
        if (!names.length) throw new Error(`No layer ${target}; the model has ${this.model.config.layers.length} layers (layer_0…)`);
        return { target, tensors: await Promise.all(names.map((n) => this.tensorSummary(n))) };
      }
      return this.tensorSummary(target);
    },

    /** A Markdown grid of a block of a tensor (at most 32 rows × 16 columns). */
    slice: async (target: string, bounds: Partial<Region> = {}) => {
      const spec = this.model.spec(target);
      const [rows, cols] = spec.shape;
      const r0 = bounds.rowStart ?? 0;
      const c0 = bounds.colStart ?? 0;
      const r1 = Math.min(bounds.rowEnd ?? r0 + 8, rows, r0 + 32);
      const c1 = Math.min(bounds.colEnd ?? c0 + 8, cols, c0 + 16);
      if (!(r0 >= 0 && r0 < r1 && c0 >= 0 && c0 < c1)) {
        throw new RangeError(`Bounds rows ${r0}..${r1}, cols ${c0}..${c1} select nothing in ${target} (${rows}×${cols})`);
      }
      const data = await this.backend.read(target, r0 * cols, (r1 - r0) * cols);
      const header = `| row \\ col | ${Array.from({ length: c1 - c0 }, (_, j) => c0 + j).join(' | ')} |`;
      const rule = `|---|${'---|'.repeat(c1 - c0)}`;
      const lines = [];
      for (let i = r0; i < r1; i++) {
        const cells = [];
        for (let j = c0; j < c1; j++) cells.push(fmt(data[(i - r0) * cols + j]));
        lines.push(`| ${i} | ${cells.join(' | ')} |`);
      }
      return {
        target,
        shape: spec.shape,
        bounds: { rowStart: r0, rowEnd: r1, colStart: c0, colEnd: c1 },
        markdown: [header, rule, ...lines].join('\n'),
      };
    },

    /**
     * Outliers found by a shader: 'values' gives elements whose z-score is at
     * least `threshold`; 'rows' gives rows (neurons, tokens, output units)
     * whose L2 norm is an outlier, dead rows included.
     */
    findAnomalies: async (target: string, threshold = 4, options: { mode?: 'values' | 'rows'; limit?: number } = {}) => {
      const spec = this.model.spec(target);
      const limit = Math.min(Math.max(1, options.limit ?? 20), 200);
      if (options.mode === 'rows') {
        const norms = await this.backend.rowNorms(target);
        let mean = 0;
        for (const v of norms) mean += v;
        mean /= norms.length || 1;
        let variance = 0;
        for (const v of norms) variance += (v - mean) ** 2;
        const std = Math.sqrt(variance / (norms.length || 1));
        const hits = [...norms.keys()]
          .map((row) => ({ row, norm: norms[row], z: std > 0 ? (norms[row] - mean) / std : 0 }))
          .filter((h) => Math.abs(h.z) >= threshold || h.norm === 0)
          .sort((a, b) => Math.abs(b.z) - Math.abs(a.z));
        return {
          target,
          mode: 'rows',
          rowNormMean: r4(mean),
          rowNormStd: r4(std),
          total: hits.length,
          rows: hits.slice(0, limit).map((h) => ({ row: h.row, norm: r4(h.norm), z: r4(h.z), ...(this.rowLabel(target, h.row)) })),
        };
      }
      const s = await this.backend.stats(target);
      const finite = s.count - s.nonFinite;
      const mean = finite ? s.sum / finite : 0;
      const std = Math.sqrt(Math.max(0, finite ? s.sumSq / finite - mean * mean : 0));
      if (std === 0 && s.nonFinite === 0) return { target, mode: 'values', mean: r4(mean), std: 0, total: 0, hits: [] };
      const found = await this.backend.outliers(target, mean, std, threshold, 4096);
      const cols = spec.shape[1];
      const hits = [...found.indices.keys()]
        .map((k) => ({ i: found.indices[k], v: found.values[k] }))
        .sort((a, b) => Math.abs(b.v - mean) - Math.abs(a.v - mean))
        .slice(0, limit)
        .map(({ i, v }) => ({ row: Math.floor(i / cols), col: i % cols, value: r4(v), z: r4((v - mean) / std) }));
      return { target, mode: 'values', mean: r4(mean), std: r4(std), total: found.total, hits };
    },

    /** How text tokenizes: ids and pieces (for editing token rows of tok_embeddings or the classifier). */
    tokenize: (text: string) => {
      const ids = this.model.tokenizer.encode(text);
      return { text, tokens: ids.map((id) => ({ id, piece: this.model.tokenizer.describe(id) })) };
    },

    /**
     * What an MLP neuron reads and writes, through the direct path: tokens whose
     * embeddings most drive its gate, and tokens its output most promotes or
     * suppresses at the classifier (a logit-lens approximation that ignores
     * later layers).
     */
    neuron: async (layer: number, index: number, k = 8) => {
      const c = this.model.config;
      const lc = c.layers[layer];
      if (!lc) throw new RangeError(`No layer ${layer}`);
      if (!(index >= 0 && index < lc.hidden)) throw new RangeError(`layer_${layer} has ${lc.hidden} MLP neurons`);
      const p = `layer_${layer}`;
      const tmp = '__neuron_col';
      this.backend.alloc(tmp, [c.dim, 1]);
      await this.backend.copyRegion(`${p}.mlp.down`, tmp, { rowStart: 0, rowEnd: c.dim, colStart: index, colEnd: index + 1 }, 0, 0);
      const out = await this.backend.read(tmp);
      this.backend.free(tmp);
      const gateRow = await this.backend.read(`${p}.mlp.gate`, index * c.dim, c.dim);
      const mlpNorm = await this.backend.read(`${p}.mlp_norm`);
      const finalNorm = await this.backend.read('norm');
      const writes = await this.backend.matvec(classifierName(c), out.map((v, i) => v * finalNorm[i]));
      const reads = await this.backend.matvec('tok_embeddings', gateRow.map((v, i) => v * mlpNorm[i]));
      const top = (v: Float32Array, dir: 1 | -1) =>
        [...v.keys()]
          .sort((a, b) => dir * (v[b] - v[a]))
          .slice(0, k)
          .map((t) => ({ token: t, piece: this.model.tokenizer.describe(t), score: r4(v[t]) }));
      let norm = 0;
      for (const v of out) norm += v * v;
      return {
        layer,
        neuron: index,
        outputNorm: r4(Math.sqrt(norm)),
        promotes: top(writes, 1),
        suppresses: top(writes, -1),
        activatedBy: top(reads, 1),
        note: 'Direct-path approximation: ignores the layers in between.',
      };
    },
  };

  // ---------------------------------------------------------------- mutate

  readonly mutate = {
    /**
     * Computed edit: makes `prompt` continue with `target` by a closed-form
     * rank-one change to one layer's mlp.down (ROME-style), protecting the
     * preserve texts. No training loop.
     */
    insertAssociation: (o: AssociationOptions) => insertAssociation(this, o),

    /** Computed edit: one neuron that fires on the prompt's context and promotes a token. */
    wireNeuron: (o: WireOptions) => wireNeuron(this, o),

    /** Overwrites (or adds to) specific floats: edits are [row, col, value]. */
    patchSparse: async (target: string, edits: Edit[], mode: 'set' | 'add' = 'set') => {
      const [rows, cols] = this.model.spec(target).shape;
      if (!Array.isArray(edits) || !edits.length) throw new Error('patchSparse needs at least one edit');
      if (edits.length > 4096) throw new Error('At most 4096 edits per patch; split larger patches');
      const merged = new Map<number, number>();
      for (const e of edits) {
        const [row, col, value] = Array.isArray(e) ? e : [e.row, e.col, e.value];
        if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0 || row >= rows || col >= cols) {
          throw new RangeError(`Edit [${row}, ${col}] is outside ${target} (${rows}×${cols})`);
        }
        if (!Number.isFinite(value)) throw new Error(`Edit [${row}, ${col}] has a non-finite value`);
        const i = row * cols + col;
        merged.set(i, mode === 'add' ? (merged.get(i) ?? 0) + value : value);
      }
      const indices = Uint32Array.from(merged.keys());
      await this.backend.scatter(target, indices, Float32Array.from(merged.values()), mode);
      return { target, mode, applied: indices.length };
    },

    /** Grows a layer by MLP neurons or attention heads, initialized by shaders. */
    graftNode: async (g: GraftConfig) => {
      const c = cloneConfig(this.model.config);
      const layer = c.layers[g.layer];
      if (!layer) throw new RangeError(`No layer ${g.layer}; the model has ${c.layers.length}`);
      const count = g.count ?? 1;
      if (!Number.isInteger(count) || count < 1 || count > 4096) throw new RangeError('count must be 1…4096');
      const init = g.init ?? 'gaussian';
      const output = g.output ?? (init === 'clone' ? 'split' : 'zeros');
      const seed = this.nextSeed(g.seed);
      const p = `layer_${g.layer}`;
      const be = this.backend;
      const hs = c.headSize;
      // Input-side tensors grow by rows, the output-side one by columns.
      const unit = g.kind === 'mlp_neurons' ? 1 : hs;
      const oldUnits = g.kind === 'mlp_neurons' ? layer.hidden : layer.nHeads;
      const inputs = g.kind === 'mlp_neurons' ? [`${p}.mlp.gate`, `${p}.mlp.up`] : [`${p}.attn.wq`];
      const outName = g.kind === 'mlp_neurons' ? `${p}.mlp.down` : `${p}.attn.wo`;
      if (g.kind !== 'mlp_neurons' && g.kind !== 'attention_head') throw new Error(`Unknown graft kind ${g.kind}`);
      if (init === 'clone' && !(g.source !== undefined && g.source >= 0 && g.source < oldUnits)) {
        throw new RangeError(`init "clone" needs a source ${g.kind === 'mlp_neurons' ? 'neuron' : 'head'} index below ${oldUnits}`);
      }
      const kv = g.kvHead ?? (g.source !== undefined && g.kind === 'attention_head' ? layer.headKv[g.source] : 0);
      if (g.kind === 'attention_head' && !(Number.isInteger(kv) && kv >= 0 && kv < c.nKvHeads)) {
        throw new RangeError(`kvHead must be an integer below ${c.nKvHeads}`);
      }
      if (output === 'split' && init !== 'clone') throw new Error('output "split" only applies to clones');
      const old = oldUnits * unit;
      const added = count * unit;

      for (const name of inputs) {
        const [, cols] = be.shape(name);
        const s = await this.stdOf(name);
        const tmp = `${name}#graft`;
        be.alloc(tmp, [old + added, cols]);
        await be.copyRegion(name, tmp, { rowStart: 0, rowEnd: old, colStart: 0, colEnd: cols }, 0, 0);
        const fresh: Region = { rowStart: old, rowEnd: old + added, colStart: 0, colEnd: cols };
        if (init === 'gaussian') await be.fill(tmp, fresh, { kind: 'gaussian', mean: 0, std: g.std ?? (s || 0.02), seed: seed ^ hash(name) });
        if (init === 'clone') {
          for (let k = 0; k < count; k++) {
            await be.copyRegion(name, tmp, { rowStart: g.source! * unit, rowEnd: (g.source! + 1) * unit, colStart: 0, colEnd: cols }, old + k * unit, 0);
          }
          await be.fill(tmp, fresh, { kind: 'add_gaussian', std: g.noise ?? 0.01 * (s || 0.02), seed: seed ^ hash(name) });
        }
        be.rename(tmp, name);
      }

      const [outRows] = be.shape(outName);
      const outStd = await this.stdOf(outName);
      const tmp = `${outName}#graft`;
      be.alloc(tmp, [outRows, old + added]);
      await be.copyRegion(outName, tmp, { rowStart: 0, rowEnd: outRows, colStart: 0, colEnd: old }, 0, 0);
      const freshCols: Region = { rowStart: 0, rowEnd: outRows, colStart: old, colEnd: old + added };
      if (output === 'gaussian') await be.fill(tmp, freshCols, { kind: 'gaussian', mean: 0, std: g.std ?? (outStd || 0.02), seed: seed ^ hash(outName) });
      if (output === 'split') {
        const src: Region = { rowStart: 0, rowEnd: outRows, colStart: g.source! * unit, colEnd: (g.source! + 1) * unit };
        for (let k = 0; k < count; k++) await be.copyRegion(outName, tmp, src, 0, old + k * unit);
        const share = 1 / (count + 1);
        await be.fill(tmp, src, { kind: 'scale', factor: share });
        await be.fill(tmp, freshCols, { kind: 'scale', factor: share });
      }
      be.rename(tmp, outName);

      if (g.kind === 'mlp_neurons') layer.hidden += count;
      else {
        layer.nHeads += count;
        for (let k = 0; k < count; k++) layer.headKv.push(kv);
      }
      this.model.setConfig(c);
      const range = [oldUnits, oldUnits + count - 1];
      return {
        layer: g.layer,
        kind: g.kind,
        added: count,
        newIndices: range,
        init,
        output,
        seed,
        functionPreserved: output !== 'gaussian',
        layerNow: { heads: layer.nHeads, mlpHidden: layer.hidden },
        parameters: parameterCount(c),
      };
    },

    /**
     * Runs the trigger texts, measures each MLP neuron's mean |activation|, and
     * zeroes the neurons most specific to them (their gate and up rows and their
     * down column). With a contrast set, specificity is the trigger/contrast
     * ratio (threshold e.g. 2); without one, the z-score within the layer
     * (threshold e.g. 3).
     */
    pruneSemantic: async (triggerDataset: string[], threshold: number, options: PruneOptions = {}) => {
      if (!triggerDataset.length) throw new Error('pruneSemantic needs at least one trigger text');
      const c = this.model.config;
      const trig = await this.model.track(triggerDataset);
      const contrast = options.contrastDataset?.length ? await this.model.track(options.contrastDataset) : undefined;
      const layers = options.layers ?? c.layers.map((_, i) => i);
      const candidates: { layer: number; neuron: number; trigger: number; contrast?: number; score: number }[] = [];
      for (const l of layers) {
        const a = trig.layers[l];
        if (!a) throw new RangeError(`No layer ${l}`);
        let mean = 0;
        for (const v of a) mean += v;
        mean /= a.length || 1;
        let variance = 0;
        for (const v of a) variance += (v - mean) ** 2;
        const std = Math.sqrt(variance / (a.length || 1));
        a.forEach((v, n) => {
          if (v === 0) return;
          if (contrast) {
            const cv = contrast.layers[l][n];
            candidates.push({ layer: l, neuron: n, trigger: v, contrast: cv, score: v / (cv + 1e-6 * (mean || 1)) });
          } else {
            candidates.push({ layer: l, neuron: n, trigger: v, score: std > 0 ? (v - mean) / std : 0 });
          }
        });
      }
      const chosen = candidates
        .filter((x) => x.score >= threshold)
        .sort((a, b) => b.score - a.score)
        .slice(0, options.maxNeurons ?? 16);
      if (!options.dryRun) {
        for (const { layer, neuron } of chosen) {
          const p = `layer_${layer}`;
          const zero = { kind: 'constant', value: 0 } as const;
          await this.backend.fill(`${p}.mlp.gate`, { rowStart: neuron, rowEnd: neuron + 1, colStart: 0, colEnd: c.dim }, zero);
          await this.backend.fill(`${p}.mlp.up`, { rowStart: neuron, rowEnd: neuron + 1, colStart: 0, colEnd: c.dim }, zero);
          await this.backend.fill(`${p}.mlp.down`, { rowStart: 0, rowEnd: c.dim, colStart: neuron, colEnd: neuron + 1 }, zero);
        }
      }
      return {
        scoring: contrast ? 'trigger/contrast activation ratio' : 'z-score of activation within the layer',
        threshold,
        tokensSeen: trig.tokens,
        candidatesAboveThreshold: candidates.filter((x) => x.score >= threshold).length,
        pruned: options.dryRun ? [] : chosen.map(({ layer, neuron }) => ({ layer, neuron })),
        dryRun: !!options.dryRun,
        neurons: chosen.map((x) => ({
          layer: x.layer,
          neuron: x.neuron,
          triggerActivation: r4(x.trigger),
          ...(x.contrast !== undefined ? { contrastActivation: r4(x.contrast) } : {}),
          score: r4(x.score),
        })),
      };
    },

    /**
     * The "intern": real backpropagation with AdamW on the chosen tensors or
     * blocks (everything else frozen), over a named dataset or a list of texts.
     * Texts are cut into windows of at most `window` tokens.
     */
    microTrain: async (targets: TargetSpec | TargetSpec[], dataset: string | string[], steps: number, options: MicroTrainOptions = {}) => {
      if (!(Number.isInteger(steps) && steps >= 1 && steps <= 20000)) throw new RangeError('steps must be 1…20000');
      const resolved = this.resolveTargets(Array.isArray(targets) ? targets : [targets]);
      const texts = typeof dataset === 'string' ? await this.datasets.get(dataset) : dataset;
      if (!texts.length) throw new Error('microTrain needs at least one text');
      const windowLen = Math.min(options.window ?? 256, this.model.config.seqLen, 512);
      const windows = this.windows(texts, windowLen);
      if (!windows.length) throw new Error('The texts are too short to train on (each needs at least two tokens)');
      const length = Math.max(...windows.map((w) => w.tokens.length));
      const batchSize = Math.min(Math.max(1, options.batchSize ?? 8), 64);
      const adam = {
        lr: options.lr ?? 3e-3,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weightDecay: options.weightDecay ?? 0,
        clip: options.clip ?? 1,
      };
      if (!(adam.lr > 0 && adam.lr < 1)) throw new RangeError('lr must be between 0 and 1');
      const started = performance.now();
      const trainer = this.backend.createTrainer(this.model.config, length, resolved);
      const random = rng(this.nextSeed(options.seed));
      const evalSet = windows.length > 64 ? sampleWithout(windows, 64, random) : windows;
      const meanLoss = async () => {
        let sum = 0;
        let count = 0;
        for (let i = 0; i < evalSet.length; i += 16) {
          const r = await trainer.loss(evalSet.slice(i, i + 16));
          sum += r.sum;
          count += r.count;
        }
        return sum / count;
      };
      let lossBefore = NaN;
      let lossAfter = NaN;
      const trajectory: number[] = [];
      let done = 0;
      let skipped = 0;
      let tokens = 0;
      let lastGradNorm = 0;
      try {
        lossBefore = await meanLoss();
        const warmup = Math.min(20, Math.ceil(steps / 10));
        let running = lossBefore;
        let lastReport = performance.now();
        for (let s = 0; s < steps; s++) {
          if (options.signal?.aborted) break;
          const batch = Array.from({ length: Math.min(batchSize, windows.length) }, () => windows[Math.floor(random() * windows.length)]);
          const r = await trainer.accumulate(batch);
          tokens += r.count;
          // Linear warm-up, then cosine decay to 10%.
          const progress = s < warmup ? (s + 1) / warmup : 0.1 + 0.9 * 0.5 * (1 + Math.cos((Math.PI * (s - warmup)) / Math.max(1, steps - warmup)));
          const u = await trainer.update({ ...adam, lr: adam.lr * progress });
          lastGradNorm = u.gradNorm;
          if (u.skipped) skipped++;
          running = 0.9 * running + 0.1 * (r.sum / Math.max(1, r.count));
          done++;
          if (done % Math.max(1, Math.ceil(steps / 20)) === 0 || done === steps) trajectory.push(r4(running));
          if (options.onProgress && (done === 1 || performance.now() - lastReport > 500)) {
            lastReport = performance.now();
            options.onProgress(`step ${done}/${steps} · loss ${running.toFixed(3)}`);
          }
        }
        lossAfter = await meanLoss();
      } finally {
        trainer.destroy();
      }
      const trainable = resolved.reduce((n, t) => n + (t.region.rowEnd - t.region.rowStart) * (t.region.colEnd - t.region.colStart), 0);
      const seconds = (performance.now() - started) / 1000;
      return {
        method: 'AdamW with backpropagation; other tensors frozen',
        targets: resolved.map((t) => ({ tensor: t.name, rows: [t.region.rowStart, t.region.rowEnd], cols: [t.region.colStart, t.region.colEnd] })),
        trainableParameters: trainable,
        windows: windows.length,
        windowLength: length,
        steps: done,
        stoppedEarly: done < steps,
        skippedNonFiniteSteps: skipped,
        lossBefore: r4(lossBefore),
        lossAfter: r4(lossAfter),
        chanceLoss: r4(Math.log(this.model.config.vocabSize)),
        trajectory,
        lastGradNorm: r4(lastGradNorm),
        tokensPerSecond: Math.round(tokens / Math.max(seconds, 1e-3)),
        seconds: r4(seconds),
      };
    },
  };

  // ---------------------------------------------------------------- data

  readonly data = {
    /** Saves texts under a name (appending if asked), for microTrain. */
    write: (name: string, texts: string[], append = false) => this.datasets.write(name, texts, append),
    get: (name: string) => this.datasets.get(name),
    list: () => this.datasets.list(),
    sample: (name: string, start?: number, count?: number) => this.datasets.sample(name, start, count),
    delete: (name: string) => this.datasets.delete(name),
  };

  // ---------------------------------------------------------------- eval

  readonly eval = {
    /**
     * Local inference: the text the model generates after `prompt`, and with
     * `expectedOutput` the loss and perplexity of that continuation (plus the
     * change since the last eval of the same pair).
     */
    run: async (prompt: string, expectedOutput?: string, options: EvalOptions = {}) => {
      const started = performance.now();
      const gen = await this.model.generate(prompt, { maxTokens: options.maxTokens ?? 48, temperature: options.temperature ?? 0 });
      const result: Record<string, unknown> = { prompt, generated: gen.text };
      if (expectedOutput !== undefined && expectedOutput !== '') {
        const tok = this.model.tokenizer;
        const context = tok.encode(prompt);
        const all = tok.encode(prompt + expectedOutput);
        let k = 0;
        while (k < context.length && k < all.length && context[k] === all[k]) k++;
        const { loss, tokens } = await this.model.nll(all.slice(0, k), all.slice(k));
        const key = `${prompt}\u0000${expectedOutput}`;
        const previous = this.lastEval.get(key);
        this.lastEval.set(key, loss);
        Object.assign(result, {
          expectedOutput,
          loss: r4(loss),
          perplexity: r4(Math.exp(loss)),
          scoredTokens: tokens,
          ...(previous !== undefined ? { previousLoss: r4(previous), delta: r4(loss - previous) } : {}),
          generatedMatchesExpected: gen.text.trimStart().startsWith(expectedOutput.trim()),
        });
      }
      result.ms = Math.round(performance.now() - started);
      return result;
    },

    /** The most likely next tokens after a prompt, with probabilities. */
    nextTokens: async (prompt: string, k = 8) => {
      const top = await this.model.topNext(prompt, Math.min(Math.max(1, k), 50));
      return { prompt, next: top.map((t) => ({ ...t, p: r4(t.p) })) };
    },
  };

  // ---------------------------------------------------------------- helpers

  private async tensorSummary(name: string): Promise<TensorSummary> {
    const spec = this.model.spec(name);
    const s = await this.backend.stats(name);
    const finite = s.count - s.nonFinite;
    const mean = finite ? s.sum / finite : 0;
    const variance = Math.max(0, finite ? s.sumSq / finite - mean * mean : 0);
    return {
      name,
      role: spec.role,
      shape: spec.shape,
      count: s.count,
      mean: r4(mean),
      variance: r4(variance),
      std: r4(Math.sqrt(variance)),
      min: r4(s.min),
      max: r4(s.max),
      sparsity: r4(s.count ? s.zeros / s.count : 0),
      l2: r4(Math.sqrt(s.sumSq)),
      nonFinite: s.nonFinite,
    };
  }

  /** Turns target specs into tensor blocks, checking every bound. */
  resolveTargets(specs: TargetSpec[]): TrainTarget[] {
    if (!specs.length) throw new Error('Name at least one tensor to train');
    const names = this.model.specs().map((s) => s.name);
    const out = new Map<string, TrainTarget>();
    for (const spec of specs) {
      const tensor = typeof spec === 'string' ? spec : spec.tensor;
      const matched = tensor === 'all' ? names : names.includes(tensor) ? [tensor] : names.filter((n) => n.startsWith(tensor + '.'));
      if (!matched.length) this.model.spec(tensor); // throws a helpful error
      for (const name of matched) {
        const shape = this.backend.shape(name);
        let region = fullRegion(shape);
        if (typeof spec !== 'string') {
          if (matched.length > 1 && (spec.row_start ?? spec.row_end ?? spec.col_start ?? spec.col_end) !== undefined) {
            throw new Error(`Row/column bounds need a single tensor, not "${tensor}"`);
          }
          region = {
            rowStart: spec.row_start ?? 0,
            rowEnd: spec.row_end ?? shape[0],
            colStart: spec.col_start ?? 0,
            colEnd: spec.col_end ?? shape[1],
          };
          checkRegion(shape, region, name);
        }
        if (out.has(name)) throw new Error(`${name} is named twice`);
        if (region.rowEnd > region.rowStart && region.colEnd > region.colStart) out.set(name, { name, region });
      }
    }
    if (!out.size) throw new Error('The targets select no parameters (empty tensors or blocks)');
    return [...out.values()];
  }

  /** Each text as BOS-prefixed tokens, cut into next-token windows of at most `length`. */
  private windows(texts: string[], length: number): Sequence[] {
    const out: Sequence[] = [];
    for (const text of texts) {
      const ids = this.model.tokenizer.encode(text, { eos: true });
      for (let i = 0; i + 1 < ids.length; i += length) {
        const chunk = ids.slice(i, i + length + 1);
        if (chunk.length < 2) break;
        out.push({ tokens: chunk.slice(0, -1), targets: chunk.slice(1) });
      }
    }
    return out;
  }

  private tokenizerInfo() {
    const tok = this.model.tokenizer;
    const byteLevel = tok.vocabSize === 259 && tok.pieces[3] === '<0x00>';
    return byteLevel
      ? { kind: 'byte-level', vocabSize: tok.vocabSize, special: { unk: 0, bos: 1, eos: 2 }, tokenId: 'UTF-8 byte + 3 (e.g. " " = 35, "\\n" = 13, "a" = 100)' }
      : { kind: 'SentencePiece BPE (llama2.c)', vocabSize: tok.vocabSize, special: { unk: 0, bos: 1, eos: 2 }, note: 'pieces start with " " where a word starts; unknown characters fall back to <0xXX> byte pieces' };
  }

  private async stdOf(name: string): Promise<number> {
    const s = await this.backend.stats(name);
    const n = s.count - s.nonFinite;
    if (!n) return 0;
    const mean = s.sum / n;
    return Math.sqrt(Math.max(0, s.sumSq / n - mean * mean));
  }

  /** For token-indexed rows, what token the row is. */
  private rowLabel(target: string, row: number): { piece?: string } {
    if (target === 'tok_embeddings' || target === 'lm_head') return { piece: this.model.tokenizer.describe(row) };
    return {};
  }

  /** The model as a .wspk file (for download). */
  async exportWspk(note?: string): Promise<ArrayBuffer> {
    return serializeWspk(this.model.config, await this.model.weights(), {
      modelId: this.model.id,
      createdAt: new Date().toISOString(),
      note,
    });
  }

  get config(): ModelConfig {
    return this.model.config;
  }
}

function sampleWithout<T>(items: T[], k: number, random: () => number): T[] {
  const copy = [...items];
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(random() * (copy.length - i));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, k);
}

function fmt(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return '0';
  const a = Math.abs(x);
  return a >= 1e4 || a < 1e-3 ? x.toExponential(2) : x.toPrecision(4);
}

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
