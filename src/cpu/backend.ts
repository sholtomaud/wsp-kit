/**
 * The CPU reference: the same operations as the WebGPU backend, as plain
 * loops over Float32Arrays. The forward pass follows llama2.c's run.c.
 */
import { checkRegion, type Backend, type Fill, type Hits, type Region, type RunOptions, type Runner, type Stats } from '../backend.ts';
import { gaussian } from '../model/random.ts';
import { classifierName, kvDim, type ModelConfig, type Shape } from '../model/types.ts';

interface Tensor {
  shape: Shape;
  data: Float32Array;
}

const isFinite32 = (x: number) => Number.isFinite(x) && Math.abs(x) <= 3.4028234663852886e38;

export class CpuBackend implements Backend {
  readonly kind = 'cpu' as const;
  readonly description = 'CPU reference';
  private tensors = new Map<string, Tensor>();

  private get(name: string): Tensor {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`No tensor named ${name}`);
    return t;
  }

  /** Direct access for the forward pass. */
  data(name: string): Float32Array {
    return this.get(name).data;
  }

  has(name: string) {
    return this.tensors.has(name);
  }
  shape(name: string): Shape {
    return [...this.get(name).shape];
  }
  names() {
    return [...this.tensors.keys()];
  }

  alloc(name: string, shape: Shape, data?: Float32Array) {
    const n = shape[0] * shape[1];
    if (data && data.length !== n) throw new Error(`${name}: ${data.length} values for a ${shape[0]}×${shape[1]} tensor`);
    this.tensors.set(name, { shape: [...shape], data: data ? Float32Array.from(data) : new Float32Array(n) });
  }

  free(name: string) {
    this.tensors.delete(name);
  }

  async copy(src: string, dst: string) {
    const s = this.get(src);
    const d = this.tensors.get(dst);
    if (d && d.shape[0] === s.shape[0] && d.shape[1] === s.shape[1]) d.data.set(s.data);
    else this.alloc(dst, s.shape, s.data);
  }

  rename(from: string, to: string) {
    const t = this.get(from);
    this.tensors.delete(from);
    this.tensors.set(to, t);
  }

  async matvec(name: string, vector: Float32Array) {
    const { shape, data } = this.get(name);
    if (vector.length !== shape[1]) throw new Error(`${name}: vector of ${vector.length} for ${shape[1]} columns`);
    const out = new Float32Array(shape[0]);
    matvec(out, data, vector, shape[0], shape[1]);
    return out;
  }

  async copyRegion(src: string, dst: string, r: Region, dstRow: number, dstCol: number) {
    const s = this.get(src);
    const d = this.get(dst);
    checkRegion(s.shape, r, `${src}`);
    checkRegion(d.shape, { rowStart: dstRow, rowEnd: dstRow + r.rowEnd - r.rowStart, colStart: dstCol, colEnd: dstCol + r.colEnd - r.colStart }, dst);
    for (let i = r.rowStart; i < r.rowEnd; i++) {
      for (let j = r.colStart; j < r.colEnd; j++) {
        d.data[(dstRow + i - r.rowStart) * d.shape[1] + dstCol + j - r.colStart] = s.data[i * s.shape[1] + j];
      }
    }
  }

  async read(name: string, offset = 0, length?: number) {
    const t = this.get(name);
    const end = length === undefined ? t.data.length : offset + length;
    if (offset < 0 || end > t.data.length) throw new RangeError(`${name}: read past the end`);
    return t.data.slice(offset, end);
  }

  async write(name: string, data: Float32Array) {
    const t = this.get(name);
    if (data.length !== t.data.length) throw new Error(`${name}: wrong length`);
    t.data.set(data);
  }

  async stats(name: string): Promise<Stats> {
    const { data } = this.get(name);
    const s: Stats = { count: data.length, sum: 0, sumSq: 0, min: Infinity, max: -Infinity, zeros: 0, nonFinite: 0 };
    for (const x of data) {
      if (!isFinite32(x)) {
        s.nonFinite++;
        continue;
      }
      s.sum += x;
      s.sumSq += x * x;
      if (x < s.min) s.min = x;
      if (x > s.max) s.max = x;
      if (x === 0) s.zeros++;
    }
    return s;
  }

  async rowNorms(name: string) {
    const { shape, data } = this.get(name);
    const out = new Float32Array(shape[0]);
    for (let i = 0; i < shape[0]; i++) {
      let s = 0;
      for (let j = 0; j < shape[1]; j++) s += data[i * shape[1] + j] ** 2;
      out[i] = Math.sqrt(s);
    }
    return out;
  }

  async outliers(name: string, mean: number, std: number, z: number, cap: number): Promise<Hits> {
    const { data } = this.get(name);
    const indices: number[] = [];
    const values: number[] = [];
    let total = 0;
    const inv = std > 0 ? 1 / std : Infinity;
    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      const score = isFinite32(x) ? Math.abs(x - mean) * inv : Infinity;
      if (score >= z) {
        if (total < cap) {
          indices.push(i);
          values.push(x);
        }
        total++;
      }
    }
    return { indices: Uint32Array.from(indices), values: Float32Array.from(values), total };
  }

  async scatter(name: string, indices: Uint32Array, values: Float32Array, mode: 'set' | 'add') {
    const { data } = this.get(name);
    indices.forEach((idx, k) => {
      if (idx >= data.length) throw new RangeError(`${name}: index ${idx} out of range`);
      data[idx] = mode === 'set' ? values[k] : Math.fround(data[idx] + values[k]);
    });
  }

  async fill(name: string, r: Region, fill: Fill) {
    const { shape, data } = this.get(name);
    checkRegion(shape, r, name);
    for (let i = r.rowStart; i < r.rowEnd; i++) {
      for (let j = r.colStart; j < r.colEnd; j++) {
        const idx = i * shape[1] + j;
        switch (fill.kind) {
          case 'constant':
            data[idx] = fill.value;
            break;
          case 'gaussian':
            data[idx] = fill.mean + fill.std * gaussian(fill.seed, idx);
            break;
          case 'add_gaussian':
            data[idx] += fill.std * gaussian(fill.seed, idx);
            break;
          case 'scale':
            data[idx] *= fill.factor;
            break;
        }
      }
    }
  }

  async flush() {}

  createRunner(config: ModelConfig): Runner {
    return new CpuRunner(this, config);
  }
}

function rmsnorm(out: Float32Array, x: Float32Array, w: Float32Array, eps: number) {
  let ss = 0;
  for (let j = 0; j < x.length; j++) ss += x[j] * x[j];
  const scale = 1 / Math.sqrt(ss / x.length + eps);
  for (let j = 0; j < x.length; j++) out[j] = w[j] * (scale * x[j]);
}

/** out[outOffset + i] (+)= Σ_j W[i, j] x[j] */
function matvec(out: Float32Array, w: Float32Array, x: Float32Array, rows: number, cols: number, accumulate = false, outOffset = 0) {
  for (let i = 0; i < rows; i++) {
    let s = 0;
    const o = i * cols;
    for (let j = 0; j < cols; j++) s += w[o + j] * x[j];
    out[outOffset + i] = accumulate ? out[outOffset + i] + s : s;
  }
}

class CpuRunner implements Runner {
  private kCache: Float32Array[];
  private vCache: Float32Array[];
  private acts: Float32Array[];
  private trackedTokens = 0;
  private readonly be: CpuBackend;
  private readonly c: ModelConfig;

  constructor(be: CpuBackend, c: ModelConfig) {
    this.be = be;
    this.c = c;
    this.kCache = c.layers.map(() => new Float32Array(c.seqLen * kvDim(c)));
    this.vCache = c.layers.map(() => new Float32Array(c.seqLen * kvDim(c)));
    this.acts = c.layers.map((l) => new Float32Array(l.hidden));
  }

  async run(tokens: number[], start: number, options: RunOptions = {}) {
    const c = this.c;
    const { dim, headSize: hs } = c;
    const kv = kvDim(c);
    const x = new Float32Array(dim);
    const xb = new Float32Array(dim);
    const logits = new Float32Array(c.vocabSize);
    const emb = this.be.data('tok_embeddings');
    const cls = this.be.data(classifierName(c));
    let nllSum = 0;
    let nllCount = 0;
    tokens.forEach((token, t) => {
      const pos = start + t;
      if (pos >= c.seqLen) throw new RangeError(`Position ${pos} is past the model's context (${c.seqLen})`);
      x.set(emb.subarray(token * dim, (token + 1) * dim));
      c.layers.forEach((layer, l) => {
        const p = `layer_${l}`;
        const qd = layer.nHeads * hs;
        rmsnorm(xb, x, this.be.data(`${p}.attn_norm`), c.normEps);
        const q = new Float32Array(qd);
        matvec(q, this.be.data(`${p}.attn.wq`), xb, qd, dim);
        matvec(this.kCache[l], this.be.data(`${p}.attn.wk`), xb, kv, dim, false, pos * kv);
        matvec(this.vCache[l], this.be.data(`${p}.attn.wv`), xb, kv, dim, false, pos * kv);
        const k = this.kCache[l].subarray(pos * kv, (pos + 1) * kv);
        for (let i = 0; i < Math.max(qd, kv); i += 2) {
          const freq = 1 / Math.pow(c.ropeTheta, (i % hs) / hs);
          const val = pos * freq;
          const fcr = Math.cos(val);
          const fci = Math.sin(val);
          for (const vec of i < kv ? (i < qd ? [q, k] : [k]) : [q]) {
            const v0 = vec[i];
            const v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
          }
        }
        const ao = new Float32Array(qd);
        const att = new Float32Array(pos + 1);
        for (let h = 0; h < layer.nHeads; h++) {
          const ko = layer.headKv[h] * hs;
          let max = -Infinity;
          for (let s = 0; s <= pos; s++) {
            let score = 0;
            for (let d = 0; d < hs; d++) score += q[h * hs + d] * this.kCache[l][s * kv + ko + d];
            att[s] = score / Math.sqrt(hs);
            if (att[s] > max) max = att[s];
          }
          let sum = 0;
          for (let s = 0; s <= pos; s++) {
            att[s] = Math.exp(att[s] - max);
            sum += att[s];
          }
          for (let d = 0; d < hs; d++) {
            let acc = 0;
            for (let s = 0; s <= pos; s++) acc += att[s] * this.vCache[l][s * kv + ko + d];
            ao[h * hs + d] = acc / sum;
          }
        }
        matvec(x, this.be.data(`${p}.attn.wo`), ao, dim, qd, true);
        rmsnorm(xb, x, this.be.data(`${p}.mlp_norm`), c.normEps);
        const hb = new Float32Array(layer.hidden);
        const hb2 = new Float32Array(layer.hidden);
        matvec(hb, this.be.data(`${p}.mlp.gate`), xb, layer.hidden, dim);
        matvec(hb2, this.be.data(`${p}.mlp.up`), xb, layer.hidden, dim);
        for (let i = 0; i < layer.hidden; i++) {
          const a = hb[i];
          hb[i] = (a / (1 + Math.exp(-a))) * hb2[i];
          if (options.track) this.acts[l][i] += Math.abs(hb[i]);
        }
        matvec(x, this.be.data(`${p}.mlp.down`), hb, dim, layer.hidden, true);
      });
      if (options.track) this.trackedTokens++;
      const target = options.targets?.[t] ?? -1;
      const wantLogits = options.logits && t === tokens.length - 1;
      if (target >= 0 || wantLogits) {
        rmsnorm(xb, x, this.be.data('norm'), c.normEps);
        matvec(logits, cls, xb, c.vocabSize, dim);
      }
      if (target >= 0) {
        let max = -Infinity;
        for (const v of logits) if (v > max) max = v;
        let sum = 0;
        for (const v of logits) sum += Math.exp(v - max);
        nllSum += -(logits[target] - max - Math.log(sum));
        nllCount++;
      }
    });
    return {
      logits: options.logits ? logits : undefined,
      nll: options.targets ? { sum: nllSum, count: nllCount } : undefined,
    };
  }

  async activations() {
    const n = Math.max(1, this.trackedTokens);
    return { tokens: this.trackedTokens, layers: this.acts.map((a) => a.map((v) => v / n)) };
  }

  resetActivations() {
    this.acts.forEach((a) => a.fill(0));
    this.trackedTokens = 0;
  }

  destroy() {}
}
