/**
 * The CPU reference for training: a batched forward pass that keeps every
 * activation, the matching backward pass, and AdamW on the trainable regions.
 * The WebGPU trainer is checked against it, and it is checked against finite
 * differences.
 */
import type { AdamOptions, Sequence, TrainTarget, Trainer, TrainerOptions } from '../backend.ts';
import { classifierName, kvDim, type ModelConfig } from '../model/types.ts';
import type { CpuBackend } from './backend.ts';

type Vec = Float64Array;

/** out[t, i] = Σ_j W[i, j] X[t, j] */
function linear(X: Vec, W: Float32Array, T: number, nIn: number, nOut: number): Vec {
  const out = new Float64Array(T * nOut);
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < nOut; i++) {
      let s = 0;
      for (let j = 0; j < nIn; j++) s += W[i * nIn + j] * X[t * nIn + j];
      out[t * nOut + i] = s;
    }
  }
  return out;
}

/** dX[t, j] += Σ_i dY[t, i] W[i, j] */
function linearBackX(dX: Vec, dY: Vec, W: Float32Array, T: number, nIn: number, nOut: number) {
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < nOut; i++) {
      const g = dY[t * nOut + i];
      if (g === 0) continue;
      for (let j = 0; j < nIn; j++) dX[t * nIn + j] += g * W[i * nIn + j];
    }
  }
}

/** dW[i, j] += Σ_t dY[t, i] X[t, j] */
function linearBackW(dW: Vec, dY: Vec, X: Vec, T: number, nIn: number, nOut: number) {
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < nOut; i++) {
      const g = dY[t * nOut + i];
      if (g === 0) continue;
      for (let j = 0; j < nIn; j++) dW[i * nIn + j] += g * X[t * nIn + j];
    }
  }
}

function rmsnorm(X: Vec, w: Float32Array, T: number, n: number, eps: number): { y: Vec; rstd: Vec } {
  const y = new Float64Array(T * n);
  const rstd = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    let ss = 0;
    for (let j = 0; j < n; j++) ss += X[t * n + j] ** 2;
    const r = 1 / Math.sqrt(ss / n + eps);
    rstd[t] = r;
    for (let j = 0; j < n; j++) y[t * n + j] = w[j] * (r * X[t * n + j]);
  }
  return { y, rstd };
}

/** dX += ∂/∂X of the norm; dw += ∂/∂w (when given). */
function rmsnormBack(dX: Vec, dY: Vec, X: Vec, w: Float32Array, rstd: Vec, T: number, n: number, dw?: Vec) {
  for (let t = 0; t < T; t++) {
    const r = rstd[t];
    let s = 0;
    for (let j = 0; j < n; j++) s += w[j] * dY[t * n + j] * X[t * n + j];
    for (let j = 0; j < n; j++) {
      dX[t * n + j] += r * w[j] * dY[t * n + j] - (X[t * n + j] * r * r * r * s) / n;
      if (dw) dw[j] += dY[t * n + j] * X[t * n + j] * r;
    }
  }
}

/** Rotates adjacent pairs within each head by ±pos·freq (sign −1 undoes it). */
function rope(buf: Vec, T: number, width: number, hs: number, theta: number, sign: number) {
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < width; i += 2) {
      const val = t / Math.pow(theta, (i % hs) / hs);
      const c = Math.cos(val);
      const s = sign * Math.sin(val);
      const o = t * width + i;
      const v0 = buf[o];
      const v1 = buf[o + 1];
      buf[o] = v0 * c - v1 * s;
      buf[o + 1] = v0 * s + v1 * c;
    }
  }
}

interface LayerCache {
  X: Vec;
  XN1: Vec;
  rstd1: Vec;
  Q: Vec;
  K: Vec;
  V: Vec;
  /** P[h][t] = attention weights over s ≤ t. */
  P: Vec[][];
  O: Vec;
  Xmid: Vec;
  XN2: Vec;
  rstd2: Vec;
  G: Vec;
  U: Vec;
  H: Vec;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export class CpuTrainer implements Trainer {
  readonly length: number;
  private readonly be: CpuBackend;
  private readonly c: ModelConfig;
  private readonly targets: TrainTarget[];
  private grads = new Map<string, Vec>();
  private m = new Map<string, Vec>();
  private v = new Map<string, Vec>();
  private t = 0;
  private readonly probeLayer?: number;
  private lastProbe?: Vec;

  constructor(be: CpuBackend, c: ModelConfig, length: number, targets: TrainTarget[], options: TrainerOptions = {}) {
    this.be = be;
    this.c = c;
    this.length = length;
    this.targets = targets;
    this.probeLayer = options.probeLayer;
    if (this.probeLayer !== undefined && !(this.probeLayer >= 0 && this.probeLayer < c.layers.length)) throw new RangeError(`No layer ${this.probeLayer}`);
    for (const { name } of targets) {
      const n = be.data(name).length;
      if (!this.grads.has(name)) {
        this.grads.set(name, new Float64Array(n));
        this.m.set(name, new Float64Array(n));
        this.v.set(name, new Float64Array(n));
      }
    }
  }

  private w(name: string) {
    return this.be.data(name);
  }

  private check(seq: Sequence) {
    if (seq.tokens.length !== seq.targets.length) throw new Error('tokens and targets differ in length');
    if (seq.tokens.length > this.length) throw new RangeError(`A window of ${seq.tokens.length} tokens exceeds the trainer's ${this.length}`);
  }

  /** Forward pass over one window; returns what backward needs, and the loss. */
  private forward(seq: Sequence, scale: number, wantGrad: boolean) {
    this.check(seq);
    const c = this.c;
    const { dim, headSize: hs } = c;
    const kv = kvDim(c);
    const T = seq.tokens.length;
    const emb = this.w('tok_embeddings');
    let X = new Float64Array(T * dim);
    seq.tokens.forEach((tok, t) => {
      for (let j = 0; j < dim; j++) X[t * dim + j] = emb[tok * dim + j];
    });
    const layers: LayerCache[] = [];
    c.layers.forEach((layer, l) => {
      const p = `layer_${l}`;
      const qd = layer.nHeads * hs;
      const { y: XN1, rstd: rstd1 } = rmsnorm(X, this.w(`${p}.attn_norm`), T, dim, c.normEps);
      const Q = linear(XN1, this.w(`${p}.attn.wq`), T, dim, qd);
      const K = linear(XN1, this.w(`${p}.attn.wk`), T, dim, kv);
      const V = linear(XN1, this.w(`${p}.attn.wv`), T, dim, kv);
      rope(Q, T, qd, hs, c.ropeTheta, 1);
      rope(K, T, kv, hs, c.ropeTheta, 1);
      const scaleQK = 1 / Math.sqrt(hs);
      const P: Vec[][] = [];
      const O = new Float64Array(T * qd);
      for (let h = 0; h < layer.nHeads; h++) {
        const ko = layer.headKv[h] * hs;
        P.push([]);
        for (let t = 0; t < T; t++) {
          const row = new Float64Array(t + 1);
          let max = -Infinity;
          for (let s = 0; s <= t; s++) {
            let d = 0;
            for (let i = 0; i < hs; i++) d += Q[t * qd + h * hs + i] * K[s * kv + ko + i];
            row[s] = d * scaleQK;
            if (row[s] > max) max = row[s];
          }
          let sum = 0;
          for (let s = 0; s <= t; s++) sum += row[s] = Math.exp(row[s] - max);
          for (let s = 0; s <= t; s++) row[s] /= sum;
          P[h].push(row);
          for (let i = 0; i < hs; i++) {
            let acc = 0;
            for (let s = 0; s <= t; s++) acc += row[s] * V[s * kv + ko + i];
            O[t * qd + h * hs + i] = acc;
          }
        }
      }
      const Xmid = Float64Array.from(X);
      const attnOut = linear(O, this.w(`${p}.attn.wo`), T, qd, dim);
      for (let i = 0; i < Xmid.length; i++) Xmid[i] += attnOut[i];
      const { y: XN2, rstd: rstd2 } = rmsnorm(Xmid, this.w(`${p}.mlp_norm`), T, dim, c.normEps);
      const G = linear(XN2, this.w(`${p}.mlp.gate`), T, dim, layer.hidden);
      const U = linear(XN2, this.w(`${p}.mlp.up`), T, dim, layer.hidden);
      const H = new Float64Array(T * layer.hidden);
      for (let i = 0; i < H.length; i++) H[i] = G[i] * sigmoid(G[i]) * U[i];
      const Xout = Float64Array.from(Xmid);
      const mlpOut = linear(H, this.w(`${p}.mlp.down`), T, layer.hidden, dim);
      for (let i = 0; i < Xout.length; i++) Xout[i] += mlpOut[i];
      layers.push({ X, XN1, rstd1, Q, K, V, P, O, Xmid, XN2, rstd2, G, U, H });
      X = Xout;
    });
    const { y: XF, rstd: rstdF } = rmsnorm(X, this.w('norm'), T, dim, c.normEps);
    const cls = this.w(classifierName(c));
    const logits = linear(XF, cls, T, dim, c.vocabSize);
    let sum = 0;
    let count = 0;
    const dlogits = wantGrad ? new Float64Array(T * c.vocabSize) : undefined;
    for (let t = 0; t < T; t++) {
      const target = seq.targets[t];
      if (target < 0) continue;
      const o = t * c.vocabSize;
      let max = -Infinity;
      for (let j = 0; j < c.vocabSize; j++) if (logits[o + j] > max) max = logits[o + j];
      let z = 0;
      for (let j = 0; j < c.vocabSize; j++) z += Math.exp(logits[o + j] - max);
      const lse = max + Math.log(z);
      sum += lse - logits[o + target];
      count++;
      if (dlogits) {
        for (let j = 0; j < c.vocabSize; j++) dlogits[o + j] = Math.exp(logits[o + j] - lse) * scale;
        dlogits[o + target] -= scale;
      }
    }
    return { sum, count, layers, Xfinal: X, XF, rstdF, dlogits, T };
  }

  async loss(sequences: Sequence[]) {
    let sum = 0;
    let count = 0;
    for (const s of sequences) {
      const r = this.forward(s, 0, false);
      sum += r.sum;
      count += r.count;
    }
    return { sum, count };
  }

  /** The lowest layer the backward pass must reach, or -1 to reach the embeddings. */
  private lowestLayer(): number {
    let lowest = this.probeLayer ?? this.c.layers.length;
    for (const { name } of this.targets) {
      if (name === 'tok_embeddings') return -1;
      const m = /^layer_(\d+)\./.exec(name);
      if (m) lowest = Math.min(lowest, Number(m[1]));
    }
    return lowest;
  }

  async accumulate(sequences: Sequence[]) {
    for (const g of this.grads.values()) g.fill(0);
    const total = sequences.reduce((n, s) => n + s.targets.filter((x) => x >= 0).length, 0);
    const scale = total ? 1 / total : 0;
    let sum = 0;
    let count = 0;
    for (const seq of sequences) {
      const f = this.forward(seq, scale, true);
      sum += f.sum;
      count += f.count;
      this.backward(seq, f);
    }
    return { sum, count };
  }

  private backward(seq: Sequence, f: ReturnType<CpuTrainer['forward']>) {
    const c = this.c;
    const { dim, headSize: hs } = c;
    const kv = kvDim(c);
    const T = f.T;
    const g = (name: string) => this.grads.get(name);
    const clsName = classifierName(c);
    const dXF = new Float64Array(T * dim);
    linearBackX(dXF, f.dlogits!, this.w(clsName), T, dim, c.vocabSize);
    const dCls = g(clsName);
    if (dCls) linearBackW(dCls, f.dlogits!, f.XF, T, dim, c.vocabSize);
    let dX = new Float64Array(T * dim);
    rmsnormBack(dX, dXF, f.Xfinal, this.w('norm'), f.rstdF, T, dim, g('norm'));
    const lowest = this.lowestLayer();
    for (let l = c.layers.length - 1; l >= Math.max(0, lowest); l--) {
      const layer = c.layers[l];
      const L = f.layers[l];
      const p = `layer_${l}`;
      const qd = layer.nHeads * hs;
      if (l === this.probeLayer) this.lastProbe = Float64Array.from(dX);
      // MLP: Xout = Xmid + down · H
      if (layer.hidden) {
        const dH = new Float64Array(T * layer.hidden);
        linearBackX(dH, dX, this.w(`${p}.mlp.down`), T, layer.hidden, dim);
        const dDown = g(`${p}.mlp.down`);
        if (dDown) linearBackW(dDown, dX, L.H, T, layer.hidden, dim);
        const dG = new Float64Array(dH.length);
        const dU = new Float64Array(dH.length);
        for (let i = 0; i < dH.length; i++) {
          const s = sigmoid(L.G[i]);
          dG[i] = dH[i] * L.U[i] * s * (1 + L.G[i] * (1 - s));
          dU[i] = dH[i] * L.G[i] * s;
        }
        const dXN2 = new Float64Array(T * dim);
        linearBackX(dXN2, dG, this.w(`${p}.mlp.gate`), T, dim, layer.hidden);
        linearBackX(dXN2, dU, this.w(`${p}.mlp.up`), T, dim, layer.hidden);
        const dGate = g(`${p}.mlp.gate`);
        if (dGate) linearBackW(dGate, dG, L.XN2, T, dim, layer.hidden);
        const dUp = g(`${p}.mlp.up`);
        if (dUp) linearBackW(dUp, dU, L.XN2, T, dim, layer.hidden);
        rmsnormBack(dX, dXN2, L.Xmid, this.w(`${p}.mlp_norm`), L.rstd2, T, dim, g(`${p}.mlp_norm`));
      }
      // Attention: Xmid = X + wo · O
      if (layer.nHeads) {
        const dO = new Float64Array(T * qd);
        linearBackX(dO, dX, this.w(`${p}.attn.wo`), T, qd, dim);
        const dWo = g(`${p}.attn.wo`);
        if (dWo) linearBackW(dWo, dX, L.O, T, qd, dim);
        const dQ = new Float64Array(T * qd);
        const dK = new Float64Array(T * kv);
        const dV = new Float64Array(T * kv);
        const scaleQK = 1 / Math.sqrt(hs);
        for (let h = 0; h < layer.nHeads; h++) {
          const ko = layer.headKv[h] * hs;
          for (let t = 0; t < T; t++) {
            const P = L.P[h][t];
            const dP = new Float64Array(t + 1);
            let dot = 0;
            for (let s = 0; s <= t; s++) {
              let d = 0;
              for (let i = 0; i < hs; i++) {
                const go = dO[t * qd + h * hs + i];
                d += go * L.V[s * kv + ko + i];
                dV[s * kv + ko + i] += P[s] * go;
              }
              dP[s] = d;
              dot += P[s] * d;
            }
            for (let s = 0; s <= t; s++) {
              const dS = P[s] * (dP[s] - dot) * scaleQK;
              for (let i = 0; i < hs; i++) {
                dQ[t * qd + h * hs + i] += dS * L.K[s * kv + ko + i];
                dK[s * kv + ko + i] += dS * L.Q[t * qd + h * hs + i];
              }
            }
          }
        }
        rope(dQ, T, qd, hs, c.ropeTheta, -1);
        rope(dK, T, kv, hs, c.ropeTheta, -1);
        const dXN1 = new Float64Array(T * dim);
        linearBackX(dXN1, dQ, this.w(`${p}.attn.wq`), T, dim, qd);
        linearBackX(dXN1, dK, this.w(`${p}.attn.wk`), T, dim, kv);
        linearBackX(dXN1, dV, this.w(`${p}.attn.wv`), T, dim, kv);
        for (const [name, dY, n] of [['wq', dQ, qd], ['wk', dK, kv], ['wv', dV, kv]] as const) {
          const dW = g(`${p}.attn.${name}`);
          if (dW) linearBackW(dW, dY, L.XN1, T, dim, n);
        }
        rmsnormBack(dX, dXN1, L.X, this.w(`${p}.attn_norm`), L.rstd1, T, dim, g(`${p}.attn_norm`));
      }
    }
    const dEmb = g('tok_embeddings');
    if (dEmb && lowest < 0) {
      seq.tokens.forEach((tok, t) => {
        for (let j = 0; j < dim; j++) dEmb[tok * dim + j] += dX[t * dim + j];
      });
    }
    dX = new Float64Array(0);
  }

  async capture(seq: Sequence, layer: number) {
    if (!(layer >= 0 && layer < this.c.layers.length)) throw new RangeError(`No layer ${layer}`);
    const f = this.forward(seq, 0, false);
    const L = f.layers[layer];
    const residual = layer + 1 < this.c.layers.length ? f.layers[layer + 1].X : f.Xfinal;
    return { key: Float32Array.from(L.H), mlpInput: Float32Array.from(L.XN2), residual: Float32Array.from(residual) };
  }

  async probe() {
    if (this.probeLayer === undefined) throw new Error('This trainer has no probe layer');
    if (!this.lastProbe) throw new Error('Run accumulate() first');
    return Float32Array.from(this.lastProbe);
  }

  async gradients() {
    const out = new Map<string, Float32Array>();
    for (const [name, g] of this.grads) out.set(name, Float32Array.from(g));
    return out;
  }

  async update(o: AdamOptions) {
    let sq = 0;
    for (const g of this.grads.values()) for (const x of g) sq += x * x;
    const gradNorm = Math.sqrt(sq);
    if (!Number.isFinite(gradNorm)) return { gradNorm, skipped: true };
    const gscale = o.clip > 0 && gradNorm > o.clip ? o.clip / gradNorm : 1;
    this.t++;
    const bc1 = 1 - o.beta1 ** this.t;
    const bc2 = 1 - o.beta2 ** this.t;
    for (const { name, region } of this.targets) {
      const w = this.w(name);
      const g = this.grads.get(name)!;
      const m = this.m.get(name)!;
      const v = this.v.get(name)!;
      const cols = this.be.shape(name)[1];
      for (let r = region.rowStart; r < region.rowEnd; r++) {
        for (let cc = region.colStart; cc < region.colEnd; cc++) {
          const i = r * cols + cc;
          const gi = g[i] * gscale;
          m[i] = o.beta1 * m[i] + (1 - o.beta1) * gi;
          v[i] = o.beta2 * v[i] + (1 - o.beta2) * gi * gi;
          w[i] = w[i] - o.lr * ((m[i] / bc1) / (Math.sqrt(v[i] / bc2) + o.eps) + o.weightDecay * w[i]);
        }
      }
    }
    return { gradNorm, skipped: false };
  }

  destroy() {
    this.grads.clear();
    this.m.clear();
    this.v.clear();
  }
}
