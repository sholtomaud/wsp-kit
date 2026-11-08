/**
 * Training on the GPU. The forward pass, cross-entropy and backward pass for
 * one window are a fixed list of dispatches built once; per window the CPU
 * only writes the token ids, the targets and a loss scale, then submits.
 * Activations stay in VRAM; only the summed loss and the gradient norm
 * (for clipping) are read back each step.
 */
import type { AdamOptions, Sequence, TrainTarget, Trainer } from '../backend.ts';
import { classifierName, kvDim, type ModelConfig } from '../model/types.ts';
import { grid, U, type Binding, type KernelName, type Param, type WebGpuBackend } from './backend.ts';

interface Dispatch {
  kernel: KernelName;
  group: GPUBindGroup;
  x: number;
  y: number;
}

interface Trainable {
  target: TrainTarget;
  grad: GPUBuffer;
  m: GPUBuffer;
  v: GPUBuffer;
  size: number;
  adamParams: GPUBuffer;
}

const STATS_SLOT = 256 * 8 * 4; // bytes of partial sums per tensor (256 workgroups × 8 floats)

export class GpuTrainer implements Trainer {
  readonly length: number;
  private readonly be: WebGpuBackend;
  private readonly c: ModelConfig;
  private owned: GPUBuffer[] = [];
  private forwardList: Dispatch[] = [];
  private backwardList: Dispatch[] = [];
  private adamList: Dispatch[] = [];
  private statsList: Dispatch[] = [];
  private trainables = new Map<string, Trainable>();
  private tokens: GPUBuffer;
  private goals: GPUBuffer;
  private seq: GPUBuffer;
  private lossAcc: GPUBuffer;
  private partials?: GPUBuffer;
  private step = 0;

  constructor(be: WebGpuBackend, c: ModelConfig, length: number, targets: TrainTarget[]) {
    if (!(Number.isInteger(length) && length >= 1 && length <= c.seqLen)) throw new RangeError(`Window length must be 1…${c.seqLen}`);
    this.be = be;
    this.c = c;
    this.length = length;
    const T = length;
    const { dim, headSize: hs } = c;
    const kv = kvDim(c);
    const own = (b: GPUBuffer) => (this.owned.push(b), b);
    const f32 = (n: number, label: string) => own(be.storage(Math.max(1, n) * 4, label));
    const uni = (params: Param[]) => own(be.uniform(params));
    const add = (list: Dispatch[], kernel: KernelName, buffers: Binding[], x: number, y = 1) => {
      if (x > 0 && y > 0) list.push({ kernel, group: be.bindGroup(kernel, buffers), x, y });
    };
    const elementwise = (list: Dispatch[], kernel: KernelName, n: number, params: Param[], buffers: Binding[]) => {
      if (!n) return;
      const g = grid(n);
      add(list, kernel, [uni([...params, ['u', g.strideX]]), ...buffers], g.x, g.y);
    };
    /** C[M,N] (+)= A·B; see tMatmul. */
    const matmul = (list: Dispatch[], A: GPUBuffer, B: GPUBuffer, C: GPUBuffer, M: number, N: number, K: number, transA: boolean, transB: boolean, accumulate: boolean) => {
      if (!M || !N) return;
      const params = uni([['u', M], ['u', N], ['u', K], ['u', transA ? 1 : 0], ['u', transB ? 1 : 0], ['u', accumulate ? 1 : 0]]);
      add(list, 'tMatmul', [params, A, B, C], Math.ceil(N / 16), Math.ceil(M / 16));
    };
    // y[T,out] = x[T,in] · Wᵀ   (W is [out, in])
    const linear = (list: Dispatch[], X: GPUBuffer, W: GPUBuffer, Y: GPUBuffer, nIn: number, nOut: number, accumulate = false) =>
      matmul(list, X, W, Y, T, nOut, nIn, false, true, accumulate);
    // dX[T,in] (+)= dY[T,out] · W
    const linearBackX = (list: Dispatch[], dY: GPUBuffer, W: GPUBuffer, dX: GPUBuffer, nIn: number, nOut: number, accumulate: boolean) =>
      matmul(list, dY, W, dX, T, nIn, nOut, false, false, accumulate);
    // dW[out,in] += dY[T,out]ᵀ · X[T,in]
    const linearBackW = (list: Dispatch[], dY: GPUBuffer, X: GPUBuffer, dW: GPUBuffer, nIn: number, nOut: number) =>
      matmul(list, dY, X, dW, nOut, nIn, T, true, false, true);
    const rmsF = (list: Dispatch[], X: GPUBuffer, w: GPUBuffer, Y: GPUBuffer, rstd: GPUBuffer) =>
      add(list, 'tRmsF', [uni([['u', dim], ['f', c.normEps]]), X, w, Y, rstd], T);
    const rmsB = (list: Dispatch[], dY: GPUBuffer, X: GPUBuffer, w: GPUBuffer, rstd: GPUBuffer, dX: GPUBuffer, accumulate: boolean, dw?: GPUBuffer) => {
      add(list, 'tRmsB', [uni([['u', dim], ['u', accumulate ? 1 : 0]]), dY, X, w, rstd, dX], T);
      if (dw) add(list, 'tRmsBW', [uni([['u', T], ['u', dim]]), dY, X, rstd, dw], Math.ceil(dim / 64));
    };
    // tRope's Params put strideX before the floats, so it is built by hand.
    const ropeAt = (list: Dispatch[], buf: GPUBuffer, width: number, sign: number) => {
      const n = (T * width) / 2;
      if (!n) return;
      const g = grid(n);
      add(list, 'tRope', [uni([['u', T], ['u', width], ['u', hs], ['u', g.strideX], ['f', c.ropeTheta], ['f', sign]]), buf], g.x, g.y);
    };

    // ---- trainable tensors: gradient, Adam moments, update params
    for (const target of targets) {
      if (this.trainables.has(target.name)) throw new Error(`${target.name} is named twice`);
      const [rows, cols] = be.shape(target.name);
      const size = rows * cols;
      this.trainables.set(target.name, {
        target,
        size,
        grad: f32(size, `grad ${target.name}`),
        m: f32(size, `adam-m ${target.name}`),
        v: f32(size, `adam-v ${target.name}`),
        adamParams: own(be.createBuffer(64, U.UNIFORM | U.COPY_DST, `adam ${target.name}`)),
      });
    }
    const grad = (name: string) => this.trainables.get(name)?.grad;

    // ---- per-window inputs
    this.tokens = own(be.storage(T * 4, 'tokens'));
    this.goals = own(be.storage(T * 4, 'targets'));
    this.seq = own(be.createBuffer(16, U.UNIFORM | U.COPY_DST, 'seq'));
    this.lossAcc = f32(T, 'loss');

    // ---- forward
    const w = (name: string) => be.buffer(name);
    const xs: GPUBuffer[] = [f32(T * dim, 'x0')];
    elementwise(this.forwardList, 'tGather', T * dim, [['u', T], ['u', dim]], [this.tokens, w('tok_embeddings'), xs[0]]);
    interface Saved {
      XN1: GPUBuffer; rstd1: GPUBuffer; Q: GPUBuffer; K: GPUBuffer; V: GPUBuffer; att: GPUBuffer; O: GPUBuffer;
      Xmid: GPUBuffer; XN2: GPUBuffer; rstd2: GPUBuffer; G: GPUBuffer; Up: GPUBuffer; H: GPUBuffer; headKv: GPUBuffer;
    }
    const saved: Saved[] = [];
    c.layers.forEach((layer, l) => {
      const p = `layer_${l}`;
      const qd = layer.nHeads * hs;
      const X = xs[l];
      const s: Saved = {
        XN1: f32(T * dim, `${p} xn1`),
        rstd1: f32(T, `${p} rstd1`),
        Q: f32(T * qd, `${p} q`),
        K: f32(T * kv, `${p} k`),
        V: f32(T * kv, `${p} v`),
        att: f32(layer.nHeads * T * T, `${p} att`),
        O: f32(T * qd, `${p} o`),
        Xmid: f32(T * dim, `${p} xmid`),
        XN2: f32(T * dim, `${p} xn2`),
        rstd2: f32(T, `${p} rstd2`),
        G: f32(T * layer.hidden, `${p} gate`),
        Up: f32(T * layer.hidden, `${p} up`),
        H: f32(T * layer.hidden, `${p} h`),
        headKv: own(be.storage(Uint32Array.from(layer.headKv.length ? layer.headKv : [0]), `${p} headKv`)),
      };
      saved.push(s);
      const F = this.forwardList;
      elementwise(F, 'tCopy', T * dim, [['u', T * dim]], [X, s.Xmid]);
      if (layer.nHeads) {
        rmsF(F, X, w(`${p}.attn_norm`), s.XN1, s.rstd1);
        linear(F, s.XN1, w(`${p}.attn.wq`), s.Q, dim, qd);
        linear(F, s.XN1, w(`${p}.attn.wk`), s.K, dim, kv);
        linear(F, s.XN1, w(`${p}.attn.wv`), s.V, dim, kv);
        ropeAt(F, s.Q, qd, 1);
        ropeAt(F, s.K, kv, 1);
        add(F, 'tAttnF', [uni([['u', T], ['u', hs], ['u', qd], ['u', kv]]), s.Q, s.K, s.V, s.headKv, s.att, s.O], T, layer.nHeads);
        linear(F, s.O, w(`${p}.attn.wo`), s.Xmid, qd, dim, true);
      }
      const Xout = f32(T * dim, `x${l + 1}`);
      xs.push(Xout);
      elementwise(F, 'tCopy', T * dim, [['u', T * dim]], [s.Xmid, Xout]);
      if (layer.hidden) {
        rmsF(F, s.Xmid, w(`${p}.mlp_norm`), s.XN2, s.rstd2);
        linear(F, s.XN2, w(`${p}.mlp.gate`), s.G, dim, layer.hidden);
        linear(F, s.XN2, w(`${p}.mlp.up`), s.Up, dim, layer.hidden);
        elementwise(F, 'tSwigluF', T * layer.hidden, [['u', T * layer.hidden]], [s.G, s.Up, s.H]);
        linear(F, s.H, w(`${p}.mlp.down`), Xout, layer.hidden, dim, true);
      }
    });
    const XL = xs[xs.length - 1];
    const XF = f32(T * dim, 'xf');
    const rstdF = f32(T, 'rstdf');
    const logits = f32(T * c.vocabSize, 'logits');
    const clsName = classifierName(c);
    rmsF(this.forwardList, XL, w('norm'), XF, rstdF);
    linear(this.forwardList, XF, w(clsName), logits, dim, c.vocabSize);
    add(this.forwardList, 'tCrossEntropy', [uni([['u', c.vocabSize]]), this.seq, this.goals, logits, this.lossAcc], T);

    // ---- backward (logits now hold dLoss/dlogits)
    if (this.trainables.size) {
      const B = this.backwardList;
      const maxQ = Math.max(1, ...c.layers.map((l) => l.nHeads * hs));
      const maxHidden = Math.max(1, ...c.layers.map((l) => l.hidden));
      const maxHeads = Math.max(1, ...c.layers.map((l) => l.nHeads));
      const dXF = f32(T * dim, 'dxf');
      const dX = f32(T * dim, 'dx');
      const dXN = f32(T * dim, 'dxn');
      const dO = f32(T * maxQ, 'do');
      const dQ = f32(T * maxQ, 'dq');
      const dK = f32(T * kv, 'dk');
      const dV = f32(T * kv, 'dv');
      const dS = f32(maxHeads * T * T, 'ds');
      const dH = f32(T * maxHidden, 'dh');
      const dG = f32(T * maxHidden, 'dg');
      const dU = f32(T * maxHidden, 'du');
      linearBackX(B, logits, w(clsName), dXF, dim, c.vocabSize, false);
      const dCls = grad(clsName);
      if (dCls) linearBackW(B, logits, XF, dCls, dim, c.vocabSize);
      rmsB(B, dXF, XL, w('norm'), rstdF, dX, false, grad('norm'));
      const lowest = lowestLayer(c, targets);
      for (let l = c.layers.length - 1; l >= Math.max(0, lowest); l--) {
        const layer = c.layers[l];
        const s = saved[l];
        const p = `layer_${l}`;
        const qd = layer.nHeads * hs;
        if (layer.hidden) {
          linearBackX(B, dX, w(`${p}.mlp.down`), dH, layer.hidden, dim, false);
          const dDown = grad(`${p}.mlp.down`);
          if (dDown) linearBackW(B, dX, s.H, dDown, layer.hidden, dim);
          elementwise(B, 'tSwigluB', T * layer.hidden, [['u', T * layer.hidden]], [dH, s.G, s.Up, dG, dU]);
          linearBackX(B, dG, w(`${p}.mlp.gate`), dXN, dim, layer.hidden, false);
          linearBackX(B, dU, w(`${p}.mlp.up`), dXN, dim, layer.hidden, true);
          const dGate = grad(`${p}.mlp.gate`);
          if (dGate) linearBackW(B, dG, s.XN2, dGate, dim, layer.hidden);
          const dUp = grad(`${p}.mlp.up`);
          if (dUp) linearBackW(B, dU, s.XN2, dUp, dim, layer.hidden);
          rmsB(B, dXN, s.Xmid, w(`${p}.mlp_norm`), s.rstd2, dX, true, grad(`${p}.mlp_norm`));
        }
        if (layer.nHeads) {
          linearBackX(B, dX, w(`${p}.attn.wo`), dO, qd, dim, false);
          const dWo = grad(`${p}.attn.wo`);
          if (dWo) linearBackW(B, dX, s.O, dWo, qd, dim);
          const shape: Param[] = [['u', T], ['u', hs], ['u', qd], ['u', kv]];
          elementwise(B, 'tAttnDV', T * kv, [...shape, ['u', layer.nHeads]], [s.att, dO, s.headKv, dV]);
          add(B, 'tAttnDS', [uni(shape), s.att, dO, s.V, s.headKv, dS], T, layer.nHeads);
          const scale = 1 / Math.sqrt(hs);
          {
            const g = grid(T * qd);
            add(B, 'tAttnDQ', [uni([...shape, ['u', g.strideX], ['f', scale]]), dS, s.K, s.headKv, dQ], g.x, g.y);
          }
          {
            const g = grid(T * kv);
            add(B, 'tAttnDK', [uni([...shape, ['u', layer.nHeads], ['u', g.strideX], ['f', scale]]), dS, s.Q, s.headKv, dK], g.x, g.y);
          }
          ropeAt(B, dQ, qd, -1);
          ropeAt(B, dK, kv, -1);
          linearBackX(B, dQ, w(`${p}.attn.wq`), dXN, dim, qd, false);
          linearBackX(B, dK, w(`${p}.attn.wk`), dXN, dim, kv, true);
          linearBackX(B, dV, w(`${p}.attn.wv`), dXN, dim, kv, true);
          for (const [name, dY, n] of [['wq', dQ, qd], ['wk', dK, kv], ['wv', dV, kv]] as const) {
            const dW = grad(`${p}.attn.${name}`);
            if (dW) linearBackW(B, dY, s.XN1, dW, dim, n);
          }
          rmsB(B, dXN, xs[l], w(`${p}.attn_norm`), s.rstd1, dX, true, grad(`${p}.attn_norm`));
        }
      }
      const dEmb = grad('tok_embeddings');
      if (dEmb && lowest < 0) add(B, 'tEmbB', [uni([['u', T], ['u', dim]]), this.tokens, dX, dEmb], Math.ceil(dim / 64));

      // ---- gradient norm (partial sums per tensor) and AdamW
      this.partials = f32((this.trainables.size * STATS_SLOT) / 4, 'grad-norm partials');
      [...this.trainables.values()].forEach((t, k) => {
        add(this.statsList, 'stats', [uni([['u', t.size]]), t.grad, { buffer: this.partials!, offset: k * STATS_SLOT, size: STATS_SLOT }], Math.max(1, Math.min(256, Math.ceil(t.size / 256))));
        const r = t.target.region;
        const n = (r.rowEnd - r.rowStart) * (r.colEnd - r.colStart);
        const g = grid(n);
        if (n) this.adamList.push({ kernel: 'tAdamW', group: be.bindGroup('tAdamW', [t.adamParams, w(t.target.name), t.grad, t.m, t.v]), x: g.x, y: g.y });
      });
    }
  }

  private encode(lists: Dispatch[][]) {
    const encoder = this.be.device.createCommandEncoder({ label: 'train' });
    const pass = encoder.beginComputePass();
    for (const list of lists) {
      for (const d of list) {
        pass.setPipeline(this.be.pipeline(d.kernel));
        pass.setBindGroup(0, d.group);
        pass.dispatchWorkgroups(d.x, d.y);
      }
    }
    pass.end();
    return encoder;
  }

  private writeWindow(seq: Sequence, scale: number) {
    const T = this.length;
    if (seq.tokens.length !== seq.targets.length) throw new Error('tokens and targets differ in length');
    if (seq.tokens.length > T) throw new RangeError(`A window of ${seq.tokens.length} tokens exceeds the trainer's ${T}`);
    const tokens = new Uint32Array(T);
    const goals = new Int32Array(T).fill(-1);
    seq.tokens.forEach((tok, i) => {
      if (!(tok >= 0 && tok < this.c.vocabSize)) throw new RangeError(`Token ${tok} is outside the vocabulary`);
      tokens[i] = tok;
      goals[i] = seq.targets[i];
    });
    const q = this.be.device.queue;
    q.writeBuffer(this.tokens, 0, tokens);
    q.writeBuffer(this.goals, 0, goals);
    q.writeBuffer(this.seq, 0, new Float32Array([scale, 0, 0, 0]));
  }

  private clearLoss() {
    const encoder = this.be.device.createCommandEncoder();
    encoder.clearBuffer(this.lossAcc);
    return encoder;
  }

  private async readLoss(): Promise<number> {
    const rows = new Float32Array(await this.be.readBuffer(this.lossAcc, 0, this.length * 4));
    let sum = 0;
    for (const v of rows) sum += v;
    return sum;
  }

  async loss(sequences: Sequence[]) {
    this.be.device.queue.submit([this.clearLoss().finish()]);
    let count = 0;
    for (const s of sequences) {
      this.writeWindow(s, 0);
      this.be.device.queue.submit([this.encode([this.forwardList]).finish()]);
      count += s.targets.filter((t) => t >= 0).length;
    }
    return { sum: await this.readLoss(), count };
  }

  async accumulate(sequences: Sequence[]) {
    const encoder = this.clearLoss();
    for (const t of this.trainables.values()) encoder.clearBuffer(t.grad);
    this.be.device.queue.submit([encoder.finish()]);
    const total = sequences.reduce((n, s) => n + s.targets.filter((t) => t >= 0).length, 0);
    const scale = total ? 1 / total : 0;
    for (const s of sequences) {
      this.writeWindow(s, scale);
      this.be.device.queue.submit([this.encode([this.forwardList, this.backwardList]).finish()]);
    }
    return { sum: await this.readLoss(), count: total };
  }

  async gradients() {
    const out = new Map<string, Float32Array>();
    for (const [name, t] of this.trainables) out.set(name, new Float32Array(await this.be.readBuffer(t.grad, 0, t.size * 4)));
    return out;
  }

  async update(o: AdamOptions) {
    if (!this.trainables.size || !this.partials) return { gradNorm: 0, skipped: true };
    this.be.device.queue.submit([this.encode([this.statsList]).finish()]);
    const partials = new Float32Array(await this.be.readBuffer(this.partials, 0, this.trainables.size * STATS_SLOT));
    let sq = 0;
    let bad = 0;
    [...this.trainables.values()].forEach((t, k) => {
      const groups = Math.max(1, Math.min(256, Math.ceil(t.size / 256)));
      for (let g = 0; g < groups; g++) {
        const off = (k * STATS_SLOT) / 4 + g * 8;
        sq += partials[off + 1];
        bad += partials[off + 5];
      }
    });
    const gradNorm = Math.sqrt(sq);
    if (bad > 0 || !Number.isFinite(gradNorm)) return { gradNorm: bad > 0 ? NaN : gradNorm, skipped: true };
    const gscale = o.clip > 0 && gradNorm > o.clip ? o.clip / gradNorm : 1;
    this.step++;
    const bc1 = 1 - o.beta1 ** this.step;
    const bc2 = 1 - o.beta2 ** this.step;
    for (const t of this.trainables.values()) {
      const r = t.target.region;
      const nCols = r.colEnd - r.colStart;
      const n = (r.rowEnd - r.rowStart) * nCols;
      const g = grid(n);
      const params: Param[] = [
        ['u', this.be.shape(t.target.name)[1]], ['u', r.rowStart], ['u', r.colStart], ['u', nCols], ['u', n], ['u', g.strideX],
        ['f', o.lr], ['f', o.beta1], ['f', o.beta2], ['f', o.eps], ['f', o.weightDecay], ['f', bc1], ['f', bc2], ['f', gscale],
      ];
      const data = new ArrayBuffer(64);
      const view = new DataView(data);
      params.forEach(([type, v], i) => (type === 'u' ? view.setUint32(i * 4, v, true) : view.setFloat32(i * 4, v, true)));
      this.be.device.queue.writeBuffer(t.adamParams, 0, data);
    }
    this.be.device.queue.submit([this.encode([this.adamList]).finish()]);
    return { gradNorm, skipped: false };
  }

  destroy() {
    for (const b of this.owned) b.destroy();
    this.owned = [];
    this.trainables.clear();
  }
}

/** The lowest layer the backward pass must reach, or -1 to reach the embeddings. */
export function lowestLayer(c: ModelConfig, targets: TrainTarget[]): number {
  let lowest = c.layers.length;
  for (const { name } of targets) {
    if (name === 'tok_embeddings') return -1;
    const m = /^layer_(\d+)\./.exec(name);
    if (m) lowest = Math.min(lowest, Number(m[1]));
  }
  return lowest;
}
