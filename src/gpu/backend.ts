/**
 * WebGPU backend: every tensor is a GPU storage buffer and every operation a
 * compute shader, so edits never round-trip the weights through the CPU. Only
 * what an LLM asks to see (stats, a slice, a list of coordinates, logits) is
 * read back.
 *
 * Patterned on fluxkit's engine (tsrs-monaco/packages/fluxkit), without
 * depending on it.
 */
import { checkRegion, type Backend, type TrainTarget, type Fill, type Hits, type Region, type RunOptions, type Runner, type Stats } from '../backend.ts';
import { classifierName, kvDim, type ModelConfig, type Shape } from '../model/types.ts';
import { KERNELS, type KernelName as BaseKernelName } from './kernels.ts';
import { TRAIN_KERNELS, type TrainKernelName } from './train-kernels.ts';
import { GpuTrainer } from './trainer.ts';

export type KernelName = BaseKernelName | TrainKernelName;
const ALL_KERNELS: Record<KernelName, string> = { ...KERNELS, ...TRAIN_KERNELS };

// Flag values from the WebGPU spec (TypeScript's DOM lib lacks the namespaces).
export const U = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;
const MAP_READ = 0x1;
const TENSOR_USAGE = U.STORAGE | U.COPY_SRC | U.COPY_DST;

export class WgslError extends Error {
  readonly source: string;
  constructor(label: string, messages: string[], source: string) {
    super(`WGSL errors in ${label}:\n${messages.join('\n')}`);
    this.name = 'WgslError';
    this.source = source;
  }
}

export type Param = ['u' | 'i' | 'f', number];
export type Binding = GPUBuffer | { buffer: GPUBuffer; offset: number; size: number };

export function packUniform(params: Param[]): ArrayBuffer {
  const buffer = new ArrayBuffer(Math.max(16, Math.ceil(params.length / 4) * 16));
  const view = new DataView(buffer);
  params.forEach(([type, v], i) => {
    if (type === 'u') view.setUint32(i * 4, v >>> 0, true);
    else if (type === 'i') view.setInt32(i * 4, v, true);
    else view.setFloat32(i * 4, v, true);
  });
  return buffer;
}

/** A 2D grid of 64-thread workgroups covering n threads. */
export function grid(n: number): { x: number; y: number; strideX: number } {
  const groups = Math.max(1, Math.ceil(n / 64));
  const x = Math.min(groups, 65535);
  return { x, y: Math.ceil(groups / x), strideX: x * 64 };
}

interface Tensor {
  shape: Shape;
  buffer: GPUBuffer;
}

export interface WebGpuInit {
  device: GPUDevice;
  description?: string;
}

/** Requests an adapter and a device with the adapter's largest buffer limits. */
export async function requestDevice(): Promise<{ device: GPUDevice; description: string }> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) throw new Error('WebGPU is not available in this browser (use a recent Chrome, Edge or Safari)');
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter was found');
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
    },
  });
  const info = adapter.info;
  const name = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' ');
  return { device, description: `WebGPU${name ? ` (${name})` : ''}` };
}

export class WebGpuBackend implements Backend {
  readonly kind = 'webgpu' as const;
  readonly description: string;
  readonly device: GPUDevice;
  private tensors = new Map<string, Tensor>();
  private pipelines = new Map<KernelName, GPUComputePipeline>();

  constructor({ device, description = 'WebGPU' }: WebGpuInit) {
    this.device = device;
    this.description = description;
  }

  static async create(): Promise<WebGpuBackend> {
    const { device, description } = await requestDevice();
    const backend = new WebGpuBackend({ device, description });
    await backend.compileAll();
    return backend;
  }

  /** Compiles every kernel up front, so a WGSL error surfaces at start-up. */
  async compileAll() {
    for (const name of Object.keys(ALL_KERNELS) as KernelName[]) {
      const code = ALL_KERNELS[name];
      const module = this.device.createShaderModule({ label: name, code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new WgslError(name, errors.map((m) => `  ${name}:${m.lineNum}:${m.linePos} ${m.message}`), code);
      this.device.pushErrorScope('validation');
      const pipeline = await this.device.createComputePipelineAsync({ label: name, layout: 'auto', compute: { module, entryPoint: 'main' } });
      const error = await this.device.popErrorScope();
      if (error) throw new Error(`${name}: ${error.message}`);
      this.pipelines.set(name, pipeline);
    }
  }

  pipeline(name: KernelName): GPUComputePipeline {
    const p = this.pipelines.get(name);
    if (!p) throw new Error(`Kernel ${name} is not compiled`);
    return p;
  }

  private get(name: string): Tensor {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`No tensor named ${name}`);
    return t;
  }

  /** The GPU buffer behind a tensor (for the forward pass). */
  buffer(name: string): GPUBuffer {
    return this.get(name).buffer;
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

  createBuffer(bytes: number, usage: number, label?: string): GPUBuffer {
    return this.device.createBuffer({ label, size: Math.max(16, Math.ceil(bytes / 4) * 4), usage });
  }

  uniform(params: Param[], label?: string): GPUBuffer {
    const data = packUniform(params);
    const buffer = this.createBuffer(data.byteLength, U.UNIFORM | U.COPY_DST, label);
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  storage(data: ArrayBufferView | number, label?: string, extra = 0): GPUBuffer {
    const bytes = typeof data === 'number' ? data : data.byteLength;
    const buffer = this.createBuffer(bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST | extra, label);
    if (typeof data !== 'number' && data.byteLength) {
      this.device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    }
    return buffer;
  }

  bindGroup(kernel: KernelName, buffers: Binding[]): GPUBindGroup {
    return this.device.createBindGroup({
      label: kernel,
      layout: this.pipeline(kernel).getBindGroupLayout(0),
      entries: buffers.map((b, binding) => ({ binding, resource: 'offset' in b ? b : { buffer: b } })),
    });
  }

  /** Runs one kernel and frees its temporary buffers once queued. */
  private dispatch(kernel: KernelName, buffers: GPUBuffer[], groups: { x: number; y?: number }, temporaries: GPUBuffer[] = []) {
    const encoder = this.device.createCommandEncoder({ label: kernel });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline(kernel));
    pass.setBindGroup(0, this.bindGroup(kernel, buffers));
    pass.dispatchWorkgroups(groups.x, groups.y ?? 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    // destroy() waits for the submitted work that uses the buffer.
    for (const b of temporaries) b.destroy();
  }

  /** Copies bytes out of a GPU buffer. */
  async readBuffer(buffer: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
    if (bytes === 0) return new ArrayBuffer(0);
    const staging = this.device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
    const encoder = this.device.createCommandEncoder({ label: 'read' });
    encoder.copyBufferToBuffer(buffer, offset, staging, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_READ);
    const out = staging.getMappedRange().slice(0);
    staging.destroy();
    return out;
  }

  alloc(name: string, shape: Shape, data?: Float32Array) {
    const n = shape[0] * shape[1];
    if (data && data.length !== n) throw new Error(`${name}: ${data.length} values for a ${shape[0]}×${shape[1]} tensor`);
    const old = this.tensors.get(name);
    const buffer = this.createBuffer(n * 4, TENSOR_USAGE, name);
    if (data && n) this.device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    this.tensors.set(name, { shape: [...shape], buffer });
    old?.buffer.destroy();
  }

  free(name: string) {
    this.tensors.get(name)?.buffer.destroy();
    this.tensors.delete(name);
  }

  async copy(src: string, dst: string) {
    const s = this.get(src);
    const d = this.tensors.get(dst);
    if (!d || d.shape[0] !== s.shape[0] || d.shape[1] !== s.shape[1]) this.alloc(dst, s.shape);
    const n = s.shape[0] * s.shape[1] * 4;
    if (!n) return;
    const encoder = this.device.createCommandEncoder({ label: 'copy' });
    encoder.copyBufferToBuffer(s.buffer, 0, this.get(dst).buffer, 0, n);
    this.device.queue.submit([encoder.finish()]);
  }

  rename(from: string, to: string) {
    const t = this.get(from);
    this.tensors.get(to)?.buffer.destroy();
    this.tensors.delete(from);
    this.tensors.set(to, t);
  }

  async matvec(name: string, vector: Float32Array) {
    const t = this.get(name);
    const [rows, cols] = t.shape;
    if (vector.length !== cols) throw new Error(`${name}: vector of ${vector.length} for ${cols} columns`);
    const g = grid(rows);
    const params = this.uniform([['u', rows], ['u', cols], ['u', 0], ['u', 0], ['u', g.strideX]]);
    const step = this.uniform([['u', 0], ['u', 0], ['i', -1], ['u', 0]]);
    const v = this.storage(vector, 'vector');
    const out = this.storage(rows * 4, 'matvec-out');
    this.dispatch('matvec', [params, step, t.buffer, v, out], g, [params, step, v]);
    const result = new Float32Array(await this.readBuffer(out, 0, rows * 4));
    out.destroy();
    return result;
  }

  async copyRegion(src: string, dst: string, r: Region, dstRow: number, dstCol: number) {
    const s = this.get(src);
    const d = this.get(dst);
    checkRegion(s.shape, r, src);
    const rows = r.rowEnd - r.rowStart;
    const cols = r.colEnd - r.colStart;
    checkRegion(d.shape, { rowStart: dstRow, rowEnd: dstRow + rows, colStart: dstCol, colEnd: dstCol + cols }, dst);
    const n = rows * cols;
    if (!n) return;
    const g = grid(n);
    const params = this.uniform([
      ['u', s.shape[1]], ['u', d.shape[1]], ['u', r.rowStart], ['u', r.colStart],
      ['u', dstRow], ['u', dstCol], ['u', cols], ['u', n], ['u', g.strideX],
    ]);
    this.dispatch('copyRegion', [params, s.buffer, d.buffer], g, [params]);
  }

  async read(name: string, offset = 0, length?: number) {
    const t = this.get(name);
    const n = t.shape[0] * t.shape[1];
    const len = length ?? n - offset;
    if (offset < 0 || offset + len > n) throw new RangeError(`${name}: read past the end`);
    return new Float32Array(await this.readBuffer(t.buffer, offset * 4, len * 4));
  }

  async write(name: string, data: Float32Array) {
    const t = this.get(name);
    if (data.length !== t.shape[0] * t.shape[1]) throw new Error(`${name}: wrong length`);
    if (data.length) this.device.queue.writeBuffer(t.buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  }

  async stats(name: string): Promise<Stats> {
    const t = this.get(name);
    const n = t.shape[0] * t.shape[1];
    const groups = Math.max(1, Math.min(256, Math.ceil(n / 256)));
    const params = this.uniform([['u', n]]);
    const out = this.storage(groups * 8 * 4, 'stats');
    this.dispatch('stats', [params, t.buffer, out], { x: groups }, [params]);
    const partials = new Float32Array(await this.readBuffer(out, 0, groups * 32));
    out.destroy();
    const s: Stats = { count: n, sum: 0, sumSq: 0, min: Infinity, max: -Infinity, zeros: 0, nonFinite: 0 };
    for (let g = 0; g < groups; g++) {
      const o = g * 8;
      s.sum += partials[o];
      s.sumSq += partials[o + 1];
      s.min = Math.min(s.min, partials[o + 2]);
      s.max = Math.max(s.max, partials[o + 3]);
      s.zeros += partials[o + 4];
      s.nonFinite += partials[o + 5];
    }
    if (s.nonFinite === n) {
      s.min = Infinity;
      s.max = -Infinity;
    }
    return s;
  }

  async rowNorms(name: string) {
    const t = this.get(name);
    const [rows, cols] = t.shape;
    const g = grid(rows);
    const params = this.uniform([['u', rows], ['u', cols], ['u', g.strideX]]);
    const out = this.storage(rows * 4, 'rowNorms');
    this.dispatch('rowNorms', [params, t.buffer, out], g, [params]);
    const norms = new Float32Array(await this.readBuffer(out, 0, rows * 4));
    out.destroy();
    return norms;
  }

  async outliers(name: string, mean: number, std: number, z: number, cap: number): Promise<Hits> {
    const t = this.get(name);
    const n = t.shape[0] * t.shape[1];
    const g = grid(n);
    const params = this.uniform([
      ['u', n], ['u', cap], ['u', g.strideX], ['f', mean], ['f', std > 0 ? 1 / std : 3.4e38], ['f', z],
    ]);
    const oi = this.storage(cap * 4, 'outlier-idx');
    const ov = this.storage(cap * 4, 'outlier-val');
    const count = this.storage(new Uint32Array(4), 'outlier-count');
    this.dispatch('outliers', [params, t.buffer, oi, ov, count], g, [params]);
    const total = new Uint32Array(await this.readBuffer(count, 0, 4))[0];
    const k = Math.min(total, cap);
    const indices = new Uint32Array(await this.readBuffer(oi, 0, k * 4));
    const values = new Float32Array(await this.readBuffer(ov, 0, k * 4));
    for (const b of [oi, ov, count]) b.destroy();
    return { indices, values, total };
  }

  async scatter(name: string, indices: Uint32Array, values: Float32Array, mode: 'set' | 'add') {
    const t = this.get(name);
    const len = t.shape[0] * t.shape[1];
    for (const i of indices) if (i >= len) throw new RangeError(`${name}: index ${i} out of range`);
    if (!indices.length) return;
    const g = grid(indices.length);
    const params = this.uniform([['u', indices.length], ['u', mode === 'set' ? 0 : 1], ['u', len], ['u', g.strideX]]);
    const idx = this.storage(indices, 'scatter-idx');
    const vals = this.storage(values, 'scatter-val');
    this.dispatch('scatter', [params, idx, vals, t.buffer], g, [params, idx, vals]);
  }

  async fill(name: string, r: Region, fill: Fill) {
    const t = this.get(name);
    checkRegion(t.shape, r, name);
    const cols = r.colEnd - r.colStart;
    const n = (r.rowEnd - r.rowStart) * cols;
    if (!n) return;
    const g = grid(n);
    const [mode, seed, a, b] =
      fill.kind === 'constant' ? [0, 0, fill.value, 0]
      : fill.kind === 'gaussian' ? [1, fill.seed, fill.mean, fill.std]
      : fill.kind === 'add_gaussian' ? [2, fill.seed, 0, fill.std]
      : [3, 0, fill.factor, 0];
    const params = this.uniform([
      ['u', t.shape[1]], ['u', r.rowStart], ['u', r.colStart], ['u', cols], ['u', n],
      ['u', mode], ['u', seed], ['u', g.strideX], ['f', a], ['f', b],
    ]);
    this.dispatch('fill', [params, t.buffer], g, [params]);
  }

  async flush() {
    await this.device.queue.onSubmittedWorkDone();
  }

  destroy() {
    for (const t of this.tensors.values()) t.buffer.destroy();
    this.tensors.clear();
    this.device.destroy();
  }

  createRunner(config: ModelConfig): Runner {
    return new GpuRunner(this, config);
  }

  createTrainer(config: ModelConfig, length: number, targets: TrainTarget[]) {
    return new GpuTrainer(this, config, length, targets);
  }
}

interface Dispatch {
  kernel: KernelName;
  group: GPUBindGroup;
  x: number;
  y: number;
}

/**
 * The forward pass as a fixed list of dispatches, built once per model shape:
 * per token the CPU only writes the 16-byte Step uniform (token, position,
 * target, track) and submits.
 */
class GpuRunner implements Runner {
  private readonly be: WebGpuBackend;
  private readonly c: ModelConfig;
  private owned: GPUBuffer[] = [];
  private step: GPUBuffer;
  private logits: GPUBuffer;
  private loss: GPUBuffer;
  private acts: GPUBuffer[];
  private body: Dispatch[] = [];
  private head: Dispatch[] = [];
  private trackedTokens = 0;

  constructor(be: WebGpuBackend, c: ModelConfig) {
    this.be = be;
    this.c = c;
    const own = (b: GPUBuffer) => (this.owned.push(b), b);
    const f32 = (n: number, label: string) => own(be.storage(n * 4, label));
    const uni = (params: Param[]) => own(be.uniform(params));
    const maxQ = Math.max(1, ...c.layers.map((l) => l.nHeads * c.headSize));
    const maxHeads = Math.max(1, ...c.layers.map((l) => l.nHeads));
    const maxHidden = Math.max(1, ...c.layers.map((l) => l.hidden));
    const kv = kvDim(c);

    this.step = own(be.createBuffer(16, U.UNIFORM | U.COPY_DST, 'step'));
    const x = f32(c.dim, 'x');
    const xb = f32(c.dim, 'xb');
    const q = f32(maxQ, 'q');
    const ao = f32(maxQ, 'attn-out');
    const att = f32(maxHeads * c.seqLen, 'att');
    const hb = f32(maxHidden, 'hb');
    const hb2 = f32(maxHidden, 'hb2');
    this.logits = f32(c.vocabSize, 'logits');
    this.loss = f32(4, 'loss');
    this.acts = c.layers.map((l, i) => f32(l.hidden, `acts-${i}`));

    const add = (list: Dispatch[], kernel: KernelName, buffers: GPUBuffer[], x: number, y = 1) => {
      if (x > 0) list.push({ kernel, group: be.bindGroup(kernel, buffers), x, y });
    };
    const matvec = (list: Dispatch[], w: GPUBuffer, input: GPUBuffer, out: GPUBuffer, rows: number, cols: number, accumulate: boolean, posStride = 0) => {
      if (!rows) return;
      const g = grid(rows);
      const params = uni([['u', rows], ['u', cols], ['u', accumulate ? 1 : 0], ['u', posStride], ['u', g.strideX]]);
      add(list, 'matvec', [params, this.step, w, input, out], g.x, g.y);
    };
    const rmsnorm = (list: Dispatch[], input: GPUBuffer, w: GPUBuffer, out: GPUBuffer) =>
      add(list, 'rmsnorm', [uni([['u', c.dim], ['f', c.normEps]]), input, w, out], 1);

    add(this.body, 'embed', [uni([['u', c.dim]]), this.step, be.buffer('tok_embeddings'), x], Math.ceil(c.dim / 64));
    c.layers.forEach((layer, l) => {
      const w = (s: string) => be.buffer(`layer_${l}.${s}`);
      const qd = layer.nHeads * c.headSize;
      const kc = f32(c.seqLen * kv, `kcache-${l}`);
      const vc = f32(c.seqLen * kv, `vcache-${l}`);
      rmsnorm(this.body, x, w('attn_norm'), xb);
      matvec(this.body, w('attn.wq'), xb, q, qd, c.dim, false);
      matvec(this.body, w('attn.wk'), xb, kc, kv, c.dim, false, kv);
      matvec(this.body, w('attn.wv'), xb, vc, kv, c.dim, false, kv);
      add(this.body, 'rope', [uni([['u', qd], ['u', kv], ['u', c.headSize], ['f', c.ropeTheta]]), this.step, q, kc], Math.ceil(Math.max(qd, kv) / 2 / 64));
      if (layer.nHeads) {
        const headKv = own(be.storage(Uint32Array.from(layer.headKv), `headKv-${l}`));
        add(this.body, 'attention', [uni([['u', c.headSize], ['u', kv], ['u', c.seqLen]]), this.step, q, kc, vc, headKv, att, ao], layer.nHeads);
        matvec(this.body, w('attn.wo'), ao, x, c.dim, qd, true);
      }
      if (layer.hidden) {
        rmsnorm(this.body, x, w('mlp_norm'), xb);
        matvec(this.body, w('mlp.gate'), xb, hb, layer.hidden, c.dim, false);
        matvec(this.body, w('mlp.up'), xb, hb2, layer.hidden, c.dim, false);
        add(this.body, 'swiglu', [uni([['u', layer.hidden]]), this.step, hb, hb2, this.acts[l]], Math.ceil(layer.hidden / 64));
        matvec(this.body, w('mlp.down'), hb, x, c.dim, layer.hidden, true);
      }
    });
    rmsnorm(this.head, x, be.buffer('norm'), xb);
    matvec(this.head, be.buffer(classifierName(c)), xb, this.logits, c.vocabSize, c.dim, false);
    add(this.head, 'nll', [uni([['u', c.vocabSize]]), this.step, this.logits, this.loss], 1);
  }

  private encode(pass: GPUComputePassEncoder, list: Dispatch[]) {
    for (const d of list) {
      pass.setPipeline(this.be.pipeline(d.kernel));
      pass.setBindGroup(0, d.group);
      pass.dispatchWorkgroups(d.x, d.y);
    }
  }

  async run(tokens: number[], start: number, options: RunOptions = {}) {
    const { device } = this.be;
    if (options.targets) device.queue.writeBuffer(this.loss, 0, new Float32Array(4));
    const step = new ArrayBuffer(16);
    const view = new DataView(step);
    for (let t = 0; t < tokens.length; t++) {
      const pos = start + t;
      if (pos >= this.c.seqLen) throw new RangeError(`Position ${pos} is past the model's context (${this.c.seqLen})`);
      const token = tokens[t];
      if (!(token >= 0 && token < this.c.vocabSize)) throw new RangeError(`Token ${token} is outside the vocabulary`);
      const target = options.targets?.[t] ?? -1;
      view.setUint32(0, token, true);
      view.setUint32(4, pos, true);
      view.setInt32(8, target, true);
      view.setUint32(12, options.track ? 1 : 0, true);
      device.queue.writeBuffer(this.step, 0, step);
      const encoder = device.createCommandEncoder({ label: `token ${pos}` });
      const pass = encoder.beginComputePass();
      this.encode(pass, this.body);
      if (target >= 0 || (options.logits && t === tokens.length - 1)) this.encode(pass, this.head);
      pass.end();
      device.queue.submit([encoder.finish()]);
      if (options.track) this.trackedTokens++;
    }
    const result: { logits?: Float32Array; nll?: { sum: number; count: number } } = {};
    if (options.logits) result.logits = new Float32Array(await this.be.readBuffer(this.logits, 0, this.c.vocabSize * 4));
    if (options.targets) {
      const loss = new Float32Array(await this.be.readBuffer(this.loss, 0, 8));
      result.nll = { sum: loss[0], count: loss[1] };
    }
    return result;
  }

  async activations() {
    const n = Math.max(1, this.trackedTokens);
    const layers = await Promise.all(
      this.acts.map(async (b, l) => new Float32Array(await this.be.readBuffer(b, 0, this.c.layers[l].hidden * 4)).map((v) => v / n))
    );
    return { tokens: this.trackedTokens, layers };
  }

  resetActivations() {
    this.acts.forEach((b, l) => this.be.device.queue.writeBuffer(b, 0, new Float32Array(Math.max(1, this.c.layers[l].hidden))));
    this.trackedTokens = 0;
  }

  destroy() {
    for (const b of this.owned) b.destroy();
    this.owned = [];
  }
}
